
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Pipeline File Architecture Environment
IN_PATH = Path("results/evcs_real_features.csv")
OUT_PATH = Path("results/evcs_with_predictions.csv")
MODEL_PATH = Path("results/demand_model_ensemble.joblib")
FEATURE_IMPORTANCE_PATH = Path("results/feature_importance.csv")
COMPARISON_PLOT = Path("results/model_comparison.png")
LEARNING_CURVE_PLOT = Path("results/learning_curve.png")
RESIDUAL_PLOT = Path("results/residual_plot.png")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def test_ann_for_comparison(X_train_scaled, X_test_scaled, y_train, y_test, input_dim):
  
   # ANN architecture just Testing
    from sklearn.neural_network import MLPRegressor
    
    ann = MLPRegressor(
        hidden_layer_sizes=(8, 6, 4),
        activation='relu',
        solver='adam',
        alpha=0.01,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
    
    n_params = (input_dim * 8 + 8) + (8 * 6 + 6) + (6 * 4 + 4) + (4 * 1 + 1)
    
    ann.fit(X_train_scaled, y_train)
    
    y_train_pred = ann.predict(X_train_scaled)
    y_test_pred = ann.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    return ann, train_r2, test_r2, test_mae, y_train_pred, y_test_pred


def create_comparison_plot(y_test, predictions_dict, output_path):
   #Generates comparative diagnostic visualizations of candidate model fits. testing only
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = {'ANN': '#e74c3c', 'Ridge': '#f39c12', 'Random Forest': '#27ae60', 'Ensemble': '#2980b9'}
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        if idx >= 4: break
        ax = axes[idx]
        ax.scatter(y_test, y_pred, alpha=0.6, color=colors.get(model_name, '#7f8c8d'), s=50)
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Ideal Fit Line')
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        ax.set_xlabel('Actual Spatial EV Density (EVs/km²)', fontsize=10)
        ax.set_ylabel('Estimated Spatial EV Density (EVs/km²)', fontsize=10)
        ax.set_title(f'{model_name}\nR² = {r2:.3f} | MAE = {mae:.1f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
        
    plt.suptitle('Empirical Model Evaluation Matrix: Vilnius EV Density Projections', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curve(model, X, y, model_name, output_path):
  #  Traces architectural learning metrics against varying scale baselines
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2', random_state=RANDOM_STATE, n_jobs=-1
    )
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score Baseline', color='#2980b9')
    plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-Validation Performance', color='#27ae60')
    plt.xlabel('Training Record Volume Extent')
    plt.ylabel('Explanatory Score (R²)')
    plt.title(f'Structural Model Diagnostics Learning Curve: {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_residuals(y_test, y_pred, model_name, output_path):
  #Evaluates error variance maps for structural homoscedasticity
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    axes[0].scatter(y_pred, residuals, alpha=0.6, color='#2980b9')
    axes[0].axhline(y=0, color='#c0392b', linestyle='--')
    axes[0].set_xlabel('Predicted Structural Values (EVs/km²)')
    axes[0].set_ylabel('Model Variance Error Residuals')
    axes[0].set_title('Residual Spatial Heteroscedasticity Analysis')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=12, edgecolor='black', alpha=0.7, color='#34495e')
    axes[1].set_xlabel('Error Delta Residual Bounds')
    axes[1].set_ylabel('Observation Frequency')
    axes[1].set_title('Error Skew Distribution Map')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    from scipy import stats
    _, shapiro_p = stats.shapiro(residuals)
    return shapiro_p


def check_spatial_autocorrelation(y_test, y_pred, df_test):
  #  Analyzes residuals for remaining spatial pattern dependencies
    from scipy.spatial.distance import cdist
    residuals = np.abs(y_test - y_pred)
    coords = df_test[['lat', 'lon']].values
    dist_matrix = cdist(coords, coords)
    
    print("\n Residual Spatial Autocorrelation Diagnosis:")
    near_mask = (dist_matrix > 0) & (dist_matrix <= 1500)
    far_mask = (dist_matrix > 5000)
    
    if near_mask.any() and far_mask.any():
        near_error_diff = np.mean([np.abs(residuals[i] - residuals[j]) for i, j in zip(*np.where(near_mask))])
        far_error_diff = np.mean([np.abs(residuals[i] - residuals[j]) for i, j in zip(*np.where(far_mask))])
        print(f"  Avg Error Variance Delta (Nearby Zones ≤1.5km): {near_error_diff:.3f}")
        print(f"  Avg Error Variance Delta (Distant Zones ≥5.0km): {far_error_diff:.3f}")
        if abs(near_error_diff - far_error_diff) < 0.15:
            print("  Spatial Homoscedasticity Confirmed: Residual errors are distributed uniformly across space.")
        else:
            print(" Localized error variances exist. Minor unmodeled neighborhood traits may persist.")


def main():
 
    
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing upstream feature matrix layout: {IN_PATH}")
    
    df = pd.read_csv(IN_PATH)
    print(f"Feature array successfully linked. Candidate Zone Count: {len(df)}")
    
    # ENHANCEMENT 1: Spatial Lag Consistency Check
    print("\nSpatial Lag Feature Validation:")
    print(f"  lag_neighborhood_traffic range: [{df['lag_neighborhood_traffic'].min():.3f}, {df['lag_neighborhood_traffic'].max():.3f}]")
    print(f"  lag_neighborhood_retail range:  [{df['lag_neighborhood_retail'].min():.3f}, {df['lag_neighborhood_retail'].max():.3f}]")
    print(f"  Correlation traffic vs lag_traffic: {df['jam_score_norm'].corr(df['lag_neighborhood_traffic']):.3f}")
    
    y = df['ev_density'].values
    
    # Feature selections incorporating Inverse Distance Weighting (IDW) lags
    feature_cols = [
        'jam_score_norm',
        'district_pop_density',
        'working_share',
        'poi_density',
        'lag_neighborhood_traffic',
        'lag_neighborhood_retail',
        'lag_neighborhood_leisure'
    ]
    
    #  Multicollinearity Check via Native Scikit-Learn Regression Engine
    print("\nFeature Multi-Collinearity Diagnostic Matrix:")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_cols
    vif_scores = []
    
    for col in feature_cols:
        X_other = df[[c for c in feature_cols if c != col]].fillna(0).values
        y_col = df[col].fillna(0).values
        
        # Calculate R² of this feature regressed on all other features
        r2 = LinearRegression(fit_intercept=True).fit(X_other, y_col).score(X_other, y_col)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
        vif_scores.append(vif)
        
    vif_data["VIF Score"] = vif_scores
    print(vif_data.to_string(index=False))
    
    X = df[feature_cols].fillna(0).values
    
    # Split records into core modeling partitions
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    df_test = df.iloc[test_idx]
    
    # Scale feature distributions based on training set metrics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)
    
    # Baseline Model Evaluation
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train_scaled, y_train)
    dummy_pred = dummy.predict(X_test_scaled)
    dummy_r2 = r2_score(y_test, dummy_pred)
    dummy_mae = mean_absolute_error(y_test, dummy_pred)
    
    # Run the comparative ANN model benchmark
    ann, ann_train_r2, ann_test_r2, ann_test_mae, ann_train_pred, ann_test_pred = test_ann_for_comparison(
        X_train_scaled, X_test_scaled, y_train, y_test, len(feature_cols)
    )
    
    # Model 1 Configuration: Ridge Regularization
    
    print(" Fitting Model 1: Ridge Regression (L2 Regularized Matrix) Wait please")
 
    ridge = Ridge(alpha=1.5)
    cv_engine = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ridge_cv = cross_val_score(ridge, X_train_scaled, y_train, cv=cv_engine, scoring='r2')
    ridge.fit(X_train_scaled, y_train)
    ridge_test_pred = ridge.predict(X_test_scaled)
    
    # Model 2 Configuration: Random Forest Regularization
    print("Fitting Model 2: Random Forest Regressor (Non-Linear Ensemble) PLease wait")
    rf = RandomForestRegressor(n_estimators=60, max_depth=4, min_samples_split=4, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
    rf_cv = cross_val_score(rf, X_train_scaled, y_train, cv=cv_engine, scoring='r2')
    rf.fit(X_train_scaled, y_train)
    rf_test_pred = rf.predict(X_test_scaled)
    
    # Optimizing Ensemble Blending Parameters via Cross-Validation Folds
 
    print("Optimizing Ensemble Blending Parameters via Cross-Validation Folds please wait")
    
    
    best_cv_mse = float('inf')
    best_blend_weights = {'ridge': 0.5, 'rf': 0.5}
    best_ensemble_cv_r2 = 0.0
    
    oof_ridge_preds = np.zeros(len(X_train_scaled))
    oof_rf_preds = np.zeros(len(X_train_scaled))
    
    for train_fold_idx, val_fold_idx in cv_engine.split(X_train_scaled):
        X_f_tr, X_f_va = X_train_scaled[train_fold_idx], X_train_scaled[val_fold_idx]
        y_f_tr, y_f_va = y_train[train_fold_idx], y_train[val_fold_idx]
        
        r_mod = Ridge(alpha=1.5).fit(X_f_tr, y_f_tr)
        f_mod = RandomForestRegressor(n_estimators=60, max_depth=4, min_samples_split=4, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1).fit(X_f_tr, y_f_tr)
        
        oof_ridge_preds[val_fold_idx] = r_mod.predict(X_f_va)
        oof_rf_preds[val_fold_idx] = f_mod.predict(X_f_va)
        
    for w_ridge in np.linspace(0.0, 1.0, 11):
        w_rf = 1.0 - w_ridge
        blended_oof_preds = (w_ridge * oof_ridge_preds) + (w_rf * oof_rf_preds)
        mean_fold_mse = mean_squared_error(y_train, blended_oof_preds)
        
        if mean_fold_mse < best_cv_mse:
            best_cv_mse = mean_fold_mse
            best_blend_weights = {'ridge': round(w_ridge, 2), 'rf': round(w_rf, 2)}
            best_ensemble_cv_r2 = r2_score(y_train, blended_oof_preds)
            
    print(f"  Calibrated Blending Matrix Locked In: Ridge Ratio={best_blend_weights['ridge']} | RF Ratio={best_blend_weights['rf']}")
    
    # Evaluate performance metrics on the test partition
    y_test_pred_ensemble = (best_blend_weights['ridge'] * ridge_test_pred) + (best_blend_weights['rf'] * rf_test_pred)
    ensemble_test_r2 = r2_score(y_test, y_test_pred_ensemble)
    ensemble_test_mae = mean_absolute_error(y_test, y_test_pred_ensemble)
    
    # Print Summary Matrix
   
    print("STRUCTURAL MODEL PERFORMANCE MATRIX COMPARISON SUMMARY")
   
    comparison_df = pd.DataFrame({
        'Model Pipeline': [
            'Baseline (Mean Profile)', 
            'ANN Network (REJECTED)', 
            'Ridge Regression Model', 
            'Random Forest Tree Model', 
            'Optimized Ensemble System'
        ],
        'Cross-Val R²': [
            'N/A', 
            'N/A', 
            f'{ridge_cv.mean():.3f}', 
            f'{rf_cv.mean():.3f}', 
            f'{best_ensemble_cv_r2:.3f}'
        ],
        'Test Partition R²': [
            f'{dummy_r2:.3f}', 
            f'{ann_test_r2:.3f}', 
            f'{r2_score(y_test, ridge_test_pred):.3f}', 
            f'{r2_score(y_test, rf_test_pred):.3f}', 
            f'{ensemble_test_r2:.3f}'
        ],
        'Test Deviation MAE': [
            f'{dummy_mae:.2f}', 
            f'{ann_test_mae:.2f}', 
            f'{mean_absolute_error(y_test, ridge_test_pred):.2f}', 
            f'{mean_absolute_error(y_test, rf_test_pred):.2f}', 
            f'{ensemble_test_mae:.2f}'
        ]
    })
    print(comparison_df.to_string(index=False))
    
    #  Bootstrap Test Performance Confidence Interval Bounds
    n_bootstrap = 1000
    bootstrap_r2 = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_test), len(y_test), replace=True)
        bootstrap_r2.append(r2_score(y_test[idx], y_test_pred_ensemble[idx]))
    print(f"\nEnsemble Test Performance Stability Map")
    print(f"  Generalization Score R²: {ensemble_test_r2:.3f} ± {1.96 * np.std(bootstrap_r2):.3f}")
    
    #  Residual Spatial Autocorrelation Testing Check
    check_spatial_autocorrelation(y_test, y_test_pred_ensemble, df_test)
    
    # Generate Visual Plots
    predictions_dict = {'ANN': ann_test_pred, 'Ridge': ridge_test_pred, 'Random Forest': rf_test_pred, 'Ensemble': y_test_pred_ensemble}
    create_comparison_plot(y_test, predictions_dict, COMPARISON_PLOT)
    
    from sklearn.base import clone
    plot_learning_curve(clone(rf), X_all_scaled, y, "Random Forest Base Component", LEARNING_CURVE_PLOT)
    shapiro_p = plot_residuals(y_test, y_test_pred_ensemble, "Optimized Ensemble System", RESIDUAL_PLOT)
    
    ridge_final = Ridge(alpha=1.5).fit(X_all_scaled, y)
    rf_final = RandomForestRegressor(n_estimators=60, max_depth=4, min_samples_split=4, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1).fit(X_all_scaled, y)
    
    #  Feature Importance Analysis
    feature_importance = pd.DataFrame({
        'Feature Name': feature_cols,
        'Relative Weight / Importance': rf_final.feature_importances_
    }).sort_values('Relative Weight / Importance', ascending=False)
    
    print("\nRandom Forest Structural Feature Importance Rankings:")
    for _, row in feature_importance.iterrows():
        print(f"   {row['Feature Name']}: {row['Relative Weight / Importance']:.4f}")
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    
    df['predicted_ev_density'] = (best_blend_weights['ridge'] * ridge_final.predict(X_all_scaled)) + (best_blend_weights['rf'] * rf_final.predict(X_all_scaled))
    df['predicted_ev_density'] = df['predicted_ev_density'].clip(0, None)
    
    # Generate normalized 0-1 infrastructure targets (demand_score matrix) for the Genetic Algorithm
    df['demand_score'] = df['predicted_ev_density'] / df['predicted_ev_density'].max()
    df['demand_score'] = df['demand_score'].clip(0, 1).fillna(0)
    
    print(f"\n Max Density Projections Peak at {df['predicted_ev_density'].max():.2f} EVs/km²")
    print(f"  Standard Target Vector 'demand_score' set for GA Pipeline Integration.")
    
    # Export Data Products to Disk
    df.sort_values('demand_score', ascending=False).to_csv(OUT_PATH, index=False)
    
    joblib.dump({
        'ridge_model': ridge_final, 'rf_model': rf_final, 'ensemble_weights': best_blend_weights,
        'scaler': scaler, 'features': feature_cols, 'test_r2': ensemble_test_r2
    }, MODEL_PATH)
    
    print(f"\n metrics saved successfully to results directory.")


if __name__ == "__main__":
    main()