
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

IN_PATH = Path("results/evcs_with_predictions.csv")
OUT_PATH = Path("results/evcs_with_svm_classification.csv")
MODEL_PATH = Path("results/svm_model.joblib")

# Spatial presence and pseudo-absence boundaries
PRESENCE_DISTANCE_MAX = 150       # ≤ 150m from existing EVCS 
ABSENCE_DISTANCE_MIN = 450        # ≥ 450m from existing EVCS 

RANDOM_STATE = 42
TEST_SIZE = 0.20

def main():
    print("EMPIRICAL SVM SITE CLASSIFICATION (REVEALED PREFERENCES)")
   
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input features file: {IN_PATH}.")
        
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} candidate zones across Vilnius")
    
    # 1. LABELLING ENGINE: Defining Presence vs. Background Noise
    df['svm_label'] = -1  # Default value for unassigned buffer zones
    
    # Assign Class 1 (Presence)
    df.loc[df['dist_to_existing_m'] <= PRESENCE_DISTANCE_MAX, 'svm_label'] = 1
    # Assign Class 0 (Pseudo-Absence Background)
    df.loc[df['dist_to_existing_m'] >= ABSENCE_DISTANCE_MIN, 'svm_label'] = 0
    
    # Isolate valid training rows (excluding the spatial buffer zone)
    train_df = df[df['svm_label'].isin([0, 1])].copy()
    
    # Calculate empirical imbalance ratio
    n_pos = (train_df['svm_label'] == 1).sum()
    n_neg = (train_df['svm_label'] == 0).sum()
    imbalance_ratio = n_pos / n_neg if n_neg > 0 else 1.0
    
    print(f"\n Revealed Preference Class Allocation:")
    print(f"  • Confirmed Presences (Class 1):        {n_pos}")
    print(f"  • Pseudo-Absences Background (Class 0): {n_neg}")
    print(f"  • Excluded Spatial Buffer Zones:        {(df['svm_label'] == -1).sum()}")
    print(f"  • Imbalance Ratio (Pos/Neg):            {imbalance_ratio:.3f}")
    
    # FEATURE SELECTION: Removed 'dist_to_existing_m' to prevent absolute data leakage
    feature_cols = [
        'poi_density', 'education', 'leisure', 'retail', 'transport', 'work',
        'district_pop_density', 'working_share', 'jam_score_norm'
    ]
    
    # Extract training data
    X_train_full = train_df[feature_cols].fillna(0)
    y_train_full = train_df['svm_label']
    
    # Stratified Split for robust holdout validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_train_full
    )
    
    print(f"\nModel Training Dataset Slices:")
    print(f"  Train: {len(X_train)} samples (Presences={y_train.sum()})")
    print(f"  Test:  {len(X_test)} samples (Presences={y_test.sum()})")
    
    # RBF Support Vector Classifier with balanced class weights
    svm_pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=3.0, class_weight='balanced', 
            probability=True, max_iter=10000, random_state=RANDOM_STATE)
    )
    
    # Cross-validation validation footprint
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=cv, scoring='f1')
    print(f"\n 5-Fold Cross-Validation F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Fit the production model on the training set
    svm_pipeline.fit(X_train, y_train)
    
    # Performance Evaluation
    y_test_pred = svm_pipeline.predict(X_test)
    y_train_pred = svm_pipeline.predict(X_train)
    
    print(f"\nCore Evaluation Metrics:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_test_pred):.4f}")
    
    # Generalization Integrity Check
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    print(f"\n Generalization Integrity Check:")
    print(f"  Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f} (Gap: {train_f1 - test_f1:.4f})")
    
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix (Holdout Slice):")
    print(f"  TN (True Background):   {cm[0,0]:3d}  |  FP (False Positives):  {cm[0,1]:3d}")
    print(f"  FN (Missed Signatures): {cm[1,0]:3d}  |  TP (True Matches):      {cm[1,1]:3d}")
    
    # Predict probabilities across ALL 201 urban zones in Vilnius
    X_all = df[feature_cols].fillna(0)
    df['svm_probability'] = svm_pipeline.predict_proba(X_all)[:, 1]
    df['svm_prediction'] = svm_pipeline.predict(X_all)
    
    # Save processed outputs and trained model assets
    df.to_csv(OUT_PATH, index=False)
    joblib.dump(svm_pipeline, MODEL_PATH)
    print(f"\nSaved to: {OUT_PATH}")
    
    # 5. Filter for unbuilt lookup sites clear of existing infrastructure
    print("\Top 10 High potential new sites (Not Existing Stations):")
    new_sites = df[df['dist_to_existing_m'] >= 300]
    display_cols = ['cluster', 'seniunija', 'dist_to_existing_m', 'svm_probability', 'demand_score']
    
    top_new_sites = new_sites.sort_values('svm_probability', ascending=False).head(10)
    print(top_new_sites[display_cols].to_string(index=False))
    
    print("\nSVM classification complete!")


if __name__ == "__main__":
    main()