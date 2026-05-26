
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import random
import matplotlib.pyplot as plt

# Strict Reproducibility
np.random.seed(42)
random.seed(42)

# File Paths
IN_PATH = Path("results/evcs_with_svm_classification.csv")
OUT_SELECTED = Path("results/ga_selected_sites_final.csv")
OUT_METRICS = Path("results/ga_metrics_final.txt")
CONVERGENCE_PLOT = Path("results/ga_convergence.png")

# GA Parameters
K_STATIONS = 20
GENERATIONS = 500
POP_SIZE = 500
MUTATION_RATE = 0.25
ELITE_SIZE = 5
TOURNAMENT_SIZE = 3

# Fitness Weights (sum to 1.0)
W_TRAFFIC = 0.35
W_DEMAND = 0.25      # Uses demand_score directly
W_POP = 0.15
W_SVM = 0.25

# Distance Constraints (meters)
MIN_STATION_DISTANCE = 400
MIN_EXISTING_DISTANCE = 300

# Diversity penalty
DIVERSITY_PENALTY_BASE = 0.10


def haversine(lat1, lon1, lat2, lon2):
  #  Calculate geodesic distance in meters using haversine formula
    R = 6371000
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def calculate_fitness(individual, candidates):
  
    sub = candidates.iloc[individual]
    
    # HARD CONSTRAINTS (Return -1e6 if violated)
    
    # 1. Distance to existing chargers
    if (sub['dist_to_existing_m'] < MIN_EXISTING_DISTANCE).any():
        return -1e6
    
    # 2. Distance between selected stations
    coords = sub[['lat', 'lon']].values
    if len(coords) > 1:
        min_distance = np.inf
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = haversine(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                if dist < min_distance:
                    min_distance = dist
                if dist < MIN_STATION_DISTANCE:
                    return -1e6
    

    # OBJECTIVE FUNCTION (Weighted sum using demand_score)
 
    
    raw_fitness = (
        W_TRAFFIC * sub['jam_score_norm'].mean() +
        W_DEMAND * sub['demand_score'].mean() +      
        W_POP * sub['pop_density_norm'].mean() +
        W_SVM * sub['svm_probability'].mean()
    )
    

    # DIVERSITY PENALTY (Discourage multiple stations in same district)

    
    seniunija_counts = sub['seniunija'].value_counts()
    diversity_penalty = sum((count - 1) * DIVERSITY_PENALTY_BASE 
                            for count in seniunija_counts if count > 1)
    

    # COVERAGE BONUS (Reward covering more unique districts)
  
    
    n_unique = len(seniunija_counts)
    if n_unique >= 15:
        coverage_bonus = 0.08
    elif n_unique >= 12:
        coverage_bonus = 0.06
    elif n_unique >= 10:
        coverage_bonus = 0.04
    elif n_unique >= 8:
        coverage_bonus = 0.02
    else:
        coverage_bonus = 0.0
    
    # Final fitness
    fitness = raw_fitness * (1 - diversity_penalty) + coverage_bonus
    
    return fitness


def greedy_selection(candidates, K):
    
    #Baseline greedy algorithm for comparison.
    # Calculate composite score (geometric mean using demand_score)
    candidates['composite'] = (
        candidates['jam_score_norm'] * 
        candidates['demand_score'] *          
        candidates['svm_probability']
    ) ** (1/3)
    
    sorted_df = candidates.sort_values('composite', ascending=False)
    
    selected = []
    selected_coords = []
    
    for idx, row in sorted_df.iterrows():
        if len(selected) >= K:
            break
        
        # Check distance to existing chargers
        if row['dist_to_existing_m'] < MIN_EXISTING_DISTANCE:
            continue
        
        # Check distance to already selected stations
        valid = True
        for lat, lon in selected_coords:
            if haversine(row['lat'], row['lon'], lat, lon) < MIN_STATION_DISTANCE:
                valid = False
                break
        
        if valid:
            selected.append(idx)
            selected_coords.append((row['lat'], row['lon']))
    
    return selected


def tournament_selection(population, fitness_scores, tournament_size=TOURNAMENT_SIZE):
  
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    winner_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
    return population[winner_idx].copy()


def crossover(parent1, parent2, n_candidates, K):
   
    child = np.unique(np.concatenate([parent1, parent2]))
    
    if len(child) < K:
        available = np.setdiff1d(np.arange(n_candidates), child)
        needed = K - len(child)
        if len(available) >= needed:
            child = np.concatenate([child, np.random.choice(available, needed, replace=False)])
    elif len(child) > K:
        child = np.random.choice(child, K, replace=False)
    
    return np.sort(child)


def mutate(individual, n_candidates, K, mutation_rate, candidates):
   
   # Mutation with district exploration.
  
    if np.random.random() < mutation_rate:
        if 'composite' not in candidates.columns:
            candidates['composite'] = (
                candidates['jam_score_norm'] * 
                candidates['demand_score'] * 
                candidates['svm_probability']
            ) ** (1/3)
        
        current_districts = set(candidates.iloc[individual]['seniunija'])
        all_districts = set(candidates['seniunija'].unique())
        missing_districts = all_districts - current_districts
        
        if missing_districts and len(missing_districts) > 0:
            missing_candidates = candidates[candidates['seniunija'].isin(missing_districts)]
            if len(missing_candidates) > 0:
                current_scores = candidates.iloc[individual]['composite'].values
                worst_idx_pos = np.argmin(current_scores)
                best_missing_idx = missing_candidates['composite'].idxmax()
                individual[worst_idx_pos] = best_missing_idx
                individual = np.sort(np.unique(individual))
        
        if len(individual) == K and np.random.random() < 0.3:
            idx_to_replace = np.random.randint(0, len(individual))
            available_indices = np.setdiff1d(np.arange(n_candidates), individual)
            if len(available_indices) > 0:
                individual[idx_to_replace] = np.random.choice(available_indices)
                individual = np.sort(np.unique(individual))
    
    if len(individual) < K:
        available = np.setdiff1d(np.arange(n_candidates), individual)
        needed = K - len(individual)
        if len(available) >= needed:
            individual = np.concatenate([individual, np.random.choice(available, needed, replace=False)])
            individual = np.sort(individual)
    elif len(individual) > K:
        individual = np.random.choice(individual, K, replace=False)
        individual = np.sort(individual)
    
    return individual


def main():

    
    # Load data
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run 08_ensemble_model.py first.")
    
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} candidate zones")
    
    # Verify required columns
    required_cols = ['jam_score_norm', 'demand_score', 'district_pop_density', 
                     'svm_probability', 'seniunija', 'dist_to_existing_m', 'lat', 'lon']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return
    
    # Check demand_score range
    print(f"\ndemand_score statistics:")
    print(f"  Min: {df['demand_score'].min():.4f}")
    print(f"  Max: {df['demand_score'].max():.4f}")
    print(f"  Mean: {df['demand_score'].mean():.4f}")
    print(f"  Zones with zero demand: {(df['demand_score'] == 0).sum()}")
    
    # Normalize population density (0-1 scale)
    pop_min = df['district_pop_density'].min()
    pop_max = df['district_pop_density'].max()
    df['pop_density_norm'] = (df['district_pop_density'] - pop_min) / (pop_max - pop_min + 1e-9)
    
    # Filter candidates by distance to existing chargers
    valid_mask = df['dist_to_existing_m'] >= MIN_EXISTING_DISTANCE
    valid_zones = df[valid_mask].copy()
    print(f"\nCandidate Filtering:")
    print(f"  Total zones: {len(df)}")
    print(f"  After distance filter (≥{MIN_EXISTING_DISTANCE}m): {len(valid_zones)}")
    
    # Score threshold filter (top 70% by composite score)
    valid_zones['composite'] = (
        valid_zones['jam_score_norm'] * 
        valid_zones['demand_score'] * 
        valid_zones['svm_probability']
    ) ** (1/3)
    
    threshold = valid_zones['composite'].quantile(0.30)
    candidates = valid_zones[valid_zones['composite'] >= threshold].reset_index(drop=True)
    
    print(f"  After score threshold (top 70%): {len(candidates)}")
    
    # Adjust K if needed
    K = min(K_STATIONS, len(candidates))
    print(f"\n  Selecting {K} stations from {len(candidates)} candidates")
    print(f"  Min distance between stations: {MIN_STATION_DISTANCE}m")
    
    greedy_indices = greedy_selection(candidates, K)
    if len(greedy_indices) == K:
        greedy_fitness = calculate_fitness(greedy_indices, candidates)
        print(f"  Greedy Fitness: {greedy_fitness:.4f}")
    else:
        greedy_fitness = -1e6
        print(f"  Greedy only selected {len(greedy_indices)}/{K} stations")
    
   
    # GENETIC ALGORITHM
    print(f"  Generations: {GENERATIONS}")
    print(f"  Population Size: {POP_SIZE}")
    print(f"  Mutation Rate: {MUTATION_RATE}")
    print(f"  Elite Size: {ELITE_SIZE}")
    
    n_candidates = len(candidates)
    
    # Initialize population
    population = []
    for _ in range(POP_SIZE):
        selected = np.random.choice(n_candidates, K, replace=False)
        population.append(np.sort(selected))
    
    # Evolution tracking
    history = []
    best_individual = None
    best_fitness = -1e9
    no_improvement_count = 0
    EARLY_STOPPING_GENERATIONS = 100
    EARLY_STOPPING_TOLERANCE = 0.001
    
    print("\n  Progress:")
    
    for gen in range(GENERATIONS):
        fitness_scores = np.array([calculate_fitness(ind, candidates) for ind in population])
        
        gen_best_fitness = fitness_scores.max()
        history.append(gen_best_fitness)
        
        if gen_best_fitness > best_fitness + EARLY_STOPPING_TOLERANCE:
            best_fitness = gen_best_fitness
            best_individual = population[np.argmax(fitness_scores)].copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if gen % 100 == 0:
            print(f"    Gen {gen:3d}: Best = {best_fitness:.4f}")
        
        if no_improvement_count >= EARLY_STOPPING_GENERATIONS:
            print(f"\n  Early stopping at generation {gen}")
            break
        
        # Elitism
        elite_indices = np.argsort(fitness_scores)[-ELITE_SIZE:]
        new_population = [population[idx].copy() for idx in elite_indices]
        
        # Fill rest
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child = crossover(parent1, parent2, n_candidates, K)
            child = mutate(child, n_candidates, K, MUTATION_RATE, candidates)
            new_population.append(child)
        
        population = new_population
    
    print(f"\n  Final Best Fitness: {best_fitness:.4f}")
    
   
    # RESULTS
    print("OPTIMIZATION RESULTS")
 
    
    selected_sites = candidates.iloc[best_individual].copy()
    selected_sites['priority'] = range(1, len(selected_sites) + 1)
    selected_sites = selected_sites.sort_values('priority')
    
    unique_districts = selected_sites['seniunija'].nunique()
    district_counts = selected_sites['seniunija'].value_counts()
    
    print(f"\n  GA Best Fitness:     {best_fitness:.4f}")
    print(f"  Greedy Baseline:     {greedy_fitness:.4f}")
    print(f"  Improvement:         +{best_fitness - greedy_fitness:.4f}")
    print(f"  Unique Districts:    {unique_districts} / 21")
    
    print(f"\n  Average Metrics:")
    print(f"    Traffic Score:     {selected_sites['jam_score_norm'].mean():.4f}")
    print(f"    Demand Score:      {selected_sites['demand_score'].mean():.4f}")
    print(f"    Population Density: {selected_sites['pop_density_norm'].mean():.4f}")
    print(f"    SVM Probability:   {selected_sites['svm_probability'].mean():.4f}")
    
    print(f"\n  District Distribution (top 10):")
    for district, count in district_counts.head(10).items():
        print(f"    {district}: {count} station(s)")
    
    # Save results
    selected_sites.to_csv(OUT_SELECTED, index=False)
    
    # Save metrics
    with open(OUT_METRICS, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GA OPTIMIZATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PARAMETERS:\n")
        f.write(f"  Stations Selected: {K}\n")
        f.write(f"  Generations: {GENERATIONS}\n")
        f.write(f"  Population Size: {POP_SIZE}\n")
        f.write(f"  Mutation Rate: {MUTATION_RATE}\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write(f"  GA Best Fitness: {best_fitness:.4f}\n")
        f.write(f"  Greedy Baseline: {greedy_fitness:.4f}\n")
        f.write(f"  Improvement: {best_fitness - greedy_fitness:.4f}\n\n")
        
        f.write("COVERAGE:\n")
        f.write(f"  Unique Districts: {unique_districts} / 21\n")
        for district, count in district_counts.items():
            f.write(f"    {district}: {count}\n")
        f.write("\n")
        
        f.write("AVERAGE METRICS:\n")
        f.write(f"  Traffic Score: {selected_sites['jam_score_norm'].mean():.4f}\n")
        f.write(f"  Demand Score: {selected_sites['demand_score'].mean():.4f}\n")
        f.write(f"  SVM Probability: {selected_sites['svm_probability'].mean():.4f}\n")
    
    # Convergence plot
    plt.figure(figsize=(12, 6))
    plt.plot(history, linewidth=2, color='midnightblue')
    plt.axhline(y=greedy_fitness, color='red', linestyle='--', 
                linewidth=1.5, label=f'Greedy Baseline: {greedy_fitness:.4f}')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title('Genetic Algorithm Convergence Profile', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONVERGENCE_PLOT, dpi=150)
    plt.close()
    
    print(f"\n Output files:")
    print(f"  Selected sites: {OUT_SELECTED}")
    print(f"  Metrics: {OUT_METRICS}")
    print(f"  Convergence plot: {CONVERGENCE_PLOT}")
    
    # Preview top sites
    print(f"\nTop ten selected sites:")
    preview_cols = ['cluster', 'seniunija', 'demand_score', 'svm_probability', 'priority']
    print(selected_sites[preview_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()