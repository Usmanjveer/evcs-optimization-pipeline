import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial import ConvexHull
from pathlib import Path

# File Registry Environment
POIS_CLEAN = Path("results/vilnius_pois_clean.csv")
POIS_CLUSTERED = Path("results/vilnius_pois_clustered.csv")
CANDIDATE_ZONES = Path("results/evcs_candidate_zones.csv")

# Global Hyperparameters
DBSCAN_GLOBAL_EPS = 300     # Tier 1: which is considered as 5-minute urban walking capture radius (meters)
DBSCAN_MICRO_EPS = 125      # Tier 2: Micro-radius to shatter contiguous giant components (meters)
DBSCAN_MIN_SAMPLES = 5      # Density threshold baseline
EARTH_RADIUS_M = 6371000    # Mean radius for spherical Haversine map projection


def calculate_true_cluster_area(group):
  
    if len(group) < 3:
        # Fallback to local walkability neighborhood area if coordinates are linear or tight
        return np.pi * ((DBSCAN_GLOBAL_EPS / 1000.0) ** 2)
        
    points = group[['lon', 'lat']].values
    try:
        hull = ConvexHull(points)
        # Spatial scaling matrix: Convert coordinate degrees to precise localized kilometers.
        # At 54.7°N (Vilnius), 1 Latitude degree ≈ 111.32 km, 1 Longitude degree is shrunk by the cosine scale.
        lat_scale = 111.32
        lon_scale = 111.32 * np.cos(np.radians(group['lat'].mean()))
        
        area_km2 = hull.volume * lat_scale * lon_scale
        
        return max(area_km2, 0.005)
    except:
        # Mathematical geometry fallback standard
        return np.pi * ((DBSCAN_GLOBAL_EPS / 1000.0) ** 2)


def main():
    if not POIS_CLEAN.exists():
        raise FileNotFoundError(f"Missing upstream data layer: {POIS_CLEAN}")
    
    print(f"Configuration Profile: Global ε={DBSCAN_GLOBAL_EPS}m | Micro-Splitting ε={DBSCAN_MICRO_EPS}m | Min Points={DBSCAN_MIN_SAMPLES}")
    
    poi = pd.read_csv(POIS_CLEAN)
    poi = poi.dropna(subset=["lat", "lon"])
    print(f"Total destination vertices queued for alignment: {len(poi):,}")
    
  
    # ADAPTIVE MULTI-TIER SPATIAL PARTITIONING
    # Breaking up the spatial chaining artifact where cities collapse into a giant cluster.
    
    print("\nExecuting Tier-1 Global Spatial Neighborhood Scan...")
    coords_rad = np.radians(poi[["lat", "lon"]].values)
    global_eps_rad = DBSCAN_GLOBAL_EPS / EARTH_RADIUS_M
    
    db_global = DBSCAN(eps=global_eps_rad, min_samples=DBSCAN_MIN_SAMPLES, metric="haversine", algorithm="ball_tree")
    poi["cluster"] = db_global.fit_predict(coords_rad)
    
    # Here we will Analyze macro-cluster saturation (Threshold set at 10% of global dataset entries)
    macro_limit = len(poi) * 0.10
    label_distribution = poi["cluster"].value_counts()
    giant_components = label_distribution[(label_distribution > macro_limit) & (label_distribution.index != -1)].index.tolist()
    
    if giant_components:
        print(f"  Detected {len(giant_components)} high-density macro-component")
        print(f"  Tier-2 Micro-Resolution Splitting Pass")
        
        current_max_cluster = poi["cluster"].max()
        
        for macro_id in giant_components:
            macro_mask = poi["cluster"] == macro_id
            macro_coords = np.radians(poi[macro_mask][["lat", "lon"]].values)
            
            # Tier-2 High-Resolution Processing Pass
            micro_eps_rad = DBSCAN_MICRO_EPS / EARTH_RADIUS_M
            db_micro = DBSCAN(eps=micro_eps_rad, min_samples=DBSCAN_MIN_SAMPLES, metric="haversine", algorithm="ball_tree")
            micro_labels = db_micro.fit_predict(macro_coords)
            
            # will re-map spatial IDs to clean unique structures, maintaining global noise mapping (-1)
            reconstructed_labels = []
            for sub_lbl in micro_labels:
                if sub_lbl == -1:
                    reconstructed_labels.append(-1)
                else:
                    reconstructed_labels.append(current_max_cluster + 1 + sub_lbl)
            
            poi.loc[macro_mask, "cluster"] = reconstructed_labels
            current_max_cluster = poi["cluster"].max()
            
  
    # CLUSTER METRICS & VALIDATION REPORTING
    n_clusters = len(set(poi["cluster"])) - (1 if -1 in poi["cluster"].values else 0)
    n_noise = int((poi["cluster"] == -1).sum())
    n_clustered = len(poi) - n_noise
    

    print(f" Distinct Candidate Cluster Zones Isolated: {n_clusters}")
    print(f" Structured Core Points Assigned: {n_clustered:,} ({n_clustered/len(poi)*100:.1f}%)")
    print(f" Unaligned Perimeter Noise Elements: {n_noise:,} ({n_noise/len(poi)*100:.1f}%)")
    
    if n_clusters > 1 and n_clustered > 0:
        valid_mask = poi['cluster'] != -1
        cluster_labels = poi[valid_mask]['cluster'].values
        cluster_coords = np.radians(poi[valid_mask][["lat", "lon"]].values)
        
        try:
            sil_score = silhouette_score(cluster_coords, cluster_labels, metric='haversine')
            db_score = davies_bouldin_score(cluster_coords, cluster_labels)
        except Exception as e:
            print(f" Structural Evaluation skipped: Matrix context too fine ({e})")
            
    # FEATURE AGGREGATION & CONVEX HULL EXECUTION
   
    clusters_only = poi[poi["cluster"] != -1].copy()
    
    print("\nComputing dynamic spatial geometry limits via Convex Hull engines please wait")
    cluster_groups = clusters_only.groupby("cluster")
    cluster_areas = cluster_groups.apply(calculate_true_cluster_area).reset_index(name="cluster_area_km2")
    
    centroids = clusters_only.groupby("cluster")[["lat", "lon"]].mean().reset_index()
    sizes = clusters_only.groupby("cluster").size().reset_index(name="poi_count")
    functional_counts = clusters_only.groupby(["cluster", "functional_group"]).size().unstack(fill_value=0).reset_index()
    
    # Merge structural matrices together
    zones = centroids.merge(sizes, on="cluster")
    zones = zones.merge(cluster_areas, on="cluster")
    zones = zones.merge(functional_counts, on="cluster")
    
    # Calculate True Destination Density Metrics
    zones["cluster_poi_density"] = zones["poi_count"] / zones["cluster_area_km2"]
    zones = zones.sort_values("poi_count", ascending=False)
    
    # Export Data Files to Disk
    poi.to_csv(POIS_CLUSTERED, index=False)
    zones.to_csv(CANDIDATE_ZONES, index=False)
    
    print(f" Point Array saved to: {POIS_CLUSTERED}")
    print(f"  Candidate Matrix saved to: {CANDIDATE_ZONES}")
    
    print("\nHigh Potential Zones Sample:")
    print(zones[["cluster", "poi_count", "cluster_area_km2", "cluster_poi_density", "lat", "lon"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()