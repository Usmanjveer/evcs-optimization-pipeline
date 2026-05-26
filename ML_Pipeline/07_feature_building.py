
import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

# File Registry Environment
CANDIDATE_ZONES = Path("results/evcs_candidate_zones.csv")
POIS_CLUSTERED = Path("results/vilnius_pois_clustered.csv")  
SENIUNIJOS_CSV = Path("results/vilnius_seniunijos.csv")
POP_BY_SENIUNIJA = Path("results/population_by_seniunija.csv")
TRAFFIC_JAM_SCORES = Path("results/traffic_jam_scores_by_seniunija.csv")
EV_BY_SENIUNIJA = Path("results/ev_by_seniunija.csv")
EXISTING_EVCS_GEOJSON = Path("Data_set/osm_existing_evcs_vilnius.geojson")
OUT_FEATURES = Path("results/evcs_real_features.csv")

# Hyperparameters for Advanced Spatial Engineering
NEIGHBORHOOD_RADIUS_M = 750  # Spatial lag extent boundary threshold
SMOOTHING_PARAMETER_M = 50          # Prevents division-by-zero singularities in IDW
SELF_WEIGHT_ALPHA = 0.30          # Local node value retention weight in convex lag blend

# Demographic Regional Baselines
CITY_YOUNG_SHARE = 0.254   
CITY_WORKING_SHARE = 0.577 
CITY_SENIOR_SHARE = 0.169  


def haversine_distance(lat1, lon1, lat2, lon2):
   # haversine formula
    R = 6371000
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    if hasattr(lat2, '__len__'):
        dlat = lat2_rad[:, np.newaxis] - lat1_rad
        dlon = lon2_rad[:, np.newaxis] - lon1_rad
    else:
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def custom_haversine_metric(u, v):
    return haversine_distance(u[0], u[1], v[0], v[1])

#Normalization
def normalize_seniunija_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    mapping = {
        'Antakalnis': 'Antakalnis', 'Fabijoniškės': 'Fabijoniškės', 'Fabijoniskes': 'Fabijoniškės',
        'Grigiškės': 'Grigiškės', 'Grigiskes': 'Grigiškės', 'Justiniškės': 'Justiniškės', 
        'Justiniskes': 'Justiniškės', 'Karoliniškės': 'Karoliniškės', 'Karoliniskes': 'Karoliniškės',
        'Lazdynai': 'Lazdynai', 'Naujamiestis': 'Naujamiestis', 'Naujininkai': 'Naujininkai',
        'Naujoji Vilnia': 'Naujoji Vilnia', 'NaujojiVilnia': 'Naujoji Vilnia', 'Paneriai': 'Paneriai',
        'Pašilaičiai': 'Pašilaičiai', 'Pasilaiciai': 'Pašilaičiai', 'Pilaitė': 'Pilaitė', 
        'Pilaite': 'Pilaitė', 'Rasos': 'Rasos', 'Senamiestis': 'Senamiestis', 'Šeškinė': 'Šeškinė', 
        'Seskine': 'Šeškinė', 'Šnipiškės': 'Šnipiškės', 'Snipiskes': 'Šnipiškės', 'Verkiai': 'Verkiai',
        'Vilkpėdė': 'Vilkpėdė', 'Vilkpede': 'Vilkpėdė', 'Viršuliškės': 'Viršuliškės', 
        'Virsuliskes': 'Viršuliškės', 'Žirmūnai': 'Žirmūnai', 'Zirmunai': 'Žirmūnai',
        'Žvėrynas': 'Žvėrynas', 'Zverynas': 'Žvėrynas',
    }
    return mapping.get(name, name)


def point_in_polygon(lat, lon, polygon_coords):

    x, y = lon, lat
    inside = False
    n = len(polygon_coords)
    j = n - 1
    for i in range(n):
        xi, yi = polygon_coords[i]
        xj, yj = polygon_coords[j]
        if ((yi > y) != (yj > y)):
            x_intersect = xj + (xi - xj) * (y - yj) / (yi - yj)
            if x_intersect > x:
                inside = not inside
        j = i
    return inside


def load_seniunija_polygons():
    #Extract coordinate chains from GeoJSON files
    seniunijos_geojson = Path("Data_set/vilnius_seniunijos_polygons.geojson")
    if not seniunijos_geojson.exists():
        return None
    with open(seniunijos_geojson, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    seniunija_data = []
    for feature in data.get("features", []):
        attrs = feature.get("attributes", {})
        geometry = feature.get("geometry", {})
        rings = geometry.get("rings", [])
        if rings and len(rings) > 0:
            outer_ring = rings[0]
            if len(outer_ring) >= 3:
                seniunija_data.append({
                    "seniunija": attrs.get("SENIUNIJA", ""),
                    "nr": attrs.get("NR", ""),
                    "polygon_coords": [(p[0], p[1]) for p in outer_ring],
                    "bbox": (min([p[0] for p in outer_ring]), max([p[0] for p in outer_ring]), 
                             min([p[1] for p in outer_ring]), max([p[1] for p in outer_ring]))
                })
    return seniunija_data


def assign_zones_to_seniunija(zones, seniunijos):
    polygon_data = load_seniunija_polygons()
    if polygon_data is not None:
        seniunija_list, seniunija_nr_list = [], []
        for _, zone in zones.iterrows():
            lat, lon = zone['lat'], zone['lon']
            found = False
            for data in polygon_data:
                min_lon, max_lon, min_lat, max_lat = data["bbox"]
                if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                    if point_in_polygon(lat, lon, data["polygon_coords"]):
                        seniunija_list.append(data["seniunija"])
                        seniunija_nr_list.append(data["nr"])
                        found = True
                        break
            if not found:
                seniunija_list.append(None)
                seniunija_nr_list.append(None)
        zones['seniunija'] = seniunija_list
        zones['seniunija_nr'] = seniunija_nr_list
        
        unassigned_mask = zones['seniunija'].isna()
        if unassigned_mask.any():
            sen_coords = seniunijos[['centroid_lat', 'centroid_lon']].values
            zone_coords = zones.loc[unassigned_mask, ['lat', 'lon']].values
            distances = cdist(zone_coords, sen_coords)
            nearest_idx = distances.argmin(axis=1)
            zones.loc[unassigned_mask, 'seniunija'] = seniunijos.iloc[nearest_idx]['seniunija'].values
            zones.loc[unassigned_mask, 'seniunija_nr'] = seniunijos.iloc[nearest_idx]['nr'].values
    else:
        distances = cdist(zones[['lat', 'lon']].values, seniunijos[['centroid_lat', 'centroid_lon']].values)
        nearest_idx = distances.argmin(axis=1)
        zones['seniunija'] = seniunijos.iloc[nearest_idx]['seniunija'].values
        zones['seniunija_nr'] = seniunijos.iloc[nearest_idx]['nr'].values
        
    zones['seniunija'] = zones['seniunija'].apply(normalize_seniunija_name)
    return zones


def main():
  
  # COMPOSITE FEATURE MATRICES & OUT-OF-SAMPLE WEIGHT CALIBRATION
    

    zones = pd.read_csv(CANDIDATE_ZONES)
    seniunijos = pd.read_csv(SENIUNIJOS_CSV)
    pop_data = pd.read_csv(POP_BY_SENIUNIJA)
    pois_clustered = pd.read_csv(POIS_CLUSTERED)
    jam_data = pd.read_csv(TRAFFIC_JAM_SCORES) if TRAFFIC_JAM_SCORES.exists() else None
    
    # Standardize textual tags across files
    seniunijos['seniunija'] = seniunijos['seniunija'].apply(normalize_seniunija_name)
    pop_data['seniunija'] = pop_data['seniunija'].apply(normalize_seniunija_name)
    if jam_data is not None:
        jam_data['seniunija'] = jam_data['jam_data'].apply(normalize_seniunija_name) if 'jam_data' in jam_data.columns else jam_data['seniunija'].apply(normalize_seniunija_name)
    pois_clustered['seniunija'] = pois_clustered['seniunija'].apply(normalize_seniunija_name)
    
    # Append regional characteristics to the candidates array
    zones = assign_zones_to_seniunija(zones, seniunijos)
    
    pop_lookup = pop_data[['seniunija', 'population', 'young_share', 'working_share', 'senior_share', 'area_m2']].copy().drop_duplicates(subset=['seniunija'])
    zones = zones.merge(pop_lookup, on='seniunija', how='left')
    
    if jam_data is not None:
        jam_lookup = jam_data[['seniunija', 'jam_score_norm']].copy().drop_duplicates(subset=['seniunija'])
        zones = zones.merge(jam_lookup, on='seniunija', how='left')
        zones['jam_score_norm'] = zones['jam_score_norm'].fillna(0)
    else:
        zones['jam_score_norm'] = 0
        
    zones['population'] = zones['population'].fillna(zones['population'].median())
    zones['area_m2'] = zones['area_m2'].fillna(zones['area_m2'].median())
    zones['young_share'] = zones['young_share'].fillna(CITY_YOUNG_SHARE)
    zones['working_share'] = zones['working_share'].fillna(CITY_WORKING_SHARE)
    zones['senior_share'] = zones['senior_share'].fillna(CITY_SENIOR_SHARE)
    
    # Establish structural footprint distributions from data definitions
    zones['district_area_km2'] = zones['area_m2'] / 1_000_000
    zones['district_pop_density'] = zones['population'] / zones['district_area_km2']
    
    func_cols = ["education", "leisure", "other", "religion", "retail", "transport", "work"]
    for col in func_cols:
        zones[col] = zones[col].fillna(0)
    zones['poi_total'] = zones[func_cols].sum(axis=1)
    
    # custom Convex Hull geometry areas instead of fixed circle metrics
    zones['poi_density'] = zones['poi_count'] / zones['cluster_area_km2']
    
    # Calculate proximity distances to existing operational charging stations
    existing = None
    if EXISTING_EVCS_GEOJSON.exists():
        with open(EXISTING_EVCS_GEOJSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        stations = []
        for feature in data.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') == 'Point' and len(geom.get('coordinates', [])) >= 2:
                stations.append({'lat': geom['coordinates'][1], 'lon': geom['coordinates'][0]})
        existing = pd.DataFrame(stations)
    
    if existing is not None and len(existing) > 0:
        zones['dist_to_existing_m'] = [np.min(haversine_distance(z['lat'], z['lon'], existing['lat'].values, existing['lon'].values)) for _, z in zones.iterrows()]
    else:
        zones['dist_to_existing_m'] = 2000.0

 # REFINEMENT B: OUT-OF-SAMPLE GRID SEARCH CALIBRATION LOOP
        # Synthesizes an independent multi-factor index to break circular dependencies.
    zones['pop_share_norm'] = zones['district_pop_density'] / zones['district_pop_density'].sum()
    zones['poi_share_norm'] = zones['poi_density'] / zones['poi_density'].sum()
    zones['traffic_share_norm'] = zones['jam_score_norm'] / zones['jam_score_norm'].sum()
    
    print(" Splitting candidates into out-of-sample partitions (30% Calibration / 70% Validation)...")
    calib_set, valid_set = train_test_split(zones, test_size=0.70, random_state=42)
    
    best_correlation = 1.0  # Proximity to existing infrastructure implies high demand (maximizing negative correlation)
    best_weights = {'pop': 0.40, 'poi': 0.45, 'traffic': 0.15} # Default fallback limits
    
    print("Running empirical grid sweep across weight vectors Please wait")
    for w_pop in np.linspace(0.1, 0.6, 6):
        for w_poi in np.linspace(0.1, 0.6, 6):
            w_traffic = 1.0 - (w_pop + w_poi)
            if w_traffic < 0.05 or w_traffic > 0.40:
                continue
                
            # Synthesize demand matrix on calibration partition
            calib_index = (w_pop * calib_set['pop_share_norm']) + (w_poi * calib_set['poi_share_norm']) + (w_traffic * calib_set['traffic_share_norm'])
            corr = calib_index.corr(calib_set['dist_to_existing_m'], method='spearman')
            
            if pd.notna(corr) and corr < best_correlation:
                best_correlation = corr
                best_weights = {'pop': round(w_pop, 2), 'poi': round(w_poi, 2), 'traffic': round(w_traffic, 2)}
                
    # Evaluate generalization performance on unseen validation split
    valid_index = (best_weights['pop'] * valid_set['pop_share_norm']) + (best_weights['poi'] * valid_set['poi_share_norm']) + (best_weights['traffic'] * valid_set['traffic_share_norm'])
    validation_score = valid_index.corr(valid_set['dist_to_existing_m'], method='spearman')
    
    print(f"  Optimal Weights Found: Residential={best_weights['pop']:.2f} | Destination POI={best_weights['poi']:.2f} | Transit Traffic={best_weights['traffic']:.2f}")
    print(f"  Spatial Generalization Rank Score (Validation Spearman): {validation_score:.4f}")

    # Build the final, non-circular target array (ev_density) across the full dataset
    zones['composite_demand_weight'] = (
        (best_weights['pop'] * zones['pop_share_norm']) + 
        (best_weights['poi'] * zones['poi_share_norm']) + 
        (best_weights['traffic'] * zones['traffic_share_norm'])
    )
    
    # Scale total Vilnius EVs across clusters based on the calibrated multi-factor weight
    TOTAL_VILNIUS_EVS = 11809  
    zones['simulated_ev_count'] = zones['composite_demand_weight'] * TOTAL_VILNIUS_EVS
    zones['ev_density'] = zones['simulated_ev_count'] / zones['cluster_area_km2']
    zones['total_ev_score'] = zones['simulated_ev_count']

    
    # INVERSE-DISTANCE WEIGHTED SPATIAL LAGS
   
    print(f"\nEngineering IDW Spatial Lags (Radius: {NEIGHBORHOOD_RADIUS_M}m, Decay Scaling enabled)")
    cluster_coords = zones[['lat', 'lon']].values
    
    # Generate mathematically exact geodesic cross-distance array using customized vectorized Haversine logic
    cluster_dist_matrix = cdist(cluster_coords, cluster_coords, metric=custom_haversine_metric)
    
    # Derive structural spatial entity densities to insulate against cluster footprint scale biases
    zones['retail_density'] = zones['retail'] / zones['cluster_area_km2']
    zones['leisure_density'] = zones['leisure'] / zones['cluster_area_km2']
    
    lag_traffic, lag_retail, lag_leisure = [], [], []
    
    for idx in range(len(zones)):
        # Construct neighbor indicator array strictly isolating external features
        neighbors_idx = (cluster_dist_matrix[idx] <= NEIGHBORHOOD_RADIUS_M) & (cluster_dist_matrix[idx] > 0)
        
        # Capture focal point localized baselines
        self_traffic = zones.loc[idx, 'jam_score_norm']
        self_retail = zones.loc[idx, 'retail_density']
        self_leisure = zones.loc[idx, 'leisure_density']
        
        if neighbors_idx.any():
            # Apply Inverse Distance Weighting to penalize geographic separation smoothly
            distances = cluster_dist_matrix[idx][neighbors_idx]
            weights = 1.0 / (distances + SMOOTHING_PARAMETER_M)
            normalized_weights = weights / weights.sum()
            
            # Aggregate structural continuous attributes across spatial matrices
            neighbor_traffic = np.sum(normalized_weights * zones.loc[neighbors_idx, 'jam_score_norm'].values)
            neighbor_retail = np.sum(normalized_weights * zones.loc[neighbors_idx, 'retail_density'].values)
            neighbor_leisure = np.sum(normalized_weights * zones.loc[neighbors_idx, 'leisure_density'].values)
            
            # Execute consistent convex fusion across continuous entities to shield against data discontinuities
            lag_traffic.append((SELF_WEIGHT_ALPHA * self_traffic) + ((1.0 - SELF_WEIGHT_ALPHA) * neighbor_traffic))
            lag_retail.append((SELF_WEIGHT_ALPHA * self_retail) + ((1.0 - SELF_WEIGHT_ALPHA) * neighbor_retail))
            lag_leisure.append((SELF_WEIGHT_ALPHA * self_leisure) + ((1.0 - SELF_WEIGHT_ALPHA) * neighbor_leisure))
        else:
            # Structurally clean fallback: isolated points cleanly settle to their base local density metrics
            lag_traffic.append(self_traffic)
            lag_retail.append(self_retail)
            lag_leisure.append(self_leisure)
            
    zones['lag_neighborhood_traffic'] = lag_traffic
    zones['lag_neighborhood_retail'] = lag_retail
    zones['lag_neighborhood_leisure'] = lag_leisure
    
    # Clean temporary weighting arrays out of final feature export matrix
    save_cols = ['cluster', 'lat', 'lon', 'poi_count', 'cluster_area_km2', 'education', 'leisure', 'other', 
                 'religion', 'retail', 'transport', 'work', 'seniunija', 'seniunija_nr',
                 'total_ev_score', 'ev_density', 'population', 'young_share', 'working_share', 'senior_share', 
                 'district_area_km2', 'district_pop_density', 'jam_score_norm', 'poi_total', 'poi_density', 
                 'dist_to_existing_m', 'lag_neighborhood_traffic', 'lag_neighborhood_retail', 'lag_neighborhood_leisure']
    
    zones[save_cols].to_csv(OUT_FEATURES, index=False)
    print(f"\nfeature matrix saved to: {OUT_FEATURES}")


if __name__ == "__main__":
    main()