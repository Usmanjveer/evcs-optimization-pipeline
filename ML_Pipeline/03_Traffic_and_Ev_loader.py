
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

TRAFFIC_FILE = Path("Data_set/Spustys.csv")
EV_EXCEL_FILE = Path("Data_set/data_2026.xlsx")
SENIUNIJOS_CSV = Path("results/vilnius_seniunijos.csv")
POP_BY_SENIUNIJA = Path("results/population_by_seniunija.csv")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

OUT_EV = RESULTS_DIR / "ev_by_seniunija.csv"
OUT_JAM = RESULTS_DIR / "traffic_jam_scores_by_seniunija.csv"
OUT_HOTSPOTS = RESULTS_DIR / "traffic_hotspots.csv"

# Source: Lithuanian vehicle registry as of May 1, 2026
TOTAL_PURE_EV = 11809
TOTAL_HYBRID_EV = 54838


def clean_seniunija_name(name):

    if pd.isna(name): 
        return None
    name = str(name).strip()
    for old in ['VILNIAUS APSKR. ', ' M. SAV.', ' R. SAV.', ' SAV.', ' APSKR.']:
        name = name.replace(old, '')
    return name.strip()


def convert_lks94_to_wgs84(x, y):
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:3346", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except Exception as e:
        return None, None


def calculate_advanced_temporal_multiplier(row):

    # Fallback to standard if column parsing errors occur
    try:
        piko_str = str(row.get('Piko metas', '')).lower()
        
        is_weekend = False 
        if 'vakarinis' in piko_str or 'vakaro' in piko_str:
            return 2.5  
        elif 'rytinis' in piko_str or 'ryto' in piko_str:
            return 0.7  
        elif 'savaitgalio' in piko_str:
            return 1.8  
        elif 'nakties' in piko_str:
            return 0.3 
            
        return 1.0
    except:
        return 1.0


def calculate_jam_score_for_seniunija(sen_lat, sen_lon, traffic_hotspots, radius_m=2000):
 
    if len(traffic_hotspots) == 0:
        return 0.0
    
    lat1_rad = np.radians(sen_lat)
    lon1_rad = np.radians(sen_lon)
    lat2_rad = np.radians(traffic_hotspots['lat'].values)
    lon2_rad = np.radians(traffic_hotspots['lon'].values)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = 6371000 * c
    
    nearby_mask = distances <= radius_m
    if not nearby_mask.any():
        return 0.0
    
    nearby_hotspots = traffic_hotspots[nearby_mask].copy()
    nearby_distances = distances[nearby_mask]
    
    weights = 1 / (nearby_distances + 50)
    weights = weights / weights.sum()
    
    jam_score = (nearby_hotspots['jam_weight'].values * weights).sum()
    return jam_score


def create_ev_distribution():
    
    if not POP_BY_SENIUNIJA.exists():
        print(f"Population file not found: {POP_BY_SENIUNIJA}")
        return None
    
    pop_data = pd.read_csv(POP_BY_SENIUNIJA)
    
    if not SENIUNIJOS_CSV.exists():
        print(f" Seniūnija file not found: {SENIUNIJOS_CSV}")
        return None
    
    seniunijos = pd.read_csv(SENIUNIJOS_CSV)
    merged = pop_data.merge(seniunijos[['seniunija', 'area_m2']], on='seniunija')
    
    total_population = merged['population'].sum()
    merged['pop_share'] = merged['population'] / total_population
    
    ev_data = []
    for _, row in merged.iterrows():
        pure_ev = round(TOTAL_PURE_EV * row['pop_share'])
        hybrid_ev = round(TOTAL_HYBRID_EV * row['pop_share'])
        
        ev_data.append({
            'seniunija': row['seniunija'],
            'pure_ev': pure_ev,
            'hybrid_ev': hybrid_ev,
            'total_ev_score_weighted': round(pure_ev + 0.35 * hybrid_ev, 1)
        })
    
    df = pd.DataFrame(ev_data)
    area_dict = dict(zip(seniunijos['seniunija'], seniunijos['area_m2']))
    df['area_km2'] = df['seniunija'].map(area_dict) / 1_000_000
    
    df.to_csv(OUT_EV, index=False)
    print(f"Saved base demographic EV index to: {OUT_EV}")
    return df


def process_traffic_data():
    
    if not TRAFFIC_FILE.exists():
        print(f"File not found: {TRAFFIC_FILE}")
        return None
    
    df = pd.read_csv(TRAFFIC_FILE, encoding='utf-8')
    print(f" Loaded {len(df)} traffic events")
    
    df['duration_min'] = pd.to_numeric(df['Trukmė (min.)'], errors='coerce').fillna(0)
    
    print("\nConverting coordinates from LKS94 to WGS84 wait please")
    coords = df.apply(lambda row: convert_lks94_to_wgs84(row['x'], row['y']), axis=1)
    df['lat'] = coords.apply(lambda x: x[0] if x[0] is not None else None)
    df['lon'] = coords.apply(lambda x: x[1] if x[1] is not None else None)
    df = df.dropna(subset=['lat', 'lon'])
    
    # Apply Refined Behavioral Temporal Multipliers
    df['peak_multiplier'] = df.apply(calculate_advanced_temporal_multiplier, axis=1)
    df['jam_weight'] = df['duration_min'] * df['peak_multiplier']
    
    # Aggregate to spatial hotspots
    df['lat_round'] = df['lat'].round(5)
    df['lon_round'] = df['lon'].round(5)
    
    hotspots = df.groupby(['lat_round', 'lon_round']).agg({
        'jam_weight': 'sum',
        'duration_min': ['count', 'sum', 'mean'],
        'Vieta': 'first',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    
    hotspots.columns = ['lat_round', 'lon_round', 'jam_weight', 'event_count', 
                        'total_duration', 'avg_duration', 'location_name', 'lat', 'lon']
    
    if hotspots['jam_weight'].max() > 0:
        hotspots['jam_score'] = hotspots['jam_weight'] / hotspots['jam_weight'].max()
    
    hotspots.to_csv(OUT_HOTSPOTS, index=False)
    print(f"Saved {len(hotspots)} behavior-weighted traffic hotspots to: {OUT_HOTSPOTS}")
    
    seniunijos = pd.read_csv(SENIUNIJOS_CSV)
    jam_by_seniunija = []
    for _, sen in seniunijos.iterrows():
        jam_score = calculate_jam_score_for_seniunija(
            sen['centroid_lat'], sen['centroid_lon'], hotspots, radius_m=2000
        )
        jam_by_seniunija.append({
            'seniunija': sen['seniunija'],
            'seniunija_nr': sen['nr'],
            'jam_score_raw': jam_score,
            'jam_score_norm': 0
        })
    
    jam_df = pd.DataFrame(jam_by_seniunija)
    if jam_df['jam_score_raw'].max() > 0:
        jam_df['jam_score_norm'] = jam_df['jam_score_raw'] / jam_df['jam_score_raw'].max()
        
    jam_df.to_csv(OUT_JAM, index=False)
    print(f" Saved aggregated behavior-weighted jam scores to: {OUT_JAM}")
    return hotspots, jam_df


def main():
    create_ev_distribution()
    process_traffic_data()
if __name__ == "__main__":
    main()