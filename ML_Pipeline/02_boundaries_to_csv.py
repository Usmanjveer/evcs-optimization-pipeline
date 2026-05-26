
import json
import pandas as pd
from pathlib import Path
import time

SENIUNIJOS_GEOJSON = Path("Data_set/vilnius_seniunijos_polygons.geojson")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

OUT_CSV = RESULTS_DIR / "vilnius_seniunijos.csv"
OUT_CENTROIDS_GEOJSON = RESULTS_DIR / "vilnius_seniunijos_centroids.geojson"

def normalize_seniunija_name(name):
  #NOrmalization
    name = str(name).strip()
    mapping = {
        'Naujininkai': 'Naujininkai',
        'Paneriai': 'Paneriai',
        'Lazdynai': 'Lazdynai',
        'Grigiškės': 'Grigiškės',
        'Grigiskes': 'Grigiškės',
        'Vilkpėdė': 'Vilkpėdė',
        'Vilkpede': 'Vilkpėdė',
        'Senamiestis': 'Senamiestis',
        'Naujamiestis': 'Naujamiestis',
        'Rasos': 'Rasos',
        'Karoliniškės': 'Karoliniškės',
        'Karoliniskes': 'Karoliniškės',
        'Žvėrynas': 'Žvėrynas',
        'Zverynas': 'Žvėrynas',
        'Šnipiškės': 'Šnipiškės',
        'Snipiskes': 'Šnipiškės',
        'Viršuliškės': 'Viršuliškės',
        'Virsuliskes': 'Viršuliškės',
        'Naujoji Vilnia': 'Naujoji Vilnia',
        'Šeškinė': 'Šeškinė',
        'Seskine': 'Šeškinė',
        'Justiniškės': 'Justiniškės',
        'Justiniskes': 'Justiniškės',
        'Pilaitė': 'Pilaitė',
        'Pilaite': 'Pilaitė',
        'Žirmūnai': 'Žirmūnai',
        'Zirmunai': 'Žirmūnai',
        'Fabijoniškės': 'Fabijoniškės',
        'Fabijoniskes': 'Fabijoniškės',
        'Pašilaičiai': 'Pašilaičiai',
        'Pasilaiciai': 'Pašilaičiai',
        'Antakalnis': 'Antakalnis',
        'Verkiai': 'Verkiai',
    }
    
    if name in mapping:
        return mapping[name]
    else:
        print(f"  Unknown district name: '{name}' - keeping as is")
        return name


def calculate_polygon_centroid_from_rings(rings):
    #Calculate centroid from ESRI polygon rings
    if not rings or len(rings) == 0:
        return None, None
    
    outer_ring = rings[0]
    
    if not outer_ring or len(outer_ring) < 3:
        return None, None
    
    lats = []
    lons = []
    
    for point in outer_ring:
        lon, lat = point[0], point[1]
        lats.append(lat)
        lons.append(lon)
    
    if lats and lons:
        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)
        return centroid_lat, centroid_lon
    
    return None, None


def main():
    if not SENIUNIJOS_GEOJSON.exists():
        raise FileNotFoundError(f"Missing {SENIUNIJOS_GEOJSON}")
    
    
    start_time = time.time()
    
    with open(SENIUNIJOS_GEOJSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    features = data.get("features", [])
    print(f"Found {len(features)} features")
    
    boundaries = []
    
    for i, feature in enumerate(features):
        attrs = feature.get("attributes", {})
        
        seniunija = attrs.get("SENIUNIJA", f"Seniūnija_{i+1}")
        object_id = attrs.get("OBJECTID", i + 1)
        nr = attrs.get("NR", i + 1)
        shape_area = attrs.get("SHAPE.area", 0)
        shape_len = attrs.get("SHAPE.len", 0)
        
        geometry = feature.get("geometry", {})
        rings = geometry.get("rings", [])
        
        lat, lon = calculate_polygon_centroid_from_rings(rings)
        
        if lat and lon:
            seniunija_normalized = normalize_seniunija_name(seniunija)
            
            boundaries.append({
                "object_id": object_id,
                "seniunija": seniunija_normalized,
                "nr": nr,
                "area_m2": shape_area,
                "perimeter_m": shape_len,
                "centroid_lat": lat,
                "centroid_lon": lon
            })
            print(f"  {seniunija_normalized}: ({lat:.5f}, {lon:.5f})")
        else:
            print(f"  Could not extract centroid for {seniunija}")
    
    df = pd.DataFrame(boundaries)
    
    # Validation
    print(f"  Total districts found: {len(df)}")
    print(f"  Expected: 21 districts")
    
    if len(df) != 21:
        print(f"  Expected 21 districts but found {len(df)}")
    
    if df['seniunija'].duplicated().any():
        duplicates = df[df['seniunija'].duplicated()]['seniunija'].tolist()
        print(f"  Duplicate districts: {duplicates}")
    
    # Check for zero area
    zero_area = df[df['area_m2'] <= 0]
    if len(zero_area) > 0:
        print(f"  {len(zero_area)} districts have no area")
    
    df.to_csv(OUT_CSV, index=False)
    
    print(f"\nSaved {len(boundaries)} seniūnijos to: {OUT_CSV}")
    
    # Save centroids 
    centroids_features = []
    for _, row in df.iterrows():
        centroids_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["centroid_lon"], row["centroid_lat"]]
            },
            "properties": {
                "seniunija": row["seniunija"],
                "object_id": row["object_id"],
                "nr": row["nr"]
            }
        })
    
    centroids_geojson = {
        "type": "FeatureCollection",
        "features": centroids_features
    }
    
    with open(OUT_CENTROIDS_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(centroids_geojson, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved centroids GeoJSON to: {OUT_CENTROIDS_GEOJSON}")
    print(f"Total neighborhoods: {len(df)}")
    print(f"Runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()