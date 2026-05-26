import pandas as pd
import json
import time
from pathlib import Path
from config import POIS_RAW, POIS_CLEAN, SENIUNIJOS_GEOJSON

FUNCTIONAL_KEYWORDS = {
    "education": ["school", "university", "college", "kindergarten"],
    "work": ["office", "industrial", "warehouse", "townhall"],
    "leisure": ["restaurant", "cafe", "bar", "pub", "fast_food", "theatre", "museum", 
                "sports_centre", "gym", "fitness_centre", "hotel"],
    "retail": ["shop"],
    "religion": ["place_of_worship", "church"],
    "transport": ["parking", "fuel", "charging_station"],
    "healthcare": ["hospital", "clinic", "doctors"],
}

def create_poi_type(row):
    if pd.notna(row.get('amenity')):
        return f"amenity:{row['amenity']}"
    if pd.notna(row.get('shop')):
        return f"shop:{row['shop']}"
    if pd.notna(row.get('office')):
        return f"office:{row['office']}"
    if pd.notna(row.get('building')):
        return f"building:{row['building']}"
    if pd.notna(row.get('tourism')):
        return f"tourism:{row['tourism']}"
    if pd.notna(row.get('landuse')):
        return f"landuse:{row['landuse']}"
    return "other:unknown"

def map_functional_group(poi_type):
    if pd.isna(poi_type):
        return "other"
    t = str(poi_type).lower()
    
    # Retail
    if "shop" in t:
        return "retail"
    # Work
    if "office" in t or "industrial" in t or "townhall" in t:
        return "work"
    # Education
    if "school" in t or "university" in t or "college" in t or "kindergarten" in t:
        return "education"
    # Leisure
    if "restaurant" in t or "cafe" in t or "bar" in t or "pub" in t or "fast_food" in t:
        return "leisure"
    if "theatre" in t or "museum" in t or "sports" in t or "gym" in t or "fitness" in t:
        return "leisure"
    if "hotel" in t:
        return "leisure"
    # Transport
    if "parking" in t or "fuel" in t or "charging" in t:
        return "transport"
    # Religion
    if "place_of_worship" in t or "church" in t:
        return "religion"
    # Healthcare
    if "hospital" in t or "clinic" in t or "doctor" in t:
        return "healthcare"
    return "other"

def load_seniunija_polygons():
    with open(SENIUNIJOS_GEOJSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    seniunija_data = []
    for feature in data.get("features", []):
        attrs = feature.get("attributes", {})
        seniunija = attrs.get("SENIUNIJA", "")
        nr = attrs.get("NR", "")
        geometry = feature.get("geometry", {})
        rings = geometry.get("rings", [])
        
        if rings and len(rings) > 0:
            outer_ring = rings[0]
            if outer_ring and len(outer_ring) >= 3:
                lons = [p[0] for p in outer_ring]
                lats = [p[1] for p in outer_ring]
                polygon_coords = [(p[0], p[1]) for p in outer_ring]
                seniunija_data.append({
                    "seniunija": seniunija,
                    "nr": nr,
                    "polygon_coords": polygon_coords,
                    "bbox": (min(lons), max(lons), min(lats), max(lats))
                })
    return seniunija_data

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

def main():
    if not POIS_RAW.exists():
        raise FileNotFoundError(f"Missing {POIS_RAW}")
    
    start_time = time.time()
    seniunija_data = load_seniunija_polygons()
    df = pd.read_csv(POIS_RAW)
    print(f"Loaded {len(df)} raw POIs")
    
    df = df.dropna(subset=["lat", "lon"])
    if "osm_type" in df.columns and "osm_id" in df.columns:
        df = df.drop_duplicates(subset=["osm_type", "osm_id"])
    
    df["poi_type"] = df.apply(create_poi_type, axis=1)
    df["functional_group"] = df["poi_type"].apply(map_functional_group)
    
    print("\nFunctional group distribution:")
    group_counts = df["functional_group"].value_counts()
    for group, count in group_counts.items():
        print(f"  {group}: {count} ({count/len(df)*100:.1f}%)")
    
    if seniunija_data:
        print(f"\n Assigning seniūnija to {len(df)} POIs...")
        seniunija_list, seniunija_nr_list = [], []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            lat, lon = row["lat"], row["lon"]
            found = False
            for data in seniunija_data:
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
        
        df["seniunija"] = seniunija_list
        df["seniunija_nr"] = seniunija_nr_list
        assigned = df['seniunija'].notna().sum()
        unassigned = len(df) - assigned
        print(f"Assigned seniūnija to {assigned:,} POIs")
        if unassigned > 0:
            print(f"   Some of the {unassigned} POIs could not be assigned to any seniūnija")
    
    keep_cols = ["osm_type", "osm_id", "lat", "lon", "poi_type", "functional_group", "name", "seniunija", "seniunija_nr"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(POIS_CLEAN, index=False)
    
    print(f"\n Saved {len(df)} clean POIs to: {POIS_CLEAN}")
    print(f"Runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()