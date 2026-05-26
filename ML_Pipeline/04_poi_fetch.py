import requests
import pandas as pd
from pathlib import Path
import time
import os

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
VILNIUS_BBOX = "54.60,25.05,54.80,25.45"

# Cache file
CACHED_POIS = OUT_DIR / "vilnius_pois_raw.csv"
CACHE_AGE_HOURS = 24  # Re-fetch if older than 24 hours

QUERY = f"""
[out:json][timeout:180];
(
  nwr({VILNIUS_BBOX})["amenity"="school"];
  nwr({VILNIUS_BBOX})["amenity"="university"];
  nwr({VILNIUS_BBOX})["amenity"="college"];
  nwr({VILNIUS_BBOX})["amenity"="kindergarten"];
  nwr({VILNIUS_BBOX})["amenity"="place_of_worship"];
  nwr({VILNIUS_BBOX})["office"];
  nwr({VILNIUS_BBOX})["landuse"="industrial"];
  nwr({VILNIUS_BBOX})["amenity"="parking"];
  nwr({VILNIUS_BBOX})["amenity"="charging_station"];
  nwr({VILNIUS_BBOX})["amenity"="fuel"];
  nwr({VILNIUS_BBOX})["shop"];
  nwr({VILNIUS_BBOX})["tourism"="museum"];
  nwr({VILNIUS_BBOX})["amenity"="theatre"];
  nwr({VILNIUS_BBOX})["amenity"="hospital"];
  nwr({VILNIUS_BBOX})["amenity"="clinic"];
  nwr({VILNIUS_BBOX})["tourism"="hotel"];
  nwr({VILNIUS_BBOX})["leisure"="sports_centre"];
  nwr({VILNIUS_BBOX})["amenity"="townhall"];
  nwr({VILNIUS_BBOX})["amenity"="restaurant"];
  nwr({VILNIUS_BBOX})["amenity"="fast_food"];
  nwr({VILNIUS_BBOX})["amenity"="cafe"];
  nwr({VILNIUS_BBOX})["amenity"="bar"];
  nwr({VILNIUS_BBOX})["amenity"="pub"];
);
out center;
"""


def fetch_with_retry(url, data, headers, max_retries=3, delay=5):

    for attempt in range(max_retries):
        try:
            response = requests.post(url, data=data, headers=headers, timeout=240)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Too Many Requests
                wait_time = delay * (attempt + 1)
                print(f"  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Attempt {attempt+1} failed: HTTP {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    return None
        except Exception as e:
            print(f"  Attempt {attempt+1} error: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return None
    return None


def element_to_row(el):
    tags = el.get("tags", {})
    lat, lon = None, None
    if "lat" in el and "lon" in el:
        lat, lon = el["lat"], el["lon"]
    elif "center" in el:
        lat = el["center"].get("lat")
        lon = el["center"].get("lon")
    return {
        "osm_type": el.get("type"),
        "osm_id": el.get("id"),
        "lat": lat,
        "lon": lon,
        "name": tags.get("name"),
        "amenity": tags.get("amenity"),
        "shop": tags.get("shop"),
        "office": tags.get("office"),
        "building": tags.get("building"),
        "landuse": tags.get("landuse"),
        "tourism": tags.get("tourism"),
        "historic": tags.get("historic"),
    }


def main():
    start_time = time.time()
    
    # Check cache
    if CACHED_POIS.exists():
        file_age_hours = (time.time() - CACHED_POIS.stat().st_mtime) / 3600
        if file_age_hours < CACHE_AGE_HOURS:
            print(f"  Using cached POI data (age: {file_age_hours:.1f} hours)")
            df = pd.read_csv(CACHED_POIS)
            print(f"  Loaded {len(df):,} POIs from cache")
            print(f"Runtime: {time.time() - start_time:.1f}s (cached)")
            return df
    
    print("  Fetching please wait")
    headers = {"User-Agent": "EVCS-Research/1.0"}
    
    response = fetch_with_retry(OVERPASS_URL, {"data": QUERY}, headers)
    
    if response is None:
        print("Error: Failed to fetch data from Overpass API after retries")
        return None
    
    data = response.json()
    elements = data.get("elements", [])
    print(f"  Elements received: {len(elements):,}")
    
    rows = [element_to_row(el) for el in elements]
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lat", "lon"])
    
    # Save to cache
    df.to_csv(CACHED_POIS, index=False)
    
    print(f"  Valid POIs: {len(df):,}")
    print(f"  Saved to: {CACHED_POIS}")
    print(f"  Runtime: {time.time() - start_time:.1f}s")
    
    # Print POI type summary
    print("\nfetched ")
    poi_counts = df['amenity'].value_counts().head(15)
    for amenity, count in poi_counts.items():
        print(f"  {amenity}: {count}")
    
    return df


if __name__ == "__main__":
    main()