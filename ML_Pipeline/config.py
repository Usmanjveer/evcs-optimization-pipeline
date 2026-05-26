from pathlib import Path

# Directories
DATA_DIR = Path("Data_set")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Input data files
RESIDENTS_CSV = DATA_DIR / "data_set.csv"
SENIUNIJOS_GEOJSON = DATA_DIR / "vilnius_seniunijos_polygons.geojson"
TRAFFIC_CONGESTION_FILE = DATA_DIR / "Spustys.csv"
EXISTING_EVCS_GEOJSON = DATA_DIR / "osm_existing_evcs_vilnius.geojson"
EV_DATA_PATH = DATA_DIR / "data_2026.xlsx"

# Output files
POIS_RAW = RESULTS_DIR / "vilnius_pois_raw.csv"
POIS_CLEAN = RESULTS_DIR / "vilnius_pois_clean.csv"
POIS_CLUSTERED = RESULTS_DIR / "vilnius_pois_clustered.csv"
CANDIDATE_ZONES = RESULTS_DIR / "evcs_candidate_zones.csv"

# Parameters
CHUNK_SIZE = 50000
DBSCAN_EPS_METERS = 300
DBSCAN_MIN_SAMPLES = 5

# Age groups
AGE_GROUPS = {"Under 25": (0, 24), "25–64": (25, 64), "65+": (65, 120)}