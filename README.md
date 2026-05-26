#EV Charging Station Optimization Pipeline

## Pipeline Overview

This pipeline optimizes electric vehicle charging station (EVCS) placement in Vilnius, Lithuania using a multi-stage machine learning approach:

1. **Data Processing** - Population, traffic, POI, and boundary data
2. **Clustering** - Two-tier DBSCAN with convex hull area calculation
3. **Demand Prediction** - Ensemble model (Ridge Regression + Random Forest)
4. **Site Classification** - SVM based on revealed preferences
5. **Optimization** - Diversity-aware Genetic Algorithm
6. **Visualization** - Interactive Folium map

---

## Installation

### Step 1: Clone or Download the Pipeline

```bash
git clone https://github.com/Usmanjveer/evcs-optimization-pipeline.git
cd evcs-optimization-pipeline
Step 2: Create Virtual Environment (Recommended)
Windows:

bash
python -m venv venv
venv\Scripts\activate
macOS/Linux:

bash
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
Create a requirements.txt file with the following content:

txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
folium>=0.12.0
requests>=2.25.0
scipy>=1.7.0
joblib>=1.1.0
shapely>=1.8.0
pyproj>=3.3.0
Then install:

bash
pip install -r requirements.txt
Or install all at once:

bash
pip install pandas numpy scikit-learn matplotlib folium requests scipy joblib shapely pyproj
Data Files Required
Place all data files in the Data_set/ folder:

File Name	Description	Source
vilnius_population_by_seniunija.csv	Population by seniūnija	Vilnius Open Data
vilnius_seniunijos_polygons.geojson	Seniūnija boundaries	Vilnius Open Data
Spustys.csv	Real-time traffic congestion events	Vilnius Open Data
data_2026.xlsx	EV registration data (pure/hybrid counts)	Vilnius registry
osm_existing_evcs_vilnius.geojson	Existing EV charging stations	OpenStreetMap
Note: The pipeline will fetch POI data from OpenStreetMap Overpass API automatically.

Folder Structure
Before running, ensure your folder structure looks like this:

text
evcs-optimization-pipeline/
│
├── Data_set/                          # All input data files
│   ├── vilnius_population_by_seniunija.csv
│   ├── vilnius_seniunijos_polygons.geojson
│   ├── Spustys.csv
│   ├── data_2026.xlsx
│   └── osm_existing_evcs_vilnius.geojson
│
├── results/                           # Output folder (created automatically)
│
├── config.py                          # Configuration file (create this)
│
├── 01_population_aggregates.py
├── 02_boundaries_to_csv.py
├── 03_Traffic_and_Ev_loader.py
├── 04_poi_fetch.py
├── 05_poi_preprocess.py
├── 06_poi_clustering.py
├── 07_feature_building.py
├── 08_ensemble_model.py
├── 09_svm_site_classification.py
├── 10_ga_optimization.py
├── 11_visualize_map.py
│
└── requirements.txt
