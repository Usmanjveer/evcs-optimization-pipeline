"""
11_visualize_results.py - Interactive Map Visualization
CLEAR DISTINCTION: Existing (gray) vs Proposed (colored) charging stations
Uses demand_score from ensemble model + SVM Filtered Layer
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import folium
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm
import webbrowser

# Paths
SELECTED_SITES = Path("results/ga_selected_sites_final.csv")
CANDIDATES_FILE = Path("results/evcs_with_svm_classification.csv")
FALLBACK_CANDIDATES = Path("results/evcs_with_predictions.csv")  
EXISTING_EVCS = Path("Data_set/osm_existing_evcs_vilnius.geojson")
SENIUNIJOS_GEOJSON = Path("Data_set/vilnius_seniunijos_polygons.geojson")
TRAFFIC_HOTSPOTS = Path("results/traffic_hotspots.csv")
OUTPUT_MAP = Path("results/evcs_deployment_map.html")

def standardize_demand_column(df):
    """Standardize demand column name (handles demand_score or demand_ann)"""
    if 'demand_score' in df.columns:
        return df['demand_score']
    elif 'demand_ann' in df.columns:
        print("  ⚠️ Using legacy 'demand_ann' column")
        return df['demand_ann']
    elif 'predicted_ev_density' in df.columns:
        print("  ⚠️ Creating demand from predicted_ev_density")
        return df['predicted_ev_density'] / df['predicted_ev_density'].max()
    else:
        # Try to create from svm_probability as last resort
        if 'svm_probability' in df.columns:
            print("  ⚠️ Using svm_probability as demand proxy")
            return df['svm_probability']
        raise ValueError("No demand column found!")

def load_all_data():
    """Load all necessary data with fallbacks"""
    print("📊 Loading data...")
    
    # Load selected stations
    if not SELECTED_SITES.exists():
        raise FileNotFoundError(f"Missing {SELECTED_SITES}. Run 10_ga_optimization.py first.")
    
    selected = pd.read_csv(SELECTED_SITES)
    print(f"  ✓ Loaded {len(selected)} PROPOSED stations")
    
    # Load candidates with fallback
    if CANDIDATES_FILE.exists():
        candidates = pd.read_csv(CANDIDATES_FILE)
        print(f"  ✓ Loaded {len(candidates)} candidate zones from {CANDIDATES_FILE}")
    elif FALLBACK_CANDIDATES.exists():
        candidates = pd.read_csv(FALLBACK_CANDIDATES)
        print(f"  ⚠️ Loaded {len(candidates)} candidate zones from fallback: {FALLBACK_CANDIDATES}")
        # Create svm_probability if not exists
        if 'svm_probability' not in candidates.columns:
            candidates['svm_probability'] = candidates['demand_score']
    else:
        raise FileNotFoundError(f"No candidate file found. Run 09_svm_site_classification.py first.")
    
    # Standardize demand column in candidates
    print("  ✓ Standardizing demand column...")
    candidates['demand_score'] = standardize_demand_column(candidates)
    
    # Traffic hotspots
    traffic = None
    if TRAFFIC_HOTSPOTS.exists():
        traffic = pd.read_csv(TRAFFIC_HOTSPOTS)
        print(f"  ✓ Loaded {len(traffic)} traffic hotspots")
    
    return selected, candidates, traffic

def create_color_maps():
    """Create color maps for different features"""
    phase_colors = {
        1: '#E63946',  # Phase 1 - Bright Red
        2: '#F4A261',  # Phase 2 - Orange
        3: '#2A9D8F'   # Phase 3 - Teal Green
    }
    
    demand_cmap = cm.LinearColormap(
        colors=['#0000FF', '#00FF00', '#FFFF00', '#FF0000'],
        vmin=0, vmax=1,
        caption='Demand Score'
    )
    
    traffic_cmap = cm.LinearColormap(
        colors=['#FFCCCC', '#FF9999', '#FF6666', '#FF0000', '#990000'],
        vmin=0, vmax=1,
        caption='Traffic Congestion Score'
    )
    
    return phase_colors, demand_cmap, traffic_cmap

def convert_esri_to_geojson(esri_data):
    """Convert ESRI GeoJSON format to standard GeoJSON"""
    if 'features' not in esri_data:
        return esri_data
    
    geojson_features = []
    for feature in esri_data['features']:
        attrs = feature.get('attributes', {})
        geometry = feature.get('geometry', {})
        rings = geometry.get('rings', [])
        
        if rings and len(rings) > 0:
            outer_ring = rings[0]
            polygon_coords = [[point[0], point[1]] for point in outer_ring]
            geojson_features.append({
                'type': 'Feature',
                'properties': attrs,
                'geometry': {'type': 'Polygon', 'coordinates': [polygon_coords]}
            })
    
    return {'type': 'FeatureCollection', 'features': geojson_features}

def add_selected_stations(m, selected, phase_colors):
    """Add PROPOSED stations with clear visibility"""
    print("  🔵 Adding PROPOSED stations...")
    
    n_phase1 = int(len(selected) * 0.33)
    n_phase2 = int(len(selected) * 0.67)
    
    # Use separate marker clusters for each phase
    phase1_cluster = MarkerCluster(name='📍 Phase 1 ').add_to(m)
    phase2_cluster = MarkerCluster(name='📍 Phase 2 ').add_to(m)
    phase3_cluster = MarkerCluster(name='📍 Phase 3 ').add_to(m)
    
    for _, row in selected.iterrows():
        priority = int(row['priority'])
        
        if priority <= n_phase1:
            phase = 1
            phase_name = "Phase 1 - Immediate "
            phase_desc = "Highest priority - Deploy first"
            cluster = phase1_cluster
        elif priority <= n_phase2:
            phase = 2
            phase_name = "Phase 2 - Short-term "
            phase_desc = "Medium priority - Deploy second"
            cluster = phase2_cluster
        else:
            phase = 3
            phase_name = "Phase 3 - Long-term "
            phase_desc = "Future expansion"
            cluster = phase3_cluster
        
        # Enhanced demand score retrieval with robust fallbacks
        demand_score = row.get('demand_score', row.get('demand_ann', row.get('svm_probability', 0)))
        demand_level = "High" if demand_score > 0.7 else "Medium" if demand_score > 0.4 else "Low"
        
        popup_html = f"""
        <div style="font-family: Arial; min-width: 260px;">
            <div style="background-color: {phase_colors[phase]}; color: white; padding: 10px; border-radius: 5px;">
                <b>🆕 PROPOSED STATION #{priority}</b><br>
                <b>{phase_name}</b>
            </div>
            <div style="padding: 10px; background-color: #f8f9fa;">
                <b>🏘️ Seniūnija:</b> {row['seniunija']}<br>
                <b>📊 Demand Score:</b> {demand_score:.4f} ({demand_level})<br>
                <b>🎯 SVM Probability:</b> {row.get('svm_probability', 0):.4f}<br>
                <b>🚦 Traffic Score:</b> {row.get('jam_score_norm', 0):.4f}<br>
                <b>📏 Distance to existing:</b> {row.get('dist_to_existing_m', 0):.0f} m<br>
                <b>🌐 Coordinates:</b> {row['lat']:.5f}, {row['lon']:.5f}
            </div>
            <div style="background-color: #e9ecef; padding: 5px; border-radius: 3px; font-size: 11px;">
                <b>💡 {phase_desc}</b>
            </div>
        </div>
        """
        
        icon_html = f"""
        <div style="background-color: {phase_colors[phase]}; color: white; font-weight: bold;
                    font-size: 16px; width: 36px; height: 36px; border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
            {priority}
        </div>
        """
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=380),
            icon=folium.DivIcon(icon_size=(36, 36), icon_anchor=(18, 18), html=icon_html)
        ).add_to(cluster)
        
        # Add 400m service area
        folium.Circle(
            location=[row['lat'], row['lon']], radius=400,
            color=phase_colors[phase], fill=True, fill_color=phase_colors[phase],
            fill_opacity=0.15, weight=3, opacity=0.8,
            popup=f"{phase_name} - 400m service area"
        ).add_to(m)

def add_existing_chargers(m):
    """Add EXISTING charging stations - clearly distinguished"""
    print("  ⚫ Adding EXISTING chargers...")
    
    if not EXISTING_EVCS.exists():
        print(f"  ⚠️ Existing EVCS file not found")
        return
    
    try:
        with open(EXISTING_EVCS, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        existing_cluster = MarkerCluster(
            name='🔌 Existing Chargers',
            overlay=True,
            control=True
        ).add_to(m)
        
        count = 0
        for feature in data.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') == 'Point':
                coords = geom.get('coordinates', [])
                if len(coords) >= 2:
                    lat, lon = coords[1], coords[0]
                    props = feature.get('properties', {})
                    name = props.get('name', 'Unknown')
                    operator = props.get('operator', 'Unknown')
                    
                    popup_html = f"""
                    <div style="font-family: Arial; min-width: 200px;">
                        <div style="background-color: #6c757d; color: white; padding: 8px; border-radius: 5px;">
                            <b>🔌 EXISTING CHARGING STATION</b>
                        </div>
                        <div style="padding: 8px;">
                            <b>Name:</b> {name}<br>
                            <b>Operator:</b> {operator}<br>
                            <b>Coordinates:</b> {lat:.5f}, {lon:.5f}
                        </div>
                    </div>
                    """
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=8,
                        popup=folium.Popup(popup_html, max_width=300),
                        color='#495057',
                        fill=True,
                        fill_color='#6c757d',
                        fill_opacity=0.9,
                        weight=2,
                        stroke=True
                    ).add_to(existing_cluster)
                    count += 1
        
        print(f"  ✓ Added {count} EXISTING chargers (gray markers)")
    except Exception as e:
        print(f"  ⚠️ Error loading existing chargers: {e}")

def add_unsuitable_sites(m, candidates):
    """Add SVM-filtered unsuitable candidate sites as an optional layer"""
    if 'svm_classification' not in candidates.columns:
        print("  ⚠️ 'svm_classification' column not found in candidates. Skipping background layer.")
        return
        
    print("  ❌ Adding SVM-filtered unsuitable sites layer...")
    unsuitable_sites = candidates[candidates['svm_classification'] == 0]
    
    if len(unsuitable_sites) == 0:
        print("  ✓ No filtered-out sites to map.")
        return
        
    unsuitable_cluster = MarkerCluster(
        name='❌ Unsuitable Sites (SVM Filtered)',
        overlay=True,
        control=True,
        show=False  # Hidden by default to avoid initial map clutter
    ).add_to(m)
    
    for _, row in unsuitable_sites.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; min-width: 220px;">
            <div style="background-color: #5c636a; color: white; padding: 6px; border-radius: 4px;">
                <b>❌ SVM EXCLUDED SITE</b>
            </div>
            <div style="padding: 8px; font-size: 12px; background-color: #f8f9fa;">
                <b>🏘️ Seniūnija:</b> {row.get('seniunija', 'Unknown')}<br>
                <b>📊 Predicted Demand:</b> {row['demand_score']:.4f}<br>
                <b>🎯 SVM Probability:</b> {row.get('svm_probability', 0):.4f}<br>
                <b>🌐 Coordinates:</b> {row['lat']:.5f}, {row['lon']:.5f}
            </div>
            <div style="background-color: #f8d7da; color: #721c24; padding: 5px; border-radius: 3px; font-size: 11px;">
                ⚠️ <i>Discovered unsuitable based on spatial boundary constraints or low multifeature interaction weights.</i>
            </div>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            popup=folium.Popup(popup_html, max_width=260),
            color='#721c24',
            fill=True,
            fill_color='#f8d7da',
            fill_opacity=0.5,
            weight=1
        ).add_to(unsuitable_cluster)
        
    print(f"  ✓ Added {len(unsuitable_sites)} SVM-filtered sites (hidden by default)")

def add_traffic_hotspots(m, traffic, traffic_cmap):
    """Add traffic congestion hotspots"""
    if traffic is None or len(traffic) == 0:
        return
    
    print("  🔴 Adding traffic hotspots...")
    top_traffic = traffic.nlargest(50, 'jam_weight')
    
    for _, row in top_traffic.iterrows():
        popup_html = f"""
        <div style="font-family: Arial;">
            <b>🚦 Traffic Hotspot</b><br>
            <b>Location:</b> {row['location_name'][:50]}<br>
            <b>Events:</b> {int(row['event_count'])}<br>
            <b>Total duration:</b> {row['total_duration']:.0f} min<br>
            <b>Avg duration:</b> {row['avg_duration']:.0f} min
        </div>
        """
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=row['jam_score'] * 8,
            popup=popup_html,
            color='#dc3545',
            fill=True,
            fill_color='#dc3545',
            fill_opacity=0.5,
            weight=1,
            stroke=True
        ).add_to(m)

def add_seniunija_boundaries(m):
    """Add seniūnija boundaries"""
    print("  🗺️ Adding seniūnija boundaries...")
    
    if not SENIUNIJOS_GEOJSON.exists():
        print(f"  ⚠️ GeoJSON not found")
        return
    
    with open(SENIUNIJOS_GEOJSON, 'r', encoding='utf-8') as f:
        esri_data = json.load(f)
    
    geojson_data = convert_esri_to_geojson(esri_data)
    
    folium.GeoJson(
        geojson_data,
        name='🏘️ Seniūnijos',
        style_function=lambda x: {
            'fillColor': '#ffffff',
            'color': '#2c3e50',
            'weight': 2,
            'fillOpacity': 0.05,
            'dashArray': '5, 5'
        },
        highlight_function=lambda x: {
            'fillColor': '#3498db',
            'color': '#e74c3c',
            'weight': 3,
            'fillOpacity': 0.15
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['SENIUNIJA'],
            aliases=['Seniūnija:'],
            localize=True,
            sticky=True,
            style="""
                background-color: white;
                border: 1px solid black;
                border-radius: 5px;
                padding: 5px;
                font-family: Arial;
                font-size: 12px;
            """
        )
    ).add_to(m)

def add_demand_heatmap(m, candidates):
    """Add demand heatmap"""
    print("  🔥 Adding demand heatmap...")
    
    # Use demand_score (already standardized in load_all_data)
    heat_data = [[row['lat'], row['lon'], row['demand_score']] for _, row in candidates.iterrows()]
    HeatMap(
        heat_data,
        radius=30,
        blur=15,
        max_zoom=12,
        min_opacity=0.2,
        gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'},
        name='🔥 Demand Heatmap'
    ).add_to(m)

def add_controls(m):
    """Add interactive controls"""
    plugins.Fullscreen(position='topleft', title='Fullscreen', title_cancel='Exit Fullscreen').add_to(m)
    plugins.MiniMap(toggle_display=True, position='bottomright', width=150, height=150, zoom_level_offset=-4).add_to(m)
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    plugins.MousePosition().add_to(m)
    plugins.MeasureControl(position='topleft', primary_length_unit='meters', secondary_length_unit='kilometers').add_to(m)
    plugins.LocateControl().add_to(m)

def create_map():
    """Create the complete interactive map"""
    print("\n" + "="*60)
    print("🗺️  CREATING INTERACTIVE DEPLOYMENT MAP")
    print("="*60)
    
    selected, candidates, traffic = load_all_data()
    phase_colors, demand_cmap, traffic_cmap = create_color_maps()
    
    center_lat = selected['lat'].mean()
    center_lon = selected['lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron', control_scale=True)
    
    # Add base tile layers
    folium.TileLayer('OpenStreetMap', name='🗺️ Street Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='🌙 Dark Mode').add_to(m)
    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri',
        name='🛰️ Satellite'
    ).add_to(m)
    
    # Add layers
    print("\n📊 Adding layers to map...")
    add_seniunija_boundaries(m)
    add_existing_chargers(m)
    add_unsuitable_sites(m, candidates)  # NEW: SVM Filtered Layer
    add_traffic_hotspots(m, traffic, traffic_cmap)
    add_demand_heatmap(m, candidates)
    add_selected_stations(m, selected, phase_colors)
    add_controls(m)
    
    # Title panel
    n_phase1 = int(len(selected) * 0.33)
    n_phase2 = int(len(selected) * 0.67)
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; z-index: 1000; background: white;
                padding: 15px; border-radius: 10px; border: 2px solid #2c3e50; 
                box-shadow: 2px 2px 15px rgba(0,0,0,0.3);
                font-family: Arial, sans-serif; min-width: 340px;">
        <h3 style="margin: 0 0 10px 0; color: #2c3e50;">🔌 EV Charging Station Deployment Plan</h3>
        <div style="font-size: 13px;">
            <div><span style="color: #E63946; font-weight: bold; font-size: 16px;">●</span> <b>Phase 1 </b> - Immediate: {n_phase1} stations</div>
            <div><span style="color: #F4A261; font-weight: bold; font-size: 16px;">●</span> <b>Phase 2 </b> - Short-term: {n_phase2 - n_phase1} stations</div>
            <div><span style="color: #2A9D8F; font-weight: bold; font-size: 16px;">●</span> <b>Phase 3 </b> - Future: {len(selected) - n_phase2} stations</div>
            <div style="margin-top: 8px;"><span style="color: #6c757d; font-weight: bold; font-size: 16px;">⚫</span> <b>Existing Chargers</b> (already installed)</div>
            <div><span style="color: #721c24; font-weight: bold; font-size: 14px;">❌</span> <b>SVM Unsuitable Sites</b> (Toggled off by default)</div>
            <div><span style="color: #dc3545; font-weight: bold;">🔴</span> <b>Traffic Hotspots</b> (congestion areas)</div>
            <div><span style="background: linear-gradient(90deg, blue, green, yellow, red);">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> <b>Demand Heatmap</b> (blue=low, red=high)</div>
            <div><span style="border: 1px solid #E63946;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> <b>400m Service Area</b></div>
        </div>
        <div style="margin-top: 10px; font-size: 11px; color: #666; border-top: 1px solid #ddd; padding-top: 8px;">
            📍 {len(selected)} proposed stations | 🏘️ {selected['seniunija'].nunique()} seniūnijas | 🎯 Optimized by GA
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Legend Panel with wider formatting to prevent text wrapping
    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000; background: white;
                padding: 12px; border-radius: 8px; border: 1px solid #ccc; 
                font-family: Arial; font-size: 12px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.2); min-width: 200px;">
        <b>📖 Quick Legend</b><br>
        <hr style="margin: 5px 0;">
        <span style="color: #E63946; font-weight: bold;">●</span> <b>PROPOSED</b> (New Stations)<br>
        <span style="color: #F4A261; font-weight: bold;">●</span> Phase 1 (Immediate Priority)<br>
        <span style="color: #2A9D8F; font-weight: bold;">●</span> Phase 2 (Short-term Rollout)<br>
        <span style="color: #6c757d; font-weight: bold;">●</span> Phase 3 (Future Expansion)<br>
        <span style="color: #6c757d;">⚫</span> <b>EXISTING</b> (Current Baseline)<br>
        <span style="color: #721c24; font-weight: bold;">❌</span> <b>SVM Filtered</b> (Unsuitable)<br>
        <span style="color: #dc3545;">🔴</span> Traffic Hotspot<br>
        <span style="border: 1px solid #E63946;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> 400m Walking Buffer<br>
        <hr style="margin: 5px 0;">
        <div style="font-size: 10px; color: #666;">
            💡 Click markers for details<br>
            ┆ Hover over district boundaries<br>
            📐 Measure tool in top-left
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    print(f"\n💾 Saving map to: {OUTPUT_MAP}")
    m.save(OUTPUT_MAP)
    print(f"✅ Map created! File size: {OUTPUT_MAP.stat().st_size / 1024:.1f} KB")
    
    return m

def create_summary_report():
    """Create comprehensive summary report"""
    print("\n📊 Creating summary report...")
    
    selected = pd.read_csv(SELECTED_SITES)
    candidates = pd.read_csv(CANDIDATES_FILE)
    
    # Phase assignment
    n_phase1 = int(len(selected) * 0.33)
    n_phase2 = int(len(selected) * 0.67)
    selected['phase'] = 3
    selected.loc[selected['priority'] <= n_phase1, 'phase'] = 1
    selected.loc[(selected['priority'] > n_phase1) & (selected['priority'] <= n_phase2), 'phase'] = 2
    
    # Standardize demand score column for summary
    selected['demand_score_clean'] = selected.get('demand_score', selected.get('demand_ann', selected.get('svm_probability', 0)))
    
    # Phase summary
    phase_summary = selected.groupby('phase').agg({
        'priority': 'count',
        'demand_score_clean': 'mean',
        'svm_probability': 'mean',
        'jam_score_norm': 'mean',
        'dist_to_existing_m': 'mean'
    }).round(4)
    phase_summary.columns = ['stations', 'avg_demand', 'avg_svm', 'avg_traffic', 'avg_distance_m']
    
    # Seniūnija summary
    sen_summary = selected.groupby('seniunija').size().sort_values(ascending=False)
    
    # Overall stats
    overall_stats = {
        'total_stations': int(len(selected)),
        'unique_seniunijas': int(selected['seniunija'].nunique()),
        'avg_demand': float(selected['demand_score_clean'].mean()),
        'avg_svm': float(selected['svm_probability'].mean()),
        'avg_traffic': float(selected['jam_score_norm'].mean()),
        'min_distance_existing': float(selected['dist_to_existing_m'].min()),
        'total_candidates': int(len(candidates))
    }
    
    # Save
    phase_summary.to_csv("results/deployment_phase_summary.csv")
    sen_summary.to_csv("results/seniunija_station_summary.csv")
    
    with open("results/deployment_overall_stats.json", "w") as f:
        json.dump(overall_stats, f, indent=2)
    
    print("\n" + "="*60)
    print("📊 DEPLOYMENT SUMMARY")
    print("="*60)
    print(f"✅ PROPOSED stations: {overall_stats['total_stations']}")
    print(f"📍 Unique seniūnijas covered: {overall_stats['unique_seniunijas']}")
    print(f"📈 Average demand score: {overall_stats['avg_demand']:.4f}")
    print(f"🎯 Average SVM probability: {overall_stats['avg_svm']:.4f}")
    print(f"🚦 Average traffic score: {overall_stats['avg_traffic']:.4f}")
    print(f"\n📊 Stations per seniūnija (Top 10):")
    for sen, count in sen_summary.head(10).items():
        bar = "█" * min(count, 20)
        print(f"  {sen:20s}: {count} {bar}")

if __name__ == "__main__":
    m = create_map()
    create_summary_report()
    
    print("\n" + "="*60)
    print("🎉 VISUALIZATION COMPLETE!")
    print("="*60)
    print("\n📁 Output files:")
    print("   • 🌐 Interactive map: results/evcs_deployment_map.html")
    print("   • 📊 Phase summary: results/deployment_phase_summary.csv")
    print("   • 🏘️ Seniūnija summary: results/seniunija_station_summary.csv")
    print("   • 📈 Overall stats: results/deployment_overall_stats.json")
    print("\n💡 Tips:")
    print("   • 🔴 RED markers = PROPOSED new stations (numbered by priority)")
    print("   • ⚫ GRAY markers = EXISTING charging stations")
    print("   • ❌ RED/GRAY CROSSES = SVM Filtered out candidate locations")
    print("   • Click any marker for detailed information")
    print("   • Use the layer control (top-right) to toggle layers")
    print("\n🌐 Opening map in browser...")
    webbrowser.open(f'file://{OUTPUT_MAP.absolute()}')