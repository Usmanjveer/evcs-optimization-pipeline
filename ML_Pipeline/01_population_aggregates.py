
import pandas as pd
import numpy as np
from pathlib import Path
from config import RESIDENTS_CSV, RESULTS_DIR, CHUNK_SIZE

def age_group(birth_year: float, ref_year: int = 2024):
    if pd.isna(birth_year):
        return None
    try:
        age = ref_year - int(birth_year)
        if age < 0 or age > 120:
            return None
    except Exception:
        return None
    
    if age < 25:
        return "Under 25"
    elif age < 65:
        return "25-64"
    else:
        return "65+"


def load_real_population_data():
    """Our Eisting Population data from vilnius_population_by_seniunija.csv"""
    real_pop_file = Path("Data_set/vilnius_population_by_seniunija.csv")
    
    if not real_pop_file.exists():
        print(f" file not found: {real_pop_file}")
        print("   use area-based distribution instead.")
        return None
    
    df = pd.read_csv(real_pop_file)
    
    # Clean seniūnija names
    name_mapping = {
        'Verkiai': 'Verkiai',
        'Zirmunai': 'Žirmūnai',
        'Pasilaiciai': 'Pašilaičiai',
        'Antakalnis': 'Antakalnis',
        'Fabijoniskes': 'Fabijoniškės',
        'Naujoji Vilnia': 'Naujoji Vilnia',
        'Seskine': 'Šeškinė',
        'Naujininkai': 'Naujininkai',
        'Karoliniskes': 'Karoliniškės',
        'Lazdynai': 'Lazdynai',
        'Justiniskes': 'Justiniškės',
        'Pilaite': 'Pilaitė',
        'Naujamiestis': 'Naujamiestis',
        'Vilkpede': 'Vilkpėdė',
        'Snipiskes': 'Šnipiškės',
        'Rasos': 'Rasos',
        'Zverynas': 'Žvėrynas',
        'Grigiskes': 'Grigiškės',
        'Senamiestis': 'Senamiestis',
        'Paneriai': 'Paneriai',
    }
    
    df['seniunija_clean'] = df['seniunija'].map(name_mapping)
    df['seniunija'] = df['seniunija_clean']
    
    # Remove percentage sign if present
    if 'percentage_of_city' in df.columns:
        df['percentage'] = df['percentage_of_city'].str.replace('%', '').astype(float)
    
    print(f" Loadeding population data for {len(df)} seniūnijas")
    print(f"  Total population from real data: {df['population'].sum():,}")
    
    return df


def calculate_age_shares_from_registry(residents_csv):
   
   
    print("\n  Calculating...")
    
    # Read only needed columns for efficiency
    df = pd.read_csv(residents_csv, usecols=['birth_year', 'eldership_name'])
    
    # Remove rows without district info
    df = df.dropna(subset=['eldership_name'])
    
    # Clean district names (fix encoding issues)
    name_clean = {
        'Pa≈°ilaiƒçiai': 'Pašilaičiai',
        'Fabijoni≈°kƒós': 'Fabijoniškės',
        'Senamiestis': 'Senamiestis',
        'Justini≈°kƒós': 'Justiniškės',
        '≈Ωirm≈´nai': 'Žirmūnai',
        'Naujamiestis': 'Naujamiestis',
        'Naujoji Vilnia': 'Naujoji Vilnia',
        'Karolini≈°kƒós': 'Karoliniškės',
        'Antakalnis': 'Antakalnis',
        'Lazdynai': 'Lazdynai',
        'Grigi≈°kƒós': 'Grigiškės',
        'Rasos': 'Rasos',
        'Verkiai': 'Verkiai',
        '≈Ωvƒórynas': 'Žvėrynas',
        '≈Ýe≈°kinƒó': 'Šeškinė',
        'Naujininkai': 'Naujininkai',
        'Vilkpƒódƒó': 'Vilkpėdė',
        'Paneriai': 'Paneriai',
        'Pilaitƒó': 'Pilaitė',
        'Vir≈°uli≈°kƒós': 'Viršuliškės',
        '≈Ýnipi≈°kƒós': 'Šnipiškės',
    }
    
    df['eldership_name'] = df['eldership_name'].map(name_clean).fillna(df['eldership_name'])
    
    # Calculate age
    current_year = 2024
    df['age'] = current_year - df['birth_year']
    df = df[(df['age'] >= 0) & (df['age'] <= 120)]
    def get_age_group(age):
        if age < 25:
            return 'young'
        elif age < 65:
            return 'working'
        else:
            return 'senior'
    
    df['age_group'] = df['age'].apply(get_age_group)
    # Count by district and age group
    pivot = pd.crosstab(df['eldership_name'], df['age_group'], normalize='index')
    # Ensure all three age groups exist
    for col in ['young', 'working', 'senior']:
        if col not in pivot.columns:
            pivot[col] = 0
    
    age_shares = {}
    for district in pivot.index:
        row = pivot.loc[district]
        age_shares[district] = (row['young'], row['working'], row['senior'])
    print(f"  Calculated age shares for {len(age_shares)} districts from data")
    print("\n  Calculating age share):")
    for district in list(age_shares.keys())[:5]:
        y, w, s = age_shares[district]
        print(f"    {district}: young={y:.1%}, working={w:.1%}, senior={s:.1%}")
    return age_shares


def distribute_population_by_seniunija_real(seniunijos_file, real_pop_data, total_pop, age_counts, sex_counts, age_shares):
    seniunijos = pd.read_csv(seniunijos_file)
    # Normalize seniūnija names in boundaries file
    name_mapping = {
        'Verkiai': 'Verkiai',
        'Žirmūnai': 'Žirmūnai',
        'Pašilaičiai': 'Pašilaičiai',
        'Antakalnis': 'Antakalnis',
        'Fabijoniškės': 'Fabijoniškės',
        'Naujoji Vilnia': 'Naujoji Vilnia',
        'Šeškinė': 'Šeškinė',
        'Naujininkai': 'Naujininkai',
        'Karoliniškės': 'Karoliniškės',
        'Lazdynai': 'Lazdynai',
        'Justiniškės': 'Justiniškės',
        'Pilaitė': 'Pilaitė',
        'Naujamiestis': 'Naujamiestis',
        'Vilkpėdė': 'Vilkpėdė',
        'Šnipiškės': 'Šnipiškės',
        'Rasos': 'Rasos',
        'Žvėrynas': 'Žvėrynas',
        'Grigiškės': 'Grigiškės',
        'Senamiestis': 'Senamiestis',
        'Paneriai': 'Paneriai',
    }
    
    real_pop_dict = dict(zip(real_pop_data['seniunija'], real_pop_data['population']))
    
    seniunijos['population_real'] = seniunijos['seniunija'].map(real_pop_dict)
    
    # Fill missing populations with area-based estimate
    missing_mask = seniunijos['population_real'].isna()
    if missing_mask.any():
        missing_districts = seniunijos[missing_mask]['seniunija'].tolist()
        print(f" Missing population data for: {missing_districts}")
        print(f"     Will use area-based estimate for these districts.")
        total_area = seniunijos['area_m2'].sum()
        area_based_pop = (seniunijos['area_m2'] / total_area) * total_pop
        seniunijos.loc[missing_mask, 'population_real'] = area_based_pop[missing_mask]
    
    seniunijos['population'] = seniunijos['population_real']
    
    # Calculate city-wide age shares
    young_share_city = age_counts["Under 25"] / total_pop
    working_share_city = age_counts["25-64"] / total_pop
    senior_share_city = age_counts["65+"] / total_pop
    
    # Distribute age groups using calculated shares from registry
    for seniunija in seniunijos['seniunija']:
        mask = seniunijos['seniunija'] == seniunija
        if seniunija in age_shares:
            young, working, senior = age_shares[seniunija]
            seniunijos.loc[mask, 'young_share'] = young
            seniunijos.loc[mask, 'working_share'] = working
            seniunijos.loc[mask, 'senior_share'] = senior
        else:
            seniunijos.loc[mask, 'young_share'] = young_share_city
            seniunijos.loc[mask, 'working_share'] = working_share_city
            seniunijos.loc[mask, 'senior_share'] = senior_share_city
            print(f"  No age data for {seniunija}, using city averages")
    
    # Calculate age group populations
    seniunijos['pop_under_25'] = seniunijos['population'] * seniunijos['young_share']
    seniunijos['pop_25_64'] = seniunijos['population'] * seniunijos['working_share']
    seniunijos['pop_65_plus'] = seniunijos['population'] * seniunijos['senior_share']
    for sex, count in sex_counts.items():
        seniunijos[f'pop_{sex.lower()}'] = (seniunijos['population'] / total_pop) * count
    seniunijos = seniunijos.drop(columns=['population_real'], errors='ignore')
    
    return seniunijos


def distribute_population_by_seniunija_area(seniunijos_file, total_pop, age_counts, sex_counts):
    """Fallback: Distribute population based on area """
    
    seniunijos = pd.read_csv(seniunijos_file)
    total_area = seniunijos['area_m2'].sum()
    
    # Distribute proportionally by area
    seniunijos['population'] = (seniunijos['area_m2'] / total_area) * total_pop
    
    seniunijos['young_share'] = age_counts["Under 25"] / total_pop
    seniunijos['working_share'] = age_counts["25-64"] / total_pop
    seniunijos['senior_share'] = age_counts["65+"] / total_pop
    
    seniunijos['pop_under_25'] = seniunijos['population'] * seniunijos['young_share']
    seniunijos['pop_25_64'] = seniunijos['population'] * seniunijos['working_share']
    seniunijos['pop_65_plus'] = seniunijos['population'] * seniunijos['senior_share']
    
    for sex, count in sex_counts.items():
        seniunijos[f'pop_{sex.lower()}'] = (seniunijos['population'] / total_pop) * count
    
    return seniunijos


def main():
    if not RESIDENTS_CSV.exists():
        raise FileNotFoundError(f"Missing residents CSV: {RESIDENTS_CSV}")
    
    print("\nReading resident data...")
    
    total_rows = 0
    age_counts = {"Under 25": 0, "25-64": 0, "65+": 0}
    sex_counts = {}
    all_registry_data = []
    
    usecols = ["birth_year", "sex", "eldership_name"]
    
    for chunk in pd.read_csv(RESIDENTS_CSV, usecols=usecols, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        all_registry_data.append(chunk)
        
        chunk["age_group"] = chunk["birth_year"].apply(age_group)
        vc_age = chunk["age_group"].value_counts(dropna=True).to_dict()
        for k, v in vc_age.items():
            if k in age_counts:
                age_counts[k] += int(v)
        
     
        vc_sex = chunk["sex"].value_counts(dropna=True).to_dict()
        for k, v in vc_sex.items():
            sex_counts[k] = sex_counts.get(k, 0) + int(v)
        
        if total_rows % 100000 == 0:
            print(f"  Processed rows: {total_rows:,}")
    
    df_registry = pd.concat(all_registry_data, ignore_index=True)
    
    # Create output DataFrames
    df_age = pd.DataFrame([{"age_group": k, "population": v} for k, v in age_counts.items()])
    df_sex = pd.DataFrame([{"sex": k, "population": v} for k, v in sex_counts.items()])
    
    df_age.to_csv(RESULTS_DIR / "population_by_age_group.csv", index=False)
    df_sex.to_csv(RESULTS_DIR / "population_by_sex.csv", index=False)
    
    total_pop = sum(age_counts.values())
    
    print(f"\nTotal population processed: {total_pop:,}")
    print("\nAge distribution:")
    print(df_age.to_string(index=False))
    print("\nSex distribution:")
    print(df_sex.to_string(index=False))
    
    # Calculate 
    under_25_share = age_counts["Under 25"] / total_pop
    working_share = age_counts["25-64"] / total_pop
    senior_share = age_counts["65+"] / total_pop
    
    print(f"\nAge shares (Vilnius averages):")
    print(f"  Under 25: {under_25_share:.3f} ({under_25_share*100:.1f}%)")
    print(f"  25-64: {working_share:.3f} ({working_share*100:.1f}%)")
    print(f"  65+: {senior_share:.3f} ({senior_share*100:.1f}%)")
    
    # Save 
    shares_df = pd.DataFrame([
        {'age_group': 'Under 25', 'share': under_25_share},
        {'age_group': '25-64', 'share': working_share},
        {'age_group': '65+', 'share': senior_share}
    ])
    shares_df.to_csv(RESULTS_DIR / "population_age_shares.csv", index=False)
    
    
    age_shares = calculate_age_shares_from_registry(RESIDENTS_CSV)
    
    # population by seniūnija
    seniunijos_file = RESULTS_DIR / "vilnius_seniunijos.csv"
    if seniunijos_file.exists():
        print("\nDistributing population by seniūnija...")
        
        # Try to load real population data
        real_pop_data = load_real_population_data()
        
        if real_pop_data is not None:
            print("  Using REAL population data from vilnius_population_by_seniunija.csv")
            pop_by_seniunija = distribute_population_by_seniunija_real(
                seniunijos_file, real_pop_data, total_pop, age_counts, sex_counts, age_shares
            )
        else:
            print("  Using area-based population distribution (fallback)")
            pop_by_seniunija = distribute_population_by_seniunija_area(
                seniunijos_file, total_pop, age_counts, sex_counts
            )
        
        pop_by_seniunija.to_csv(RESULTS_DIR / "population_by_seniunija.csv", index=False)
        print(f"  Saved population by seniūnija to: {RESULTS_DIR}/population_by_seniunija.csv")
        
        pop_by_seniunija['area_km2'] = pop_by_seniunija['area_m2'] / 1_000_000
        pop_by_seniunija['pop_density'] = pop_by_seniunija['population'] / pop_by_seniunija['area_km2']
        
        print("\n  Top seniūnijas by population:")
        display_cols = ['seniunija', 'population', 'area_km2', 'pop_density', 'working_share']
        print(pop_by_seniunija.nlargest(10, 'population')[display_cols].to_string(index=False))
        
        density_check = pop_by_seniunija[['seniunija', 'pop_density']].sort_values('pop_density', ascending=False)
        print(density_check.head(10).to_string(index=False))
        
        # Verify working_share variation
        working_check = pop_by_seniunija[['seniunija', 'working_share']].sort_values('working_share', ascending=False)
        print(working_check.head(10).to_string(index=False))
     
         
        
    else:
        print(f"\n Seniūnija file not found. Run 02_boundaries_to_csv.py first.")
    
    print(f"\n Saved to: {RESULTS_DIR}/population_by_age_group.csv")


if __name__ == "__main__":
    main()