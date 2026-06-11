import pandas as pd

def add_emissions(df, energy_col="energy_j", carbon_intensity_gco2_per_kwh=200):
    df = df.copy()

    df["energy_kwh"] = (df[energy_col] / 3_600_000).round(8)
    
    df["emissions_kgco2e"] = (df["energy_kwh"] * carbon_intensity_gco2_per_kwh / 1000).round(8)

    return df
