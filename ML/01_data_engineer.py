"""
================================================================================
ROLE 01: DATA ENGINEER
Disaster Response Coordination System - Stage 01 Machine Learning
================================================================================
Responsibilities:
- Stitch gauge, weather, infrastructure, and 911 call logs into a clean master dataset.
- Implement 72-hour rolling window metrics per district/river station.
- Engineer domain-specific features (call surges, vulnerability index, resource pressure).
- Generate clean, stratified train/test datasets for ML models.
================================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataEngineer:
    def __init__(self, raw_data_path=None, output_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_path = raw_data_path or os.path.join(base_dir, "Data", "Raw data.xlsx")
        self.output_dir = output_dir or os.path.join(base_dir, "ML")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_raw_data(self):
        """Loads raw dataset from Excel sheet, falling back to master_dataset.csv if locked."""
        print(f"[Data Engineer] Loading raw data from {self.raw_data_path}...")
        master_path = os.path.join(self.output_dir, "master_dataset.csv")
        
        try:
            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f"Raw data file not found at: {self.raw_data_path}")
            df = pd.read_excel(self.raw_data_path, sheet_name="ML_Training_Data")
            print(f"[Data Engineer] Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        except Exception as e:
            if os.path.exists(master_path):
                print(f"[Data Engineer] Raw file read restricted ({e}). Loading existing master dataset from {master_path}...")
                df = pd.read_csv(master_path)
                return df
            else:
                raise e

    def clean_and_engineer_features(self, df):
        """Cleans data and constructs rolling & domain features."""
        print("[Data Engineer] Cleaning data & engineering features...")
        df = df.copy()
        
        # Sort chronologically by district and timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['district', 'timestamp']).reset_index(drop=True)
        
        # 1. Rolling 72-hour window features per district (12 6-hr steps = 72 hours)
        df['water_level_72h_avg'] = df.groupby('district')['water_level_m'].transform(
            lambda x: x.rolling(window=12, min_periods=1).mean()
        )
        df['water_level_72h_max'] = df.groupby('district')['water_level_m'].transform(
            lambda x: x.rolling(window=12, min_periods=1).max()
        )
        df['water_level_72h_std'] = df.groupby('district')['water_level_m'].transform(
            lambda x: x.rolling(window=12, min_periods=1).std().fillna(0)
        )
        
        # 2. Rainfall Intensity Ratio (6h vs 24h accumulation surge)
        df['rainfall_intensity_ratio'] = df['rainfall_mm_6h'] / (df['rainfall_mm_24h'] + 1e-5)
        
        # 3. Emergency Call Surge & Density (calls per 10k exposed population)
        df['call_density_per_10k'] = (df['emergency_calls_6h'] / (df['estimated_exposed_population'] + 1.0)) * 10000.0
        df['call_surge_6h'] = df.groupby('district')['emergency_calls_6h'].diff().fillna(0)
        
        # 4. Infrastructure Stress Index
        df['infrastructure_stress'] = df['road_closures'] * (1.0 + df['power_outage_probability'])
        
        # 5. Resource Pressure Index
        df['resource_pressure'] = df['estimated_exposed_population'] / (
            df['shelter_beds_available'] + df['ambulances_available'] * 15 + df['rescue_teams_available'] * 50 + 1.0
        )
        
        # Fill any remaining NaNs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        print(f"[Data Engineer] Feature engineering complete. Total features: {df.shape[1]}")
        return df

    def save_and_split(self, df):
        """Saves master dataset and stratified train/test split."""
        master_path = os.path.join(self.output_dir, "master_dataset.csv")
        df.to_csv(master_path, index=False)
        print(f"[Data Engineer] Master dataset saved to: {master_path}")
        
        # Stratified Train/Test split based on risk_level
        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=42,
            stratify=df['risk_level']
        )
        
        train_path = os.path.join(self.output_dir, "train_data.csv")
        test_path = os.path.join(self.output_dir, "test_data.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"[Data Engineer] Train dataset ({train_df.shape[0]} rows) saved to: {train_path}")
        print(f"[Data Engineer] Test dataset ({test_df.shape[0]} rows) saved to: {test_path}")
        
        return master_path, train_path, test_path

    def process_data(self):
        """Pipeline execution for Data Engineer."""
        df_raw = self.load_raw_data()
        df_clean = self.clean_and_engineer_features(df_raw)
        master_path, train_path, test_path = self.save_and_split(df_clean)
        return df_clean

if __name__ == "__main__":
    engineer = DataEngineer()
    engineer.process_data()
