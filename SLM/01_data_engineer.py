"""
================================================================================
ROLE 01: DATA ENGINEER (SLM PIPELINE)
Disaster Response Coordination System - Stage 04 Small Language Model (SLM)
================================================================================
Responsibilities:
- Load 20,000 disaster report-summary pairs from Data/SLM_Disaster_Report_Summary_Dataset_20000.xlsx.
- Remove duplicate/messy pairs (pair_status == 'Yes', quality_flag == 'Verified').
- Categorize report length tiers (Short, Medium, Long) & calculate target briefing length metrics.
- Export clean serialized dataset to SLM/saved_models/slm_dataset.pkl.
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

class SLMDataEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.base_dir, "Data", "SLM_Disaster_Report_Summary_Dataset_20000.xlsx")
        self.slm_dir = os.path.join(self.base_dir, "SLM")
        self.models_dir = os.path.join(self.slm_dir, "saved_models")
        os.makedirs(self.models_dir, exist_ok=True)

    def load_and_clean_data(self):
        """Loads and cleans 20,000 disaster report-summary pairs."""
        print("==========================================================")
        print("[SLM Data Engineer] Loading and cleaning 20,000 records...")
        print("==========================================================")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.data_path}")

        df = pd.read_excel(self.data_path)
        print(f"[SLM Data Engineer] Raw Dataset Shape: {df.shape}")

        # Clean text columns
        text_cols = ['report_text', 'reference_summary', 'disaster_type', 'urgency', 'location', 'resource_needed']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Filter verified pairs
        valid_df = df[(df['pair_status'] == 'Yes') & (df['quality_flag'] == 'Verified')].copy()
        if len(valid_df) < 5000:
            print("[SLM Data Engineer] Warning: Quality filter too strict, falling back to non-null pairs...")
            valid_df = df[df['report_text'].notnull() & df['reference_summary'].notnull()].copy()

        # Compute word lengths
        valid_df['report_word_count'] = valid_df['report_text'].apply(lambda x: len(x.split()))
        valid_df['summary_word_count'] = valid_df['reference_summary'].apply(lambda x: len(x.split()))
        valid_df['compression_ratio'] = (1.0 - (valid_df['summary_word_count'] / (valid_df['report_word_count'] + 1e-5))) * 100.0

        # Assign length tier
        def assign_tier(w):
            if w < 120:
                return 'Short'
            elif w <= 250:
                return 'Medium'
            else:
                return 'Long'

        valid_df['length_tier'] = valid_df['report_word_count'].apply(assign_tier)

        # Separate Train and Test splits based on dataset 'split' column
        if 'split' in valid_df.columns:
            train_df = valid_df[valid_df['split'] == 'train'].copy()
            test_df = valid_df[valid_df['split'] == 'test'].copy()
        else:
            train_df = valid_df.iloc[:16000].copy()
            test_df = valid_df.iloc[16000:].copy()

        print(f"[SLM Data Engineer] Clean Verified Dataset Size: {len(valid_df)}")
        print(f"[SLM Data Engineer] Train Set: {len(train_df)} | Test Set: {len(test_df)}")

        # Save to pkl
        out_pkl = os.path.join(self.models_dir, "slm_dataset.pkl")
        joblib.dump({
            'full_df': valid_df,
            'train_df': train_df,
            'test_df': test_df
        }, out_pkl)

        print(f"[SLM Data Engineer] Saved clean dataset pkl to: {out_pkl}\n")
        return valid_df, train_df, test_df

if __name__ == "__main__":
    engineer = SLMDataEngineer()
    engineer.load_and_clean_data()
