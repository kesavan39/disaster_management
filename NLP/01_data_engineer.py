"""
================================================================================
ROLE 01: DATA ENGINEER (NLP PIPELINE)
Disaster Response Coordination System - Stage 03 Natural Language Processing
================================================================================
Responsibilities:
- Load 20,000-record Excel dataset (Data/Disaster_Response_NLP_Dataset_20000_Records.xlsx).
- Clean raw text: missing value handling, whitespace trimming, noise handling.
- Format labels & target entities: Urgency, Disaster Type, Location, People Affected, Resource.
- Perform stratified train/test split (80/20) and export serialized artifacts.
================================================================================
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class NLPDataEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "Data")
        self.nlp_dir = os.path.join(self.base_dir, "NLP")
        self.models_dir = os.path.join(self.nlp_dir, "saved_models")
        
        os.makedirs(self.nlp_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.excel_path = os.path.join(self.data_dir, "Disaster_Response_NLP_Dataset_20000_Records.xlsx")

    def clean_text(self, text):
        """Cleans raw text message."""
        if pd.isna(text) or not str(text).strip():
            return "no text message provided"
        text = str(text).strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def prepare_dataset(self, test_size=0.2, random_state=42):
        """Loads and prepares full 20,000 record NLP dataset."""
        print(f"[NLP Data Engineer] Loading NLP dataset from {self.excel_path}...")
        df = pd.read_excel(self.excel_path)
        
        print(f"[NLP Data Engineer] Dataset Shape: {df.shape}")
        
        # Clean text
        df['cleaned_message'] = df['message'].apply(self.clean_text)
        
        # Normalize labels
        df['urgency'] = df['urgency'].astype(str).str.upper().str.strip()
        df['disaster_type'] = df['disaster_type'].astype(str).str.strip()
        df['location'] = df['location'].fillna('Unknown Location').astype(str).str.strip()
        df['people_affected'] = df['people_affected'].fillna(0).astype(int)
        df['resource_required'] = df['resource_required'].fillna('none').astype(str).str.strip()
        df['recommended_action'] = df['recommended_action'].fillna('Monitor situation').astype(str).str.strip()
        df['source'] = df['source'].fillna('Direct Message').astype(str).str.strip()
        df['is_noisy'] = df['is_noisy'].fillna(False).astype(bool)

        # Train/Test Split
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['urgency']
        )
        
        print(f"[NLP Data Engineer] Created Train Set: {len(train_df)} records | Test Set: {len(test_df)} records.")
        
        dataset_meta = {
            "total_records": len(df),
            "train_records": len(train_df),
            "test_records": len(test_df),
            "columns": df.columns.tolist(),
            "urgency_distribution": df['urgency'].value_counts().to_dict(),
            "disaster_type_distribution": df['disaster_type'].value_counts().to_dict(),
            "sources": df['source'].unique().tolist()
        }

        # Save artifacts
        joblib.dump({
            "train_df": train_df,
            "test_df": test_df,
            "full_df": df
        }, os.path.join(self.models_dir, "nlp_dataset.pkl"))

        with open(os.path.join(self.models_dir, "dataset_metadata.json"), "w") as f:
            json.dump(dataset_meta, f, indent=2)

        print("[NLP Data Engineer] NLP Data Pipeline completed successfully!\n")
        return train_df, test_df, dataset_meta

    def run_pipeline(self):
        """Runs Data Engineering Pipeline."""
        print("==========================================================")
        print("[NLP Data Engineer] Starting NLP Data Engineering...")
        print("==========================================================")
        return self.prepare_dataset()

if __name__ == "__main__":
    engineer = NLPDataEngineer()
    engineer.run_pipeline()
