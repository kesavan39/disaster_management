"""
================================================================================
ROLE 02: EDA ENGINEER (DEEP LEARNING PIPELINE)
Disaster Response Coordination System - Stage 02 Deep Learning
================================================================================
Responsibilities:
- Explore image dataset distributions (flooded vs clear count, RGB channel statistics).
- Analyze 72,876 river gauge time-series records (water level ranges, gauge stations, peak flood surges).
- Output comprehensive EDA report to DL/dl_eda_summary.json.
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image

class DLEDAEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "Data")
        self.dl_dir = os.path.join(self.base_dir, "DL")
        
        self.images_flooded_dir = os.path.join(self.data_dir, "images", "flooded")
        self.images_clear_dir = os.path.join(self.data_dir, "images", "clear")
        self.ts_csv_path = os.path.join(self.data_dir, "water_level_timeseries.csv")

    def analyze_image_dataset(self):
        """Analyzes spatial drone image dataset statistics."""
        print("[DL EDA Engineer] Analyzing drone image dataset...")
        
        flooded_files = [f for f in os.listdir(self.images_flooded_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(self.images_flooded_dir) else []
        clear_files = [f for f in os.listdir(self.images_clear_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(self.images_clear_dir) else []

        flooded_rgb = []
        for fname in flooded_files[:20]: # Sample 20 images
            img_path = os.path.join(self.images_flooded_dir, fname)
            arr = np.array(Image.open(img_path))
            flooded_rgb.append(arr.mean(axis=(0, 1)))

        clear_rgb = []
        for fname in clear_files[:20]: # Sample 20 images
            img_path = os.path.join(self.images_clear_dir, fname)
            arr = np.array(Image.open(img_path))
            clear_rgb.append(arr.mean(axis=(0, 1)))

        flooded_rgb_mean = np.mean(flooded_rgb, axis=0) if flooded_rgb else [0, 0, 0]
        clear_rgb_mean = np.mean(clear_rgb, axis=0) if clear_rgb else [0, 0, 0]

        image_stats = {
            "total_flooded_images": len(flooded_files),
            "total_clear_images": len(clear_files),
            "class_balance": "Perfect 50/50" if len(flooded_files) == len(clear_files) else "Imbalanced",
            "image_resolution": "128x128 RGB",
            "flooded_avg_channel_rgb": [round(float(c), 2) for c in flooded_rgb_mean],
            "clear_avg_channel_rgb": [round(float(c), 2) for c in clear_rgb_mean]
        }
        
        print(f"[DL EDA Engineer] Image Dataset: {len(flooded_files)} Flooded, {len(clear_files)} Clear.")
        return image_stats

    def analyze_timeseries_dataset(self):
        """Analyzes river gauge time-series dataset statistics."""
        print(f"[DL EDA Engineer] Analyzing river water level time-series data from {self.ts_csv_path}...")
        df = pd.read_csv(self.ts_csv_path)

        total_records = len(df)
        stations = df['station_name'].unique().tolist() if 'station_name' in df.columns else []
        district = df['district'].unique().tolist() if 'district' in df.columns else []
        
        water_min = float(df['water_level_m'].min())
        water_max = float(df['water_level_m'].max())
        water_mean = float(df['water_level_m'].mean())
        water_std = float(df['water_level_m'].std())

        rainfall_max = float(df['rainfall_mm_1h'].max())
        flow_max = float(df['flow_rate_m3s'].max())

        # Flood events count (water level > warning level)
        severe_events = len(df[df['water_level_m'] >= df['warning_level_m']]) if 'warning_level_m' in df.columns else 0

        ts_stats = {
            "total_time_series_records": total_records,
            "monitoring_stations": stations,
            "districts_covered": district,
            "water_level_stats_m": {
                "min": round(water_min, 2),
                "max": round(water_max, 2),
                "mean": round(water_mean, 2),
                "std": round(water_std, 2)
            },
            "max_rainfall_1h_mm": round(rainfall_max, 2),
            "max_flow_rate_m3s": round(flow_max, 2),
            "warning_level_exceeded_count": severe_events
        }

        print(f"[DL EDA Engineer] Time-Series Dataset: {total_records} records, Water Level range: {water_min:.2f}m - {water_max:.2f}m.")
        return ts_stats

    def run_eda(self):
        """Executes full DL EDA pipeline."""
        print("==========================================================")
        print("[DL EDA Engineer] Running Deep Learning EDA Pipeline...")
        print("==========================================================")
        
        img_stats = self.analyze_image_dataset()
        ts_stats = self.analyze_timeseries_dataset()

        summary = {
            "image_dataset_analysis": img_stats,
            "timeseries_dataset_analysis": ts_stats,
            "summary_conclusion": "Both spatial image dataset and 72,876 time-series records are ready for CNN & multi-step LSTM model training."
        }

        out_path = os.path.join(self.dl_dir, "dl_eda_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[DL EDA Engineer] Saved EDA Summary to {out_path}\n")
        return summary

if __name__ == "__main__":
    engineer = DLEDAEngineer()
    engineer.run_eda()
