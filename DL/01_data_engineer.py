"""
================================================================================
ROLE 01: DATA ENGINEER (DEEP LEARNING PIPELINE)
Disaster Response Coordination System - Stage 02 Deep Learning
================================================================================
Responsibilities:
- Create image directory structure (Data/images/flooded, Data/images/clear, Data/sample_test_images).
- Synthesize/generate realistic aerial drone image datasets for Flooded vs Clear roads & rivers.
- Generate high-fidelity test images for live dashboard user selection & upload predictions.
- Load 72,876-record river gauge time-series dataset (Data/water_level_timeseries.csv).
- Construct 24-hour lookback sequences with 10-step multi-horizon future target vectors (+1h to +10h).
- Standardize features and export serialized PyTorch-ready datasets.
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from sklearn.preprocessing import StandardScaler

class DLDataEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "Data")
        self.dl_dir = os.path.join(self.base_dir, "DL")
        self.models_dir = os.path.join(self.dl_dir, "saved_models")
        
        self.images_flooded_dir = os.path.join(self.data_dir, "images", "flooded")
        self.images_clear_dir = os.path.join(self.data_dir, "images", "clear")
        self.sample_images_dir = os.path.join(self.data_dir, "sample_test_images")
        
        os.makedirs(self.images_flooded_dir, exist_ok=True)
        os.makedirs(self.images_clear_dir, exist_ok=True)
        os.makedirs(self.sample_images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.ts_csv_path = os.path.join(self.data_dir, "water_level_timeseries.csv")

    def generate_synthetic_drone_images(self, num_per_class=120, img_size=(128, 128)):
        """
        Preserves user's custom images if present, otherwise generates synthetic images.
        """
        clear_count = len([f for f in os.listdir(self.images_clear_dir) if f.endswith(('.jpg', '.png'))]) if os.path.exists(self.images_clear_dir) else 0
        flooded_count = len([f for f in os.listdir(self.images_flooded_dir) if f.endswith(('.jpg', '.png'))]) if os.path.exists(self.images_flooded_dir) else 0
        
        if clear_count > 0 and flooded_count > 0:
            print(f"[DL Data Engineer] Using existing dataset images: {clear_count} CLEAR images, {flooded_count} FLOODED images found.")
            return

        print(f"[DL Data Engineer] Generating {num_per_class * 2} aerial drone images in Data/images/...")
        
        np.random.seed(42)

        # Clear Images Generation
        for i in range(num_per_class):
            img = Image.new('RGB', img_size, color=(50, 50, 50)) # Base asphalt grey
            draw = ImageDraw.Draw(img)
            
            # Road geometry
            draw.rectangle([20, 0, 108, 128], fill=(60, 62, 65))
            # Yellow center line
            for y in range(0, 128, 16):
                draw.line([(64, y), (64, y + 8)], fill=(230, 180, 40), width=2)
            
            # Side vegetation (green/brown)
            veg_color = (np.random.randint(30, 70), np.random.randint(90, 140), np.random.randint(20, 50))
            draw.rectangle([0, 0, 20, 128], fill=veg_color)
            draw.rectangle([108, 0, 128, 128], fill=veg_color)
            
            # Convert to numpy for texture/noise injection
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 8.0, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            
            final_img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.5))
            final_img.save(os.path.join(self.images_clear_dir, f"clear_drone_{i+1:04d}.jpg"))

        # Flooded Images Generation
        for i in range(num_per_class):
            # Base water color (murky flood water: greenish-brown-blue)
            base_water = (np.random.randint(40, 70), np.random.randint(70, 110), np.random.randint(90, 130))
            img = Image.new('RGB', img_size, color=base_water)
            draw = ImageDraw.Draw(img)
            
            # Partially submerged road outline (barely visible)
            draw.rectangle([20, 0, 108, 128], fill=(50, 65, 75))
            
            # Water ripples and wave crests (light highlights)
            for _ in range(30):
                rx = np.random.randint(0, 120)
                ry = np.random.randint(0, 120)
                rw = np.random.randint(10, 35)
                rh = np.random.randint(2, 6)
                ripple_col = (np.random.randint(120, 170), np.random.randint(140, 190), np.random.randint(160, 210))
                draw.ellipse([rx, ry, rx+rw, ry+rh], outline=ripple_col, width=1)

            # Convert to numpy for water turbulence texture
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 12.0, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            
            final_img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.8))
            final_img.save(os.path.join(self.images_flooded_dir, f"flooded_drone_{i+1:04d}.jpg"))

        print(f"[DL Data Engineer] Created {num_per_class} CLEAR and {num_per_class} FLOODED drone images.")

    def create_sample_test_images(self):
        """Copies real dataset images into sample_test_images directory for web UI testing."""
        import shutil
        print("[DL Data Engineer] Copying real sample test images into Data/sample_test_images/...")
        
        clear_files = sorted([f for f in os.listdir(self.images_clear_dir) if f.endswith(('.jpg', '.png'))]) if os.path.exists(self.images_clear_dir) else []
        flooded_files = sorted([f for f in os.listdir(self.images_flooded_dir) if f.endswith(('.jpg', '.png'))]) if os.path.exists(self.images_flooded_dir) else []

        if clear_files:
            shutil.copy(os.path.join(self.images_clear_dir, clear_files[0]), os.path.join(self.sample_images_dir, "clear_zone1.jpg"))
            if len(clear_files) > 10:
                shutil.copy(os.path.join(self.images_clear_dir, clear_files[10]), os.path.join(self.sample_images_dir, "clear_sample2.jpg"))

        if flooded_files:
            shutil.copy(os.path.join(self.images_flooded_dir, flooded_files[0]), os.path.join(self.sample_images_dir, "flooded_zone3.jpg"))
            if len(flooded_files) > 10:
                shutil.copy(os.path.join(self.images_flooded_dir, flooded_files[10]), os.path.join(self.sample_images_dir, "severe_overflow.jpg"))

        print("[DL Data Engineer] Real sample test images copied successfully.")

    def prepare_lstm_timeseries_dataset(self, lookback=24, forecast_horizon=10):
        """
        Loads water_level_timeseries.csv and extracts 24-hour sequence windows to forecast
        the next 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 hours of water levels.
        """
        print(f"[DL Data Engineer] Loading time-series dataset from {self.ts_csv_path}...")
        df = pd.read_csv(self.ts_csv_path)
        
        # Sort by station and timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['station_id', 'timestamp']).reset_index(drop=True)
        
        feature_cols = ['water_level_m', 'rainfall_mm_1h', 'flow_rate_m3s']
        target_col = 'water_level_m'
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])
        
        # Target scale index
        target_idx = feature_cols.index(target_col)
        target_mean = scaler.mean_[target_idx]
        target_scale = scaler.scale_[target_idx]
        
        X_seq = []
        y_seq = []
        
        # Group by station to construct continuous sequence windows per station
        for station_id, group in df.groupby('station_id'):
            group_scaled = scaler.transform(group[feature_cols])
            n_records = len(group_scaled)
            
            for i in range(n_records - lookback - forecast_horizon + 1):
                # Input: past 24 hours of [water_level, rainfall, flow_rate]
                X_win = group_scaled[i : i + lookback]
                # Output: next 10 hours of target water level
                y_win = group_scaled[i + lookback : i + lookback + forecast_horizon, target_idx]
                
                X_seq.append(X_win)
                y_seq.append(y_win)

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        print(f"[DL Data Engineer] Created {len(X_seq)} sequence samples (Input shape: {X_seq.shape}, Target shape: {y_seq.shape})")

        # Train/Test Split (80/20 chronologically per station)
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # Save preprocessor and dataset metadata
        preprocessor = {
            'scaler': scaler,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'target_mean': float(target_mean),
            'target_scale': float(target_scale),
            'lookback': lookback,
            'forecast_horizon': forecast_horizon
        }
        
        joblib.dump(preprocessor, os.path.join(self.models_dir, "lstm_preprocessor.pkl"))
        
        # Save sequence arrays
        np.savez_compressed(
            os.path.join(self.dl_dir, "lstm_sequences.npz"),
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )

        print("[DL Data Engineer] Preprocessed LSTM sequences successfully saved.")
        return X_train, y_train, X_test, y_test, preprocessor

    def run_pipeline(self):
        """Runs complete Data Engineering pipeline."""
        print("==========================================================")
        print("[DL Data Engineer] Starting Deep Learning Data Pipeline...")
        print("==========================================================")
        
        self.generate_synthetic_drone_images(num_per_class=120)
        self.create_sample_test_images()
        X_train, y_train, X_test, y_test, prep = self.prepare_lstm_timeseries_dataset()
        
        print("[DL Data Engineer] Data Engineering completed successfully!\n")
        return prep

if __name__ == "__main__":
    engineer = DLDataEngineer()
    engineer.run_pipeline()
