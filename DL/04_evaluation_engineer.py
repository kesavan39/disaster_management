"""
================================================================================
ROLE 04: EVALUATION ENGINEER (DEEP LEARNING PIPELINE)
Disaster Response Coordination System - Stage 02 Deep Learning
================================================================================
Responsibilities:
- Evaluate CNN Classifier on test drone images (Accuracy, Confusion Matrix, F1-score).
- Evaluate Multi-Horizon LSTM Forecaster on test time-series across horizons +1h to +10h (MAE, RMSE, R² in meters).
- Save comprehensive metrics evaluation report to DL/dl_evaluation_report.json.
================================================================================
"""

import os
import json
import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import importlib
dl_03 = importlib.import_module("DL.03_dl_engineer")
DroneImageCNN = dl_03.DroneImageCNN
WaterLevelLSTM = dl_03.WaterLevelLSTM
AerialImageDataset = dl_03.AerialImageDataset

class DLEvaluationEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "Data")
        self.dl_dir = os.path.join(self.base_dir, "DL")
        self.models_dir = os.path.join(self.dl_dir, "saved_models")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate_cnn(self):
        """Evaluates CNN model performance on test aerial images."""
        print("[DL Evaluation Engineer] Evaluating Drone Image CNN Classifier...")
        
        flooded_dir = os.path.join(self.data_dir, "images", "flooded")
        clear_dir = os.path.join(self.data_dir, "images", "clear")
        
        dataset = AerialImageDataset(flooded_dir, clear_dir)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        cnn_path = os.path.join(self.models_dir, "cnn_model.pth")
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"CNN model file {cnn_path} not found.")

        model = DroneImageCNN(num_classes=2).to(self.device)
        model.load_state_dict(torch.load(cnn_path, map_location=self.device))
        model.eval()

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for X_b, y_b in loader:
                X_b = X_b.to(self.device)
                outputs = model(X_b)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(y_b.numpy())
                all_probs.extend(probs)

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        cm = confusion_matrix(all_targets, all_preds).tolist()

        cnn_report = {
            "test_accuracy": round(float(acc) * 100, 2),
            "test_weighted_f1": round(float(f1), 4),
            "confusion_matrix": cm,
            "classes": ["Clear", "Flooded"]
        }

        print(f"[DL Evaluation Engineer] CNN Classifier -> Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
        return cnn_report

    def evaluate_lstm(self):
        """Evaluates Multi-Horizon LSTM Forecaster on time-series test set."""
        print("[DL Evaluation Engineer] Evaluating Multi-Horizon LSTM Forecaster (+1h to +10h)...")
        
        seq_path = os.path.join(self.dl_dir, "lstm_sequences.npz")
        prep_path = os.path.join(self.models_dir, "lstm_preprocessor.pkl")
        lstm_path = os.path.join(self.models_dir, "lstm_model.pth")

        if not os.path.exists(seq_path) or not os.path.exists(prep_path) or not os.path.exists(lstm_path):
            raise FileNotFoundError("LSTM data or model files missing.")

        data = np.load(seq_path)
        X_test, y_test = data['X_test'], data['y_test']
        prep = joblib.load(prep_path)

        target_mean = prep['target_mean']
        target_scale = prep['target_scale']

        model = WaterLevelLSTM(input_size=3, hidden_size=64, num_layers=2, forecast_horizon=10).to(self.device)
        model.load_state_dict(torch.load(lstm_path, map_location=self.device))
        model.eval()

        # Sample test set if large
        if len(X_test) > 3000:
            idx = np.random.choice(len(X_test), 3000, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds_scaled = model(X_tensor).cpu().numpy()

        # Unscale back to meters (m)
        y_test_m = y_test * target_scale + target_mean
        preds_m = preds_scaled * target_scale + target_mean

        horizon_metrics = {}
        overall_mae = float(mean_absolute_error(y_test_m, preds_m))
        overall_rmse = float(np.sqrt(mean_squared_error(y_test_m, preds_m)))
        overall_r2 = float(r2_score(y_test_m, preds_m))

        for h in range(10):
            h_mae = float(mean_absolute_error(y_test_m[:, h], preds_m[:, h]))
            h_rmse = float(np.sqrt(mean_squared_error(y_test_m[:, h], preds_m[:, h])))
            h_r2 = float(r2_score(y_test_m[:, h], preds_m[:, h]))
            horizon_metrics[f"+{h+1}h"] = {
                "mae_m": round(h_mae, 3),
                "rmse_m": round(h_rmse, 3),
                "r2_score": round(h_r2, 4)
            }

        lstm_report = {
            "overall_mae_m": round(overall_mae, 3),
            "overall_rmse_m": round(overall_rmse, 3),
            "overall_r2_score": round(overall_r2, 4),
            "horizon_breakdown": horizon_metrics
        }

        print(f"[DL Evaluation Engineer] LSTM Forecaster -> Overall MAE: {overall_mae:.3f}m | RMSE: {overall_rmse:.3f}m | R2: {overall_r2:.4f}")
        return lstm_report

    def run_evaluation(self):
        """Runs evaluation for both CNN and LSTM models."""
        print("==========================================================")
        print("[DL Evaluation Engineer] Starting Deep Learning Evaluation...")
        print("==========================================================")
        
        cnn_rep = self.evaluate_cnn()
        lstm_rep = self.evaluate_lstm()

        summary = {
            "cnn_classifier_evaluation": cnn_rep,
            "lstm_forecaster_evaluation": lstm_rep
        }

        out_path = os.path.join(self.dl_dir, "dl_evaluation_report.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[DL Evaluation Engineer] Saved DL Evaluation Report to {out_path}\n")
        return summary

if __name__ == "__main__":
    engineer = DLEvaluationEngineer()
    engineer.run_evaluation()
