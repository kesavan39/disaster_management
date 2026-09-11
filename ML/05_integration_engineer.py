"""
================================================================================
ROLE 05: INTEGRATION ENGINEER
Disaster Response Coordination System - Stage 01 Machine Learning
================================================================================
Responsibilities:
- Wrap trained ML classifier & regressor models into an ultra-fast REST API (<10ms).
- Serve real-time district triage endpoints, feature leaderboards, and stress-tests.
- Host the modern web UI dashboard.
================================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

class IntegrationEngineer:
    def __init__(self, output_dir=None):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = output_dir or os.path.join(self.base_dir, "ML")
        self.models_dir = os.path.join(self.output_dir, "saved_models")

        # Initialize Flask App
        template_folder = os.path.join(self.output_dir, "templates")
        static_folder = os.path.join(self.output_dir, "static")
        
        os.makedirs(template_folder, exist_ok=True)
        os.makedirs(static_folder, exist_ok=True)

        self.app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
        self.load_artifacts()
        self.register_routes()

    def load_artifacts(self):
        """Loads serialized models, preprocessors, and JSON metadata."""
        clf_path = os.path.join(self.models_dir, "risk_classifier.pkl")
        reg_path = os.path.join(self.models_dir, "risk_regressor.pkl")
        prep_path = os.path.join(self.models_dir, "preprocessor.pkl")

        if os.path.exists(clf_path) and os.path.exists(prep_path):
            self.classifier = joblib.load(clf_path)
            self.preprocessor = joblib.load(prep_path)
            self.regressor = joblib.load(reg_path) if os.path.exists(reg_path) else None
            self.scaler = self.preprocessor['scaler']
            self.label_encoder = self.preprocessor['label_encoder']
            self.feature_cols = self.preprocessor['feature_cols']
            print("[Integration Engineer] Successfully loaded ML models & preprocessor.")
        else:
            self.classifier = None
            self.preprocessor = None
            self.regressor = None
            print("[Integration Engineer] Warning: ML models not found yet. Run pipeline to train models.")

        # Load master dataset for district baseline predictions
        master_path = os.path.join(self.output_dir, "master_dataset.csv")
        if os.path.exists(master_path):
            self.master_df = pd.read_csv(master_path)
        else:
            self.master_df = None

    def register_routes(self):
        """Registers API & Web Dashboard routes."""
        app = self.app

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/api/predict', methods=['POST'])
        def predict():
            if not self.classifier or not self.preprocessor:
                return jsonify({"error": "Models not trained yet. Run pipeline first."}), 500

            data = request.get_json() or {}
            
            # Extract raw inputs with sensible defaults
            water_level_m = float(data.get('water_level_m', 2.0))
            rainfall_mm_6h = float(data.get('rainfall_mm_6h', 25.0))
            water_level_change_6h_m = float(data.get('water_level_change_6h_m', 0.2))
            emergency_calls_6h = float(data.get('emergency_calls_6h', 5))
            road_closures = float(data.get('road_closures', 0))
            estimated_exposed_population = float(data.get('estimated_exposed_population', 300))
            rainfall_mm_24h = float(data.get('rainfall_mm_24h', 60.0))
            ambulances_available = float(data.get('ambulances_available', 5))
            rescue_teams_available = float(data.get('rescue_teams_available', 2))
            shelter_beds_available = float(data.get('shelter_beds_available', 150))
            power_outage_probability = float(data.get('power_outage_probability', 0.1))

            # Dynamically engineer missing rolling features
            water_level_72h_avg = float(data.get('water_level_72h_avg', water_level_m * 0.9))
            water_level_72h_max = float(data.get('water_level_72h_max', max(water_level_m, water_level_m + 0.3)))
            water_level_72h_std = float(data.get('water_level_72h_std', 0.1))
            rainfall_intensity_ratio = rainfall_mm_6h / (rainfall_mm_24h + 1e-5)
            call_density_per_10k = (emergency_calls_6h / (estimated_exposed_population + 1.0)) * 10000.0
            call_surge_6h = float(data.get('call_surge_6h', emergency_calls_6h * 0.4))
            infrastructure_stress = road_closures * (1.0 + power_outage_probability)
            resource_pressure = estimated_exposed_population / (shelter_beds_available + ambulances_available * 15 + rescue_teams_available * 50 + 1.0)

            input_dict = {
                'water_level_m': water_level_m,
                'rainfall_mm_6h': rainfall_mm_6h,
                'water_level_change_6h_m': water_level_change_6h_m,
                'emergency_calls_6h': emergency_calls_6h,
                'road_closures': road_closures,
                'estimated_exposed_population': estimated_exposed_population,
                'rainfall_mm_24h': rainfall_mm_24h,
                'ambulances_available': ambulances_available,
                'rescue_teams_available': rescue_teams_available,
                'shelter_beds_available': shelter_beds_available,
                'power_outage_probability': power_outage_probability,
                'water_level_72h_avg': water_level_72h_avg,
                'water_level_72h_max': water_level_72h_max,
                'water_level_72h_std': water_level_72h_std,
                'rainfall_intensity_ratio': rainfall_intensity_ratio,
                'call_density_per_10k': call_density_per_10k,
                'call_surge_6h': call_surge_6h,
                'infrastructure_stress': infrastructure_stress,
                'resource_pressure': resource_pressure
            }

            input_df = pd.DataFrame([input_dict])[self.feature_cols]
            X_scaled = self.scaler.transform(input_df)

            pred_enc = self.classifier.predict(X_scaled)[0]
            probs = self.classifier.predict_proba(X_scaled)[0]
            risk_level = self.label_encoder.inverse_transform([pred_enc])[0]

            prob_dict = {cname: round(float(probs[idx]), 4) for idx, cname in enumerate(self.label_encoder.classes_)}
            confidence = prob_dict[risk_level]

            # Risk Score Regression
            if self.regressor:
                risk_score = float(self.regressor.predict(X_scaled)[0])
                risk_score = max(0.0, min(100.0, round(risk_score, 1)))
            else:
                # Heuristic mapping if regressor absent
                base_score = 20.0 if risk_level == 'Low' else (60.0 if risk_level == 'Moderate' else 90.0)
                risk_score = base_score

            # Tactical recommendation
            if risk_level == 'Severe':
                action = "RED ALERT: Immediate evacuation dispatch! Deploy high-clearance rescue vehicles & boats."
            elif risk_level == 'Moderate':
                action = "AMBER WARNING: Pre-stage rescue units in low-lying zones. Issue flash flood alert."
            else:
                action = "GREEN CLEAR: Normal patrol & routine monitoring. Maintain standard standby."

            return jsonify({
                "status": "success",
                "district": data.get('district', 'Custom Location'),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "confidence_score": round(confidence * 100, 1),
                "probabilities": prob_dict,
                "tactical_action": action,
                "key_metrics": {
                    "water_level_m": water_level_m,
                    "rainfall_24h_mm": rainfall_mm_24h,
                    "emergency_calls_6h": int(emergency_calls_6h),
                    "road_closures": int(road_closures),
                    "power_outage_prob": round(power_outage_probability * 100, 1)
                }
            })

        @app.route('/api/districts', methods=['GET'])
        def get_districts():
            if self.master_df is None:
                return jsonify({"error": "Master dataset not found"}), 404

            # Get latest entry per district
            latest_df = self.master_df.sort_values(by='timestamp').groupby('district').last().reset_index()
            
            districts = []
            for _, row in latest_df.iterrows():
                districts.append({
                    "district": str(row['district']),
                    "station_name": str(row.get('station_name', 'Central Gauge')),
                    "river": str(row.get('river', 'Main River')),
                    "water_level_m": round(float(row['water_level_m']), 2),
                    "rainfall_mm_24h": round(float(row['rainfall_mm_24h']), 1),
                    "emergency_calls_6h": int(row['emergency_calls_6h']),
                    "road_closures": int(row['road_closures']),
                    "risk_score": round(float(row['risk_score']), 1),
                    "risk_level": str(row['risk_level']),
                    "ambulances_available": int(row['ambulances_available']),
                    "rescue_teams_available": int(row['rescue_teams_available']),
                    "shelter_beds_available": int(row['shelter_beds_available'])
                })
            
            return jsonify({"total_districts": len(districts), "districts": districts})

        @app.route('/api/leaderboard', methods=['GET'])
        def get_leaderboard():
            lead_path = os.path.join(self.output_dir, "feature_leaderboard.json")
            if os.path.exists(lead_path):
                with open(lead_path, "r") as f:
                    return jsonify(json.load(f))
            return jsonify({"error": "Leaderboard not generated yet"}), 404

        @app.route('/api/eda', methods=['GET'])
        def get_eda():
            eda_path = os.path.join(self.output_dir, "eda_summary.json")
            if os.path.exists(eda_path):
                with open(eda_path, "r") as f:
                    return jsonify(json.load(f))
            return jsonify({"error": "EDA summary not generated yet"}), 404

        @app.route('/api/evaluation', methods=['GET'])
        def get_evaluation():
            eval_path = os.path.join(self.output_dir, "evaluation_report.json")
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    return jsonify(json.load(f))
            return jsonify({"error": "Evaluation report not generated yet"}), 404

        @app.route('/api/field_briefing', methods=['GET'])
        def get_field_briefing():
            brief_path = os.path.join(self.output_dir, "field_briefing_template.json")
            if os.path.exists(brief_path):
                with open(brief_path, "r") as f:
                    return jsonify(json.load(f))
            return jsonify({"error": "Field briefing sheet not generated yet"}), 404

    def run_server(self, host="127.0.0.1", port=5000, debug=False):
        """Starts Flask Web Server."""
        print(f"[Integration Engineer] Launching Disaster Response Command Center Server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    engineer = IntegrationEngineer()
    engineer.run_server(port=5000)
