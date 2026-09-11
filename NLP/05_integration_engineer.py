"""
================================================================================
ROLE 05: INTEGRATION ENGINEER (NLP PIPELINE)
Disaster Response Coordination System - Stage 03 Natural Language Processing
================================================================================
Responsibilities:
- Expose REST API endpoints for live NLP emergency message predictions.
- Serve preset sample emergency messages for 1-click UI testing.
- Expose evaluation metrics and EDA statistics.
================================================================================
"""

import os
import json
import joblib
from flask import Flask, request, jsonify

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import importlib
nlp_03 = importlib.import_module("NLP.03_nlp_engineer")
NLPEngineer = nlp_03.NLPEngineer

class NLPIntegrationEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.nlp_dir = os.path.join(self.base_dir, "NLP")
        self.models_dir = os.path.join(self.nlp_dir, "saved_models")
        
        self.nlp_engine = NLPEngineer(base_dir=self.base_dir)
        self.app = Flask(__name__)
        self.register_routes()

    def register_routes(self):
        app = self.app

        @app.route('/api/nlp/predict_message', methods=['POST'])
        def predict_message():
            data = request.get_json() or {}
            message_text = data.get('message', '').strip()
            
            if not message_text:
                return jsonify({"status": "error", "error": "No emergency message text provided."}), 400

            try:
                res = self.nlp_engine.predict_message(message_text)
                return jsonify(res)
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500

        @app.route('/api/nlp/sample_messages', methods=['GET'])
        def get_sample_messages():
            samples = [
                {
                    "id": "high_flood_rescue",
                    "title": "🚨 High Urgency Flood Rescue",
                    "message": "8 people are trapped in Zone 2. Water is rising rapidly. Send a rescue boat immediately.",
                    "disaster_type": "Flood",
                    "expected_urgency": "HIGH"
                },
                {
                    "id": "high_power_outage",
                    "title": "⚡ Cyclone Power Line Hazard",
                    "message": "URGENT: power lines are down near Harbor Area HELP!",
                    "disaster_type": "Cyclone",
                    "expected_urgency": "HIGH"
                },
                {
                    "id": "mod_water_level",
                    "title": "⚡ Moderate Water Rise",
                    "message": "Water level is rising near Market Road. Need extra drinking water and food packets.",
                    "disaster_type": "Flood",
                    "expected_urgency": "MODERATE"
                },
                {
                    "id": "low_inquiry",
                    "title": "🟢 Low Urgency Inquiry",
                    "message": "Can someone confirm the boat schedule for tomorrow morning?",
                    "disaster_type": "Flood",
                    "expected_urgency": "LOW"
                },
                {
                    "id": "high_fire_evac",
                    "title": "🔥 Fire Evacuation Request",
                    "message": "Fire spreading rapidly in East Village! 15 people trapped near building B. Send firefighters and ambulance!",
                    "disaster_type": "Fire",
                    "expected_urgency": "HIGH"
                }
            ]
            return jsonify({"status": "success", "samples": samples})

        @app.route('/api/nlp/stats', methods=['GET'])
        def get_nlp_stats():
            eda_path = os.path.join(self.nlp_dir, "nlp_eda_summary.json")
            eval_path = os.path.join(self.nlp_dir, "nlp_evaluation_report.json")
            
            res = {}
            if os.path.exists(eda_path):
                with open(eda_path, "r") as f:
                    res["eda"] = json.load(f)
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    res["evaluation"] = json.load(f)
            return jsonify(res)

    def run_server(self, host="127.0.0.1", port=5001, debug=False):
        print(f"[NLP Integration Engineer] Launching NLP Server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    engineer = NLPIntegrationEngineer()
    engineer.run_server(port=5001)
