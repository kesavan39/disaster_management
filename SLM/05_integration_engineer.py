"""
================================================================================
ROLE 05: INTEGRATION ENGINEER (SLM PIPELINE)
Disaster Response Coordination System - Stage 04 Small Language Model (SLM)
================================================================================
Responsibilities:
- Expose REST API endpoints for SLM report summarization.
- Serve sample disaster reports for 1-click dashboard testing.
- Expose ROUGE evaluation metrics & EDA summaries.
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
slm_03 = importlib.import_module("SLM.03_slm_engineer")
SLMSummarizerEngine = slm_03.SLMSummarizerEngine

class SLMIntegrationEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.slm_dir = os.path.join(self.base_dir, "SLM")
        self.models_dir = os.path.join(self.slm_dir, "saved_models")
        
        self.engine = SLMSummarizerEngine(base_dir=self.base_dir)
        self.app = Flask(__name__)
        self.register_routes()

    def register_routes(self):
        app = self.app

        @app.route('/api/slm/summarize', methods=['POST'])
        def summarize():
            data = request.get_json() or {}
            report_text = data.get('report', '').strip()
            
            try:
                res = self.engine.summarize_report(report_text)
                return jsonify(res)
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500

        @app.route('/api/slm/sample_reports', methods=['GET'])
        def get_sample_reports():
            samples = [
                {
                    "id": "flood_zone1",
                    "title": "🌊 Severe Flood Overtopping - Zone 1",
                    "disaster_type": "Flood",
                    "urgency": "Severe",
                    "report_text": "Emergency coordination report for Zone 1. A flood incident was reported after heavy rainfall caused river overflow. Initial field teams recorded approximately 349 people affected or requiring assistance. The current urgency level is assessed as severe. Responders requested water pumps and emergency shelters. The immediate response priority is to evacuate residents to higher ground. Teams are checking access routes, the safety of nearby residents, and the availability of emergency support. Updates are being collected from mobile app logs and field personnel."
                },
                {
                    "id": "fire_sector4",
                    "title": "🔥 Structural Fire Crisis - Sector 4",
                    "disaster_type": "Fire",
                    "urgency": "Severe",
                    "report_text": "Emergency coordination report for Sector 4. A structural fire incident was reported inside an industrial complex. Initial field teams recorded approximately 45 people trapped or affected by smoke inhalation. The current urgency level is assessed as severe. Responders requested firefighting units and high-capacity ambulances. The immediate response priority is to establish a safety perimeter and deploy aerial ladders to evacuate upper floors. Teams are clearing access roads and maintaining emergency communication."
                },
                {
                    "id": "cyclone_coastal",
                    "title": "⚡ Cyclone Infrastructure Damage - Coastal Zone",
                    "disaster_type": "Cyclone",
                    "urgency": "Severe",
                    "report_text": "Emergency coordination report for Coastal Zone 1. A cyclone incident was reported after category 3 storm winds downed high-voltage power lines and damaged roof structures. Initial field teams recorded approximately 121 people affected. The current urgency level is assessed as severe. Responders requested power restoration crews and structural reinforcement teams. The immediate response priority is to secure vulnerable structures and clear debris."
                },
                {
                    "id": "landslide_mountain",
                    "title": "⛰️ Mountain Pass Landslide - Zone 2",
                    "disaster_type": "Landslide",
                    "urgency": "Moderate",
                    "report_text": "Emergency coordination report for Zone 2. A landslide incident was reported after soil movement blocked the main highway pass. Initial field teams recorded approximately 163 people stranded in vehicles. The current urgency level is assessed as moderate. Responders requested heavy excavation machinery and rescue teams. The immediate response priority is to close the affected road and establish detour routes."
                }
            ]
            return jsonify({"status": "success", "samples": samples})

        @app.route('/api/slm/stats', methods=['GET'])
        def get_slm_stats():
            eda_path = os.path.join(self.slm_dir, "slm_eda_summary.json")
            eval_path = os.path.join(self.slm_dir, "slm_evaluation_report.json")
            
            res = {}
            if os.path.exists(eda_path):
                with open(eda_path, "r") as f:
                    res["eda"] = json.load(f)
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    res["evaluation"] = json.load(f)
            return jsonify(res)

    def run_server(self, host="127.0.0.1", port=5002, debug=False):
        print(f"[SLM Integration Engineer] Launching SLM Server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    engineer = SLMIntegrationEngineer()
    engineer.run_server(port=5002)
