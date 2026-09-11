"""
================================================================================
MASTER PIPELINE ORCHESTRATOR: STAGE 02 DEEP LEARNING (CNN + LSTM)
Disaster Response Coordination System
================================================================================
Sequential Execution of 5 Roles:
1. Data Engineer: Prepares 240 aerial drone images & 72,742 windowed LSTM sequences.
2. EDA Engineer: Computes spatial & temporal stats (exports dl_eda_summary.json).
3. DL Engineer: Trains PyTorch CNN (drone eyes) & PyTorch Multi-Horizon LSTM (water forecaster).
4. Evaluation Engineer: Evaluates CNN accuracy & LSTM MAE/RMSE (+1h to +10h).
5. Integration Engineer: Launches Flask Command Center Web Dashboard.
================================================================================
"""

import sys
import os

# Ensure base directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import importlib

DLDataEngineer = importlib.import_module("DL.01_data_engineer").DLDataEngineer
DLEDAEngineer = importlib.import_module("DL.02_eda_engineer").DLEDAEngineer
DLEngineer = importlib.import_module("DL.03_dl_engineer").DLEngineer
DLEvaluationEngineer = importlib.import_module("DL.04_evaluation_engineer").DLEvaluationEngineer
DLIntegrationEngineer = importlib.import_module("DL.05_integration_engineer").DLIntegrationEngineer

def run_master_pipeline(launch_server=True, port=5000):
    print("\n" + "="*80)
    print(" DISASTER RESPONSE COORDINATION - DEEP LEARNING MASTER PIPELINE ")
    print(" Stage 02: CNN Drone Image Classifier + Multi-Horizon LSTM Water Level Forecaster ")
    print("="*80 + "\n")

    # Step 1: Data Engineering
    print("[1/5] Executing DL Data Engineering...")
    data_engineer = DLDataEngineer()
    data_engineer.run_pipeline()

    # Step 2: Exploratory Data Analysis
    print("\n[2/5] Executing DL EDA Engineering...")
    eda_engineer = DLEDAEngineer()
    eda_engineer.run_eda()

    # Step 3: Deep Learning Model Training
    print("\n[3/5] Executing Deep Learning Model Training (PyTorch CNN & LSTM)...")
    dl_engineer = DLEngineer()
    dl_engineer.run_training()

    # Step 4: Model Evaluation
    print("\n[4/5] Executing DL Model Evaluation...")
    eval_engineer = DLEvaluationEngineer()
    eval_engineer.run_evaluation()

    # Step 5: Web Integration & Command Dashboard
    print("\n[5/5] Launching Integration Web Application...")
    integration_engineer = DLIntegrationEngineer()
    
    if launch_server:
        integration_engineer.run_server(host="127.0.0.1", port=port, debug=False)
    else:
        print("[Master Pipeline] All 4 DL pipeline stages completed cleanly!")

if __name__ == "__main__":
    launch = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-server":
        launch = False
    run_master_pipeline(launch_server=launch)
