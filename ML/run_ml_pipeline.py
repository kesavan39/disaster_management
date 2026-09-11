"""
================================================================================
MASTER PIPELINE ORCHESTRATOR
Disaster Response Coordination System - Stage 01 Machine Learning
================================================================================
Executes all 5 Squad Roles sequentially:
1. Data Engineer: Data cleaning, 72h rolling window & feature engineering.
2. EDA Engineer: Leading indicator analysis & false alarm diagnostics.
3. ML Engineer: Imbalanced learning, model training & feature leaderboard.
4. Evaluation Engineer: Stress-testing, overconfidence checks & briefing sheet.
5. Integration Engineer: Prepares API & command dashboard environment.
================================================================================
"""

import sys
import os
import time
import importlib

# Ensure UTF-8 output encoding for Windows command prompt compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure ML directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def run_pipeline():
    print("=" * 80)
    print("      DISASTER RESPONSE COORDINATION SYSTEM - STAGE 01 (ML PIPELINE)      ")
    print("=" * 80)
    start_time = time.time()

    # Step 1: Data Engineer
    print("\n>>> STAGE 1/5: RUNNING DATA ENGINEER MODULE...")
    mod1 = importlib.import_module('01_data_engineer')
    data_eng = mod1.DataEngineer()
    df_clean = data_eng.process_data()
    print("[OK] Data Engineer Stage Completed Successfully.")

    # Step 2: EDA Engineer
    print("\n>>> STAGE 2/5: RUNNING EDA ENGINEER MODULE...")
    mod2 = importlib.import_module('02_eda_engineer')
    eda_eng = mod2.EDAEngineer()
    eda_summary = eda_eng.run_analysis()
    print("[OK] EDA Engineer Stage Completed Successfully.")

    # Step 3: ML Engineer
    print("\n>>> STAGE 3/5: RUNNING ML ENGINEER MODULE...")
    mod3 = importlib.import_module('03_ml_engineer')
    ml_eng = mod3.MLEngineer()
    classifier, regressor, leaderboard = ml_eng.train_models()
    print("[OK] ML Engineer Stage Completed Successfully.")

    # Step 4: Evaluation Engineer
    print("\n>>> STAGE 4/5: RUNNING EVALUATION ENGINEER MODULE...")
    mod4 = importlib.import_module('04_evaluation_engineer')
    eval_eng = mod4.EvaluationEngineer()
    eval_report = eval_eng.evaluate()
    print("[OK] Evaluation Engineer Stage Completed Successfully.")

    # Step 5: Integration Engineer Verification
    print("\n>>> STAGE 5/5: VERIFYING INTEGRATION ENGINEER MODULE...")
    mod5 = importlib.import_module('05_integration_engineer')
    integration_eng = mod5.IntegrationEngineer()
    print("[OK] Integration Engineer Module Ready for Deployment.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"SUCCESS: FULL STAGE 01 ML PIPELINE EXECUTED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 80)
    print("All artifacts generated:")
    print(" - Processed Datasets: master_dataset.csv, train_data.csv, test_data.csv")
    print(" - EDA Diagnostics: eda_summary.json")
    print(" - Trained Models: saved_models/risk_classifier.pkl, saved_models/risk_regressor.pkl")
    print(" - Feature Leaderboard: feature_leaderboard.json")
    print(" - Evaluation & Stress Test: evaluation_report.json")
    print(" - Responder Guide: field_briefing_template.json")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
