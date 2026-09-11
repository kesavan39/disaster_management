"""
================================================================================
MASTER PIPELINE ORCHESTRATOR: STAGE 03 NATURAL LANGUAGE PROCESSING (NLP)
Disaster Response Coordination System
================================================================================
Sequential Execution of 5 Roles:
1. Data Engineer: Prepares 20,000 emergency text messages dataset.
2. EDA Engineer: Computes vocabulary n-grams & channel stats.
3. NLP Engineer: Trains TF-IDF Classifiers & NER Entity Extractor.
4. Evaluation Engineer: Validates precision, recall, F1, & noise robustness.
5. Integration Engineer: Launches Flask REST API server.
================================================================================
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import importlib

NLPDataEngineer = importlib.import_module("NLP.01_data_engineer").NLPDataEngineer
NLPEDAEngineer = importlib.import_module("NLP.02_eda_engineer").NLPEDAEngineer
NLPEngineer = importlib.import_module("NLP.03_nlp_engineer").NLPEngineer
NLPEvaluationEngineer = importlib.import_module("NLP.04_evaluation_engineer").NLPEvaluationEngineer
NLPIntegrationEngineer = importlib.import_module("NLP.05_integration_engineer").NLPIntegrationEngineer

def run_master_pipeline(launch_server=True, port=5001):
    print("\n" + "="*80)
    print(" DISASTER RESPONSE COORDINATION - DAY 3 NLP MASTER PIPELINE ")
    print(" 20,000 Record Emergency Text Classification + NER + Cosine Similarity Engine ")
    print("="*80 + "\n")

    # Step 1: Data Engineering
    print("[1/5] Executing NLP Data Engineering...")
    data_engineer = NLPDataEngineer()
    data_engineer.run_pipeline()

    # Step 2: Exploratory Data Analysis
    print("\n[2/5] Executing NLP EDA Engineering...")
    eda_engineer = NLPEDAEngineer()
    eda_engineer.run_eda()

    # Step 3: NLP Model & Entity Extraction Training
    print("\n[3/5] Executing NLP Model & NER Extractor Training...")
    nlp_engineer = NLPEngineer()
    nlp_engineer.run_pipeline()

    # Step 4: Model Evaluation
    print("\n[4/5] Executing NLP Model Evaluation...")
    eval_engineer = NLPEvaluationEngineer()
    eval_engineer.run_evaluation()

    # Step 5: Web Integration & API Server
    print("\n[5/5] Launching NLP Integration API Application...")
    integration_engineer = NLPIntegrationEngineer()
    
    if launch_server:
        integration_engineer.run_server(host="127.0.0.1", port=port, debug=False)
    else:
        print("[NLP Master Pipeline] All 4 NLP pipeline stages completed cleanly!")

if __name__ == "__main__":
    launch = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-server":
        launch = False
    run_master_pipeline(launch_server=launch)
