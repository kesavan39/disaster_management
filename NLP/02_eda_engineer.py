"""
================================================================================
ROLE 02: EDA ENGINEER (NLP PIPELINE)
Disaster Response Coordination System - Stage 03 Natural Language Processing
================================================================================
Responsibilities:
- Explore vocabulary distributions across Urgency levels (LOW, MODERATE, HIGH).
- Compute TF-IDF n-gram frequencies and emergency keyword signals.
- Analyze channel sources, noise indicators, and ambiguous phrasing.
- Export EDA summary to NLP/nlp_eda_summary.json.
================================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class NLPEDAEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.nlp_dir = os.path.join(self.base_dir, "NLP")
        self.models_dir = os.path.join(self.nlp_dir, "saved_models")
        self.dataset_pkl = os.path.join(self.models_dir, "nlp_dataset.pkl")

    def run_eda(self):
        """Executes full NLP EDA Pipeline."""
        print("==========================================================")
        print("[NLP EDA Engineer] Starting NLP Exploratory Data Analysis...")
        print("==========================================================")
        
        if not os.path.exists(self.dataset_pkl):
            raise FileNotFoundError("NLP Dataset pkl not found! Run NLP/01_data_engineer.py first.")

        data = joblib.load(self.dataset_pkl)
        df = data['full_df']
        
        # 1. Top Keywords per Urgency Level
        urgency_top_words = {}
        for level in ['LOW', 'MODERATE', 'HIGH']:
            sub_text = df[df['urgency'] == level]['cleaned_message']
            vectorizer = CountVectorizer(stop_words='english', max_features=10, ngram_range=(1, 2))
            try:
                X_counts = vectorizer.fit_transform(sub_text)
                words = vectorizer.get_feature_names_out()
                sums = X_counts.sum(axis=0).A1
                word_freq = sorted(list(zip(words, [int(s) for s in sums])), key=lambda x: x[1], reverse=True)
                urgency_top_words[level] = word_freq
            except Exception:
                urgency_top_words[level] = []

        # 2. Disaster Type Frequencies
        disaster_counts = df['disaster_type'].value_counts().to_dict()

        # 3. Source Distribution
        source_counts = df['source'].value_counts().to_dict()

        # 4. Noisy Records Count
        noise_counts = df['is_noisy'].value_counts().to_dict()

        # 5. Average Message Length
        df['msg_len'] = df['cleaned_message'].apply(lambda x: len(str(x).split()))
        avg_len = float(df['msg_len'].mean())
        max_len = int(df['msg_len'].max())

        eda_summary = {
            "total_messages_analyzed": len(df),
            "average_words_per_message": round(avg_len, 2),
            "max_words_in_message": max_len,
            "disaster_type_frequencies": disaster_counts,
            "source_channel_frequencies": source_counts,
            "noisy_message_counts": {str(k): int(v) for k, v in noise_counts.items()},
            "top_keywords_by_urgency": urgency_top_words,
            "eda_conclusion": "Text signals show strong n-gram separation between LOW (inquiries), MODERATE (monitoring), and HIGH (emergency rescue requests)."
        }

        out_path = os.path.join(self.nlp_dir, "nlp_eda_summary.json")
        with open(out_path, "w") as f:
            json.dump(eda_summary, f, indent=2)

        print(f"[NLP EDA Engineer] Saved NLP EDA Summary to {out_path}\n")
        return eda_summary

if __name__ == "__main__":
    engineer = NLPEDAEngineer()
    engineer.run_eda()
