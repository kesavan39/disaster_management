"""
================================================================================
ROLE 03: NLP ENGINEER (MODEL TRAINING & ENTITY EXTRACTION)
Disaster Response Coordination System - Stage 03 Natural Language Processing
================================================================================
Responsibilities:
- Train TF-IDF + Calibrated Classifiers for Urgency & Disaster Type prediction.
- Build NER Extractor for Location, Headcount (People Affected), & Resource Required.
- Build TF-IDF Cosine Similarity engine against 20,000 historical emergency logs.
- Serialize trained models and preprocessors to NLP/saved_models/.
================================================================================
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

class NLPEngineer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.nlp_dir = os.path.join(self.base_dir, "NLP")
        self.models_dir = os.path.join(self.nlp_dir, "saved_models")
        self.dataset_pkl = os.path.join(self.models_dir, "nlp_dataset.pkl")
        os.makedirs(self.models_dir, exist_ok=True)

        self.tfidf_vectorizer = None
        self.urgency_model = None
        self.disaster_model = None
        self.urgency_encoder = None
        self.disaster_encoder = None
        
        # Entity Reference Sets
        self.known_locations = []
        self.known_resources = []

    def extract_entities(self, text, message_record=None):
        """
        Named Entity Recognition (NER) & Information Extraction Engine:
        Extracts Location, Headcount (People Affected), and Resource Required.
        """
        text_lower = text.lower()

        # 1. Extract Location
        found_location = "Unknown Location"
        for loc in self.known_locations:
            if loc.lower() in text_lower:
                found_location = loc
                break
        
        if found_location == "Unknown Location":
            # Pattern matching for Zone / Area / Camp
            match_loc = re.search(r'\b(zone\s*\d+|harbor\s*area|east\s*village|north\s*camp|market\s*road|shelter\s*area|hill\s*view)\b', text_lower, re.IGNORECASE)
            if match_loc:
                found_location = match_loc.group(0).title()

        # 2. Extract People Affected (Headcount)
        people_count = 0
        match_people = re.search(r'(\d+)\s*(people|persons|trapped|casualties|injured|residents|victims)?', text_lower, re.IGNORECASE)
        if match_people:
            try:
                num = int(match_people.group(1))
                if 0 < num < 5000:
                    people_count = num
            except ValueError:
                people_count = 0

        # 3. Extract Resource Required
        found_resources = []
        for res in self.known_resources:
            if res.lower() in text_lower:
                found_resources.append(res)

        if not found_resources:
            # Common emergency resource pattern matching
            resource_patterns = [
                ('rescue boat', r'\b(boat|rescue boat|water craft)\b'),
                ('ambulance', r'\b(ambulance|medical van|paramedic)\b'),
                ('rescue team', r'\b(rescue team|search and rescue|help team)\b'),
                ('drinking water', r'\b(water|drinking water|clean water)\b'),
                ('excavator', r'\b(excavator|heavy machinery|crane)\b'),
                ('medical team', r'\b(doctor|medical team|first aid|nurse)\b'),
                ('food packets', r'\b(food|food packets|ration)\b'),
                ('firefighters', r'\b(fire engine|firefighters|fire truck)\b')
            ]
            for name, pat in resource_patterns:
                if re.search(pat, text_lower):
                    found_resources.append(name)

        resource_str = ", ".join(list(set(found_resources))) if found_resources else "General Relief Assistance"

        return {
            "location": found_location,
            "people_affected": people_count,
            "resource_required": resource_str
        }

    def generate_recommended_action(self, urgency, disaster_type, resource):
        """Generates AI Recommended Action protocol based on extracted entities."""
        urgency = str(urgency).upper()
        disaster_type = str(disaster_type).title()
        
        if urgency == "HIGH" or urgency == "SEVERE":
            if "boat" in resource.lower():
                return "⚠️ RED ALERT: Prioritize high-clearance rescue boat & flood evacuation team immediately!"
            elif "ambulance" in resource.lower() or "medical" in resource.lower():
                return "⚠️ RED ALERT: Dispatch emergency ambulance & trauma response team immediately!"
            elif "fire" in resource.lower() or disaster_type == "Fire":
                return "⚠️ RED ALERT: Contain fire surge & execute immediate neighborhood evacuation!"
            else:
                return "⚠️ RED ALERT: Prioritize search & rescue operation immediately!"
        elif urgency == "MODERATE":
            return f"⚡ AMBER WATCH: Pre-stage {resource} & monitor emergency situation in zone."
        else:
            return "🟢 GREEN CLEAR: Maintain standard standby & log routine monitoring query."

    def train_models(self):
        """Trains TF-IDF models and Classifiers on 20,000 records dataset."""
        print("[NLP Engineer] Loading dataset for model training...")
        if not os.path.exists(self.dataset_pkl):
            raise FileNotFoundError("Dataset pkl not found! Run NLP/01_data_engineer.py first.")

        data = joblib.load(self.dataset_pkl)
        train_df = data['train_df']
        full_df = data['full_df']

        # Populate Entity References
        self.known_locations = [loc for loc in full_df['location'].unique() if loc != 'Unknown Location']
        self.known_resources = [res for res in full_df['resource_required'].dropna().unique() if res != 'none']

        print(f"[NLP Engineer] Vectorizing {len(train_df)} training messages with TF-IDF...")
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), 
            max_features=5000, 
            sublinear_tf=True
        )
        
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(train_df['cleaned_message'])
        X_full_tfidf = self.tfidf_vectorizer.transform(full_df['cleaned_message'])

        # Encoders
        self.urgency_encoder = LabelEncoder()
        y_train_urgency = self.urgency_encoder.fit_transform(train_df['urgency'])

        self.disaster_encoder = LabelEncoder()
        y_train_disaster = self.disaster_encoder.fit_transform(train_df['disaster_type'])

        # 1. Train Calibrated Classifier for Urgency
        print("[NLP Engineer] Training Calibrated Classifier for Urgency Classification...")
        base_clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=1000, random_state=42)
        self.urgency_model = CalibratedClassifierCV(estimator=base_clf, cv=5)
        self.urgency_model.fit(X_train_tfidf, y_train_urgency)

        # 2. Train Classifier for Disaster Type
        print("[NLP Engineer] Training Calibrated Classifier for Disaster Type Classification...")
        base_disaster = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=1000, random_state=42)
        self.disaster_model = CalibratedClassifierCV(estimator=base_disaster, cv=5)
        self.disaster_model.fit(X_train_tfidf, y_train_disaster)

        # Save Pipeline Artifacts
        pipeline_artifacts = {
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "urgency_model": self.urgency_model,
            "disaster_model": self.disaster_model,
            "urgency_encoder": self.urgency_encoder,
            "disaster_encoder": self.disaster_encoder,
            "known_locations": self.known_locations,
            "known_resources": self.known_resources,
            "X_full_tfidf": X_full_tfidf,
            "full_messages": full_df['cleaned_message'].values,
            "full_df_records": full_df.to_dict(orient='records')
        }

        model_path = os.path.join(self.models_dir, "nlp_pipeline_model.pkl")
        joblib.dump(pipeline_artifacts, model_path)

        print(f"[NLP Engineer] Trained NLP Pipeline successfully saved to {model_path}\n")
        return pipeline_artifacts

    def predict_message(self, message_text):
        """Predicts Urgency, Disaster Type, Entities, Similarity Score, and Recommended Action."""
        if not self.tfidf_vectorizer or not self.urgency_model:
            model_path = os.path.join(self.models_dir, "nlp_pipeline_model.pkl")
            if os.path.exists(model_path):
                artifacts = joblib.load(model_path)
                self.tfidf_vectorizer = artifacts['tfidf_vectorizer']
                self.urgency_model = artifacts['urgency_model']
                self.disaster_model = artifacts['disaster_model']
                self.urgency_encoder = artifacts['urgency_encoder']
                self.disaster_encoder = artifacts['disaster_encoder']
                self.known_locations = artifacts['known_locations']
                self.known_resources = artifacts['known_resources']
                self.X_full_tfidf = artifacts['X_full_tfidf']
                self.full_messages = artifacts['full_messages']
            else:
                raise FileNotFoundError("NLP Pipeline model not found! Run training first.")

        text_clean = message_text.strip()
        X_query = self.tfidf_vectorizer.transform([text_clean])

        # Urgency Prediction
        urgency_probs = self.urgency_model.predict_proba(X_query)[0]
        urgency_idx = np.argmax(urgency_probs)
        predicted_urgency = self.urgency_encoder.inverse_transform([urgency_idx])[0]
        urgency_conf = float(urgency_probs[urgency_idx]) * 100.0

        # Disaster Type Prediction
        disaster_probs = self.disaster_model.predict_proba(X_query)[0]
        disaster_idx = np.argmax(disaster_probs)
        predicted_disaster = self.disaster_encoder.inverse_transform([disaster_idx])[0]
        disaster_conf = float(disaster_probs[disaster_idx]) * 100.0

        # Entity Extraction
        entities = self.extract_entities(text_clean)

        # Historical Similarity Matching
        sim_scores = cosine_similarity(X_query, self.X_full_tfidf)[0]
        top_idx = np.argmax(sim_scores)
        top_sim_score = float(sim_scores[top_idx]) * 100.0
        most_similar_message = str(self.full_messages[top_idx])

        # Recommended Action
        recommended_action = self.generate_recommended_action(
            predicted_urgency, predicted_disaster, entities['resource_required']
        )

        return {
            "status": "success",
            "input_message": message_text,
            "predicted_urgency": predicted_urgency,
            "urgency_confidence": round(urgency_conf, 1),
            "urgency_probabilities": {
                name: round(float(urgency_probs[i]) * 100, 1) 
                for i, name in enumerate(self.urgency_encoder.classes_)
            },
            "predicted_disaster_type": predicted_disaster,
            "disaster_confidence": round(disaster_conf, 1),
            "location": entities['location'],
            "people_affected": entities['people_affected'],
            "resource_required": entities['resource_required'],
            "historical_similarity_percentage": round(top_sim_score, 1),
            "most_similar_historical_message": most_similar_message,
            "recommended_action": recommended_action
        }

    def run_pipeline(self):
        """Runs NLP Engineer pipeline."""
        print("==========================================================")
        print("[NLP Engineer] Running NLP Model Training Pipeline...")
        print("==========================================================")
        return self.train_models()

if __name__ == "__main__":
    engineer = NLPEngineer()
    engineer.run_pipeline()
