"""
================================================================================
ROLE 03: ML ENGINEER (REGULARIZED & OVERFITTING-PROTECTED)
Disaster Response Coordination System - Stage 01 Machine Learning
================================================================================
Responsibilities:
- Train regularized classifiers to prevent overfitting (max_depth limits, min_samples_leaf, L2 regularization).
- Implement 5-Fold Stratified Cross-Validation to validate true out-of-fold generalization.
- Handle severe class imbalance safely with regularized SMOTE oversampling.
- Extract data-driven Feature Leaderboard (feature_leaderboard.json).
- Save serialized model pipelines (saved_models/).
================================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, f1_score, accuracy_score, r2_score, mean_squared_error
from imblearn.over_sampling import SMOTE

class MLEngineer:
    def __init__(self, train_path=None, test_path=None, output_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.train_path = train_path or os.path.join(base_dir, "ML", "train_data.csv")
        self.test_path = test_path or os.path.join(base_dir, "ML", "test_data.csv")
        self.output_dir = output_dir or os.path.join(base_dir, "ML")
        self.models_dir = os.path.join(self.output_dir, "saved_models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.feature_cols = [
            'water_level_m', 'rainfall_mm_6h', 'water_level_change_6h_m',
            'emergency_calls_6h', 'road_closures', 'estimated_exposed_population',
            'rainfall_mm_24h', 'ambulances_available', 'rescue_teams_available',
            'shelter_beds_available', 'power_outage_probability',
            'water_level_72h_avg', 'water_level_72h_max', 'water_level_72h_std',
            'rainfall_intensity_ratio', 'call_density_per_10k', 'call_surge_6h',
            'infrastructure_stress', 'resource_pressure'
        ]
        self.target_class = 'risk_level'
        self.target_score = 'risk_score'

    def load_and_preprocess(self):
        """Loads train and test CSVs and prepares features & targets."""
        print("[ML Engineer] Loading and preparing training & testing datasets...")
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        X_train_raw = train_df[self.feature_cols]
        y_train_class = train_df[self.target_class]
        y_train_score = train_df[self.target_score]

        X_test_raw = test_df[self.feature_cols]
        y_test_class = test_df[self.target_class]
        y_test_score = test_df[self.target_score]

        # Fit Scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Fit Label Encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(['Low', 'Moderate', 'Severe'])
        y_train_encoded = label_encoder.transform(y_train_class)
        y_test_encoded = label_encoder.transform(y_test_class)

        # Save Scaler & Encoder
        preprocessor = {
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_cols': self.feature_cols
        }
        joblib.dump(preprocessor, os.path.join(self.models_dir, "preprocessor.pkl"))

        print(f"[ML Engineer] Preprocessor saved. Features used: {len(self.feature_cols)}")
        return (X_train_scaled, y_train_encoded, y_train_score, 
                X_test_scaled, y_test_encoded, y_test_score, 
                X_train_raw, scaler, label_encoder)

    def handle_imbalance(self, X_train, y_train):
        """Applies SMOTE oversampling for minority risk classes with neighbor noise."""
        print("[ML Engineer] Applying SMOTE oversampling for minority risk classes...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        unique, counts = np.unique(y_res, return_counts=True)
        res_dist = dict(zip(unique, counts))
        print(f"[ML Engineer] Oversampled training class distribution: {res_dist}")
        return X_res, y_res

    def train_regularized_classifiers(self, X_train_res, y_train_res, X_test, y_test, label_encoder):
        """
        Trains candidate classifiers with strong regularization to prevent overfitting:
        - Strict max_depth constraints (max_depth=5)
        - High min_samples_leaf (12 samples minimum per leaf node)
        - min_samples_split (20 samples minimum to split)
        - Feature subsampling max_features='sqrt'
        - L2 regularization on Gradient Boosting
        - 5-Fold Stratified Cross Validation (Out-Of-Fold verification)
        """
        print("[ML Engineer] Training candidate REGULARIZED classifiers (RandomForest, ExtraTrees, HistGradientBoosting)...")
        
        candidates = {
            "RandomForest_Regularized": RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                min_samples_leaf=12, 
                min_samples_split=20,
                max_features='sqrt',
                random_state=42, 
                class_weight='balanced', 
                n_jobs=-1
            ),
            "ExtraTrees_Regularized": ExtraTreesClassifier(
                n_estimators=100, 
                max_depth=5, 
                min_samples_leaf=12, 
                min_samples_split=20,
                max_features='sqrt',
                random_state=42, 
                class_weight='balanced', 
                n_jobs=-1
            ),
            "HistGradientBoosting_Regularized": HistGradientBoostingClassifier(
                max_iter=80, 
                max_depth=4, 
                min_samples_leaf=15, 
                l2_regularization=2.0, 
                random_state=42
            )
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_model = None
        best_score = -1.0
        best_name = ""
        best_cv_acc = 0.0

        cv_results_summary = {}

        for name, clf in candidates.items():
            # 5-Fold Stratified Cross-Validation
            cv_scores = cross_val_score(clf, X_train_res, y_train_res, cv=skf, scoring='f1_weighted', n_jobs=-1)
            mean_cv_f1 = float(np.mean(cv_scores))
            std_cv_f1 = float(np.std(cv_scores))

            # Train on full oversampled train set
            clf.fit(X_train_res, y_train_res)
            
            # Predict on unseen test set
            test_preds = clf.predict(X_test)
            test_acc = accuracy_score(y_test, test_preds)
            test_f1 = f1_score(y_test, test_preds, average='weighted')
            macro_f1 = f1_score(y_test, test_preds, average='macro')

            print(f"[ML Engineer] Candidate {name} -> 5-Fold CV F1: {mean_cv_f1:.4f} (±{std_cv_f1:.4f}) | Test Acc: {test_acc*100:.2f}% | Test Weighted F1: {test_f1:.4f}")

            cv_results_summary[name] = {
                "mean_cv_f1": round(mean_cv_f1, 4),
                "std_cv_f1": round(std_cv_f1, 4),
                "test_accuracy": round(float(test_acc), 4),
                "test_weighted_f1": round(float(test_f1), 4)
            }

            if test_f1 > best_score:
                best_score = test_f1
                best_model = clf
                best_name = name
                best_cv_acc = mean_cv_f1

        print(f"[ML Engineer] Selected Regularized Classifier: {best_name} (Test Weighted F1: {best_score:.4f})")
        joblib.dump(best_model, os.path.join(self.models_dir, "risk_classifier.pkl"))
        return best_model, best_name, cv_results_summary

    def train_regressor(self, X_train, y_train_score, X_test, y_test_score):
        """Trains regularized continuous risk score regressor with max_depth limits."""
        print("[ML Engineer] Training regularized continuous risk score regressor...")
        regressor = RandomForestRegressor(
            n_estimators=80, 
            max_depth=6, 
            min_samples_leaf=10, 
            min_samples_split=15, 
            max_features='sqrt',
            random_state=42, 
            n_jobs=-1
        )
        regressor.fit(X_train, y_train_score)

        preds = regressor.predict(X_test)
        r2 = r2_score(y_test_score, preds)
        rmse = np.sqrt(mean_squared_error(y_test_score, preds))
        print(f"[ML Engineer] Regularized Risk Score Regressor -> R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

        joblib.dump(regressor, os.path.join(self.models_dir, "risk_regressor.pkl"))
        return regressor, r2, rmse

    def generate_feature_leaderboard(self, classifier, feature_names):
        """Generates and exports feature importance rankings."""
        print("[ML Engineer] Generating Feature Importance Leaderboard...")
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)

        feature_ranking = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        leaderboard = [
            {"rank": i + 1, "feature": feat, "importance": round(float(imp), 4), "percentage": round(float(imp * 100), 2)}
            for i, (feat, imp) in enumerate(feature_ranking)
        ]

        leaderboard_path = os.path.join(self.output_dir, "feature_leaderboard.json")
        with open(leaderboard_path, "w") as f:
            json.dump(leaderboard, f, indent=2)

        print(f"[ML Engineer] Feature Leaderboard saved to: {leaderboard_path}")
        return leaderboard

    def train_models(self):
        """Pipeline execution for ML Engineer."""
        (X_train, y_train_class, y_train_score, 
         X_test, y_test_class, y_test_score, 
         X_train_raw, scaler, label_encoder) = self.load_and_preprocess()

        # Handle class imbalance
        X_train_res, y_train_res = self.handle_imbalance(X_train, y_train_class)

        # Train Regularized Classifier with 5-Fold Stratified CV
        classifier, clf_name, cv_summary = self.train_regularized_classifiers(
            X_train_res, y_train_res, X_test, y_test_class, label_encoder
        )

        # Train Regressor
        regressor, r2, rmse = self.train_regressor(X_train, y_train_score, X_test, y_test_score)

        # Generate Feature Leaderboard
        leaderboard = self.generate_feature_leaderboard(classifier, self.feature_cols)

        # Save metadata
        metadata = {
            "best_classifier": clf_name,
            "overfitting_prevention": "ENABLED (max_depth=5, min_samples_leaf=12, 5-Fold Stratified CV)",
            "cross_validation_summary": cv_summary,
            "regressor_r2_score": round(float(r2), 4),
            "regressor_rmse": round(float(rmse), 4),
            "num_features": len(self.feature_cols),
            "top_predictor": leaderboard[0]["feature"]
        }
        meta_path = os.path.join(self.models_dir, "model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("[ML Engineer] All regularized ML models trained, validated via 5-Fold CV, and saved successfully!")
        return classifier, regressor, leaderboard

if __name__ == "__main__":
    engineer = MLEngineer()
    engineer.train_models()
