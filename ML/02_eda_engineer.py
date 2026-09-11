"""
================================================================================
ROLE 02: EDA ENGINEER
Disaster Response Coordination System - Stage 01 Machine Learning
================================================================================
Responsibilities:
- Plot and analyze leading risk indicators across 38 Tamil Nadu districts.
- Perform false alarm diagnostic analysis (911 call spikes vs actual water level rises).
- Export comprehensive EDA summary metrics (eda_summary.json).
================================================================================
"""

import os
import json
import pandas as pd
import numpy as np

class EDAEngineer:
    def __init__(self, dataset_path=None, output_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_path = dataset_path or os.path.join(base_dir, "ML", "master_dataset.csv")
        self.output_dir = output_dir or os.path.join(base_dir, "ML")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Loads master engineered dataset."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Master dataset not found at: {self.dataset_path}. Run Data Engineer first.")
        df = pd.read_csv(self.dataset_path)
        return df

    def analyze_leading_indicators(self, df):
        """Identifies features with strongest correlation to risk score."""
        print("[EDA Engineer] Analyzing leading risk indicators...")
        numeric_df = df.select_dtypes(include=[np.number])
        if 'risk_score' not in numeric_df.columns:
            return {}
        
        correlations = numeric_df.corr()['risk_score'].drop('risk_score').sort_values(ascending=False)
        leading_indicators = correlations.round(4).to_dict()
        print(f"[EDA Engineer] Top 5 Leading Indicators: {list(leading_indicators.items())[:5]}")
        return leading_indicators

    def false_alarm_diagnostics(self, df):
        """
        Hunts down historical false alarms:
        Cases with high emergency call volume (>75th percentile) but normal water level (<median)
        versus true crisis events (high water level + high emergency calls).
        """
        print("[EDA Engineer] Running false alarm diagnostic analysis...")
        call_threshold = df['emergency_calls_6h'].quantile(0.75)
        water_median = df['water_level_m'].median()
        
        # High calls but low/normal water level -> Potential False Alarm / Non-flood panic
        false_alarms = df[(df['emergency_calls_6h'] >= call_threshold) & (df['water_level_m'] <= water_median)]
        true_crises = df[(df['emergency_calls_6h'] >= call_threshold) & (df['water_level_m'] > water_median)]
        
        total_high_calls = len(df[df['emergency_calls_6h'] >= call_threshold])
        false_alarm_rate = (len(false_alarms) / total_high_calls) if total_high_calls > 0 else 0.0
        
        diagnostics = {
            "high_call_threshold": float(call_threshold),
            "median_water_level_m": float(water_median),
            "false_alarm_incidents_count": int(len(false_alarms)),
            "true_crisis_incidents_count": int(len(true_crises)),
            "false_alarm_rate": round(float(false_alarm_rate), 4),
            "avg_road_closures_in_false_alarm": round(float(false_alarms['road_closures'].mean() if len(false_alarms) > 0 else 0), 2),
            "avg_road_closures_in_true_crisis": round(float(true_crises['road_closures'].mean() if len(true_crises) > 0 else 0), 2),
            "insight": "High emergency call volume alone creates false alarms unless validated against river gauge 72h max trend & road closure logs."
        }
        print(f"[EDA Engineer] False Alarm Rate detected: {diagnostics['false_alarm_rate']*100:.1f}%")
        return diagnostics

    def district_vulnerability_profile(self, df):
        """Aggregates risk score and severe incident rate per district."""
        print("[EDA Engineer] Calculating district vulnerability profiles...")
        district_stats = df.groupby('district').agg(
            avg_risk_score=('risk_score', 'mean'),
            max_risk_score=('risk_score', 'max'),
            max_water_level=('water_level_m', 'max'),
            severe_count=('risk_level', lambda x: (x == 'Severe').sum()),
            moderate_count=('risk_level', lambda x: (x == 'Moderate').sum()),
            total_observations=('risk_level', 'count')
        ).reset_index()
        
        district_stats['severe_rate'] = (district_stats['severe_count'] / district_stats['total_observations']).round(4)
        top_vulnerable = district_stats.sort_values(by='avg_risk_score', ascending=False).head(10).to_dict(orient='records')
        return top_vulnerable

    def run_analysis(self):
        """Pipeline execution for EDA Engineer."""
        df = self.load_data()
        
        class_dist = df['risk_level'].value_counts().to_dict()
        class_pct = (df['risk_level'].value_counts(normalize=True) * 100).round(2).to_dict()
        
        leading_indicators = self.analyze_leading_indicators(df)
        false_alarm_info = self.false_alarm_diagnostics(df)
        district_profiles = self.district_vulnerability_profile(df)
        
        eda_summary = {
            "total_records": int(len(df)),
            "num_districts": int(df['district'].nunique()),
            "class_distribution": class_dist,
            "class_percentages": class_pct,
            "leading_indicators_correlation": leading_indicators,
            "false_alarm_diagnostics": false_alarm_info,
            "top_10_vulnerable_districts": district_profiles
        }
        
        summary_path = os.path.join(self.output_dir, "eda_summary.json")
        with open(summary_path, "w") as f:
            json.dump(eda_summary, f, indent=2)
            
        print(f"[EDA Engineer] EDA Summary exported to: {summary_path}")
        return eda_summary

if __name__ == "__main__":
    engineer = EDAEngineer()
    engineer.run_analysis()
