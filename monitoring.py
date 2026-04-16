import pandas as pd
import mlflow
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric

def run_monitoring(reference_path="data/raw/original_data.csv", current_path="data/processed/train.csv"):
    # 1. Load the datasets
    # In a real scenario, 'current' would be data logged from your API
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    # 2. Create the Monitoring Report
    # This checks if the statistical distribution of your data has shifted
    report = Report(metrics=[
        DataDriftPreset(), 
        TargetDriftPreset(),
        DatasetDriftMetric()
    ])

    report.run(reference_data=reference_df, current_data=current_df)

    # 3. Save the report as an Artifact
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.html"
    report.save_html(report_path)

    # 4. Log the result to MLflow
    # This fulfills the requirement to 'support evaluation'
    with mlflow.start_run(run_name="Data_Monitoring"):
        mlflow.log_artifact(report_path)
        # We can extract specific metrics to log as numbers
        result = report.as_dict()
        drift_score = result['metrics'][2]['result']['number_of_drifted_columns']
        mlflow.log_metric("drifted_columns", drift_score)
        
        print(f"✅ Monitoring complete. Drifted columns: {drift_score}")
        print(f"📊 Report saved to {report_path}")

if __name__ == "__main__":
    run_monitoring()
