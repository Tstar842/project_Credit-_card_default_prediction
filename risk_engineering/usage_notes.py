"""Usage notes for the engineered credit default pipeline.

Train from the project root:
    python -m risk_engineering.train_pipeline --input sample.csv --output-dir risk_engineering_outputs

Score a new batch:
    python -m risk_engineering.score_batch --input new_sample_3.csv --model-package risk_engineering_outputs/model_package.pkl

Primary outputs:
    risk_engineering_outputs/model_package.pkl
    risk_engineering_outputs/metrics_summary_engineered.csv
    risk_engineering_outputs/business_threshold_optimized_metrics.csv
    risk_engineering_outputs/approval_decision_output.csv
    risk_engineering_outputs/global_feature_importance.csv
    risk_engineering_outputs/monitoring_report.csv
"""

