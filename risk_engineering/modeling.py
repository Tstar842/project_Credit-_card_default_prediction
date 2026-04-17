"""Training, evaluation, and model package orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.combine import SMOTETomek
except ImportError:  # pragma: no cover
    SMOTETomek = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

from .business import make_approval_output, search_best_business_threshold, threshold_grid
from .config import BusinessConfig, MonitoringConfig, PipelineConfig, ScoreConfig
from .monitoring import (
    MonitoringBaseline,
    build_monitoring_baseline,
    generate_reason_codes,
    global_feature_importance,
    monitor_scoring_batch,
)
from .preprocessing import PreprocessArtifacts, fit_preprocess_rules, transform_with_rules


@dataclass
class ModelPackage:
    config: PipelineConfig
    business_config: BusinessConfig
    score_config: ScoreConfig
    monitoring_config: MonitoringConfig
    preprocess_artifacts: PreprocessArtifacts
    model: Any
    model_name: str
    threshold: float
    feature_importance: pd.DataFrame
    monitoring_baseline: MonitoringBaseline
    train_reference_frame: pd.DataFrame
    train_reference_scores: np.ndarray


def calculate_ks(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def calculate_psi(train_scores: pd.Series | np.ndarray, test_scores: pd.Series | np.ndarray, bins: int = 10) -> float:
    train = pd.Series(train_scores).replace([np.inf, -np.inf], np.nan)
    test = pd.Series(test_scores).replace([np.inf, -np.inf], np.nan)
    train = train.fillna(train.median())
    test = test.fillna(train.median())
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_bin = pd.cut(train, bins=edges, include_lowest=True)
    test_bin = pd.cut(test, bins=edges, include_lowest=True)
    categories = train_bin.cat.categories
    train_counts = train_bin.value_counts().reindex(categories, fill_value=0).to_numpy()
    test_counts = test_bin.value_counts().reindex(categories, fill_value=0).to_numpy()
    train_ratio = (train_counts + 1e-6) / (train_counts.sum() + 1e-6)
    test_ratio = (test_counts + 1e-6) / (test_counts.sum() + 1e-6)
    return float(np.sum((test_ratio - train_ratio) * np.log(test_ratio / train_ratio)))


def build_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=random_state),
        "LogisticRegression_L1": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            random_state=random_state,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=random_state,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
        ),
    }
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbosity=-1,
        )
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    return models


def evaluate_model(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    monitoring_config: MonitoringConfig,
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray]:
    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    result = {
        "model": model_name,
        "Accuracy": float(accuracy_score(y_test, test_pred)),
        "Precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "F1": float(f1_score(y_test, test_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_test, test_prob)),
        "AP": float(average_precision_score(y_test, test_prob)),
        "KS": calculate_ks(y_test, test_prob),
        "PSI": calculate_psi(train_prob, test_prob, monitoring_config.bins),
    }
    return result, train_prob, test_prob


def fit_credit_risk_pipeline(
    df: pd.DataFrame,
    config: PipelineConfig | None = None,
    business_config: BusinessConfig | None = None,
    score_config: ScoreConfig | None = None,
    monitoring_config: MonitoringConfig | None = None,
    final_cols: list[str] | None = None,
    bin_cols: list[str] | None = None,
) -> tuple[ModelPackage, dict[str, pd.DataFrame]]:
    config = config or PipelineConfig()
    business_config = business_config or BusinessConfig()
    score_config = score_config or ScoreConfig()
    monitoring_config = monitoring_config or MonitoringConfig()

    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df[config.target_col],
    )
    final_cols = final_cols if final_cols is not None else list(config.final_cols)
    bin_cols = bin_cols if bin_cols is not None else list(config.bin_cols)
    preprocess_artifacts = fit_preprocess_rules(train_df, config, final_cols=final_cols, bin_cols=bin_cols)
    train_processed = transform_with_rules(train_df, preprocess_artifacts, config, require_target=True)
    test_processed = transform_with_rules(test_df, preprocess_artifacts, config, require_target=True)

    x_train = train_processed[preprocess_artifacts.feature_cols]
    y_train = train_processed[config.target_col].astype(int)
    x_test = test_processed[preprocess_artifacts.feature_cols]
    y_test = test_processed[config.target_col].astype(int)

    x_fit = x_train
    y_fit = y_train
    if SMOTETomek is not None:
        sampler = SMOTETomek(random_state=config.random_state)
        x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
        x_fit = pd.DataFrame(x_resampled, columns=x_train.columns)
        y_fit = pd.Series(y_resampled, name=config.target_col)

    models = build_models(config.random_state)
    metric_rows: list[dict[str, float | str]] = []
    business_rows: list[dict[str, float | str]] = []
    trained: dict[str, Any] = {}
    prob_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grid = threshold_grid(config.threshold_grid_min, config.threshold_grid_max, config.threshold_grid_size)

    for model_name, model in models.items():
        model.fit(x_fit, y_fit)
        trained[model_name] = model
        metrics, train_prob_original, test_prob = evaluate_model(
            model, x_train, y_train, x_test, y_test, model_name, monitoring_config
        )
        metric_rows.append(metrics)
        best_threshold, _ = search_best_business_threshold(y_train, train_prob_original, business_config, grid)
        test_business = search_best_business_threshold(y_test, test_prob, business_config, np.array([best_threshold["threshold"]]))[0]
        business_row = {"model": model_name, **test_business.to_dict()}
        business_rows.append(business_row)
        prob_cache[model_name] = (train_prob_original, test_prob)

    metrics_df = pd.DataFrame(metric_rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    business_df = pd.DataFrame(business_rows).sort_values(
        "expected_profit_per_user",
        ascending=False,
    ).reset_index(drop=True)

    if config.best_model_metric == "auc":
        best_model_name = str(metrics_df.iloc[0]["model"])
    else:
        best_model_name = str(business_df.iloc[0]["model"])

    best_model = trained[best_model_name]
    train_scores, test_scores = prob_cache[best_model_name]
    best_threshold = float(business_df.loc[business_df["model"] == best_model_name, "threshold"].iloc[0])
    feature_importance = global_feature_importance(best_model, preprocess_artifacts.feature_cols, best_model_name)
    train_threshold_decisions = np.where(train_scores >= best_threshold, "reject", "approve")
    reason_codes = generate_reason_codes(x_test, feature_importance, x_train)
    approval_output = make_approval_output(
        ids=test_processed[config.id_col],
        probs=test_scores,
        threshold=best_threshold,
        model_name=best_model_name,
        score_config=score_config,
        y_true=y_test,
        reason_codes=reason_codes,
    )
    monitoring_baseline = build_monitoring_baseline(
        x_train,
        train_scores,
        y_train,
        train_threshold_decisions,
        monitoring_config,
    )
    monitoring_report = monitor_scoring_batch(
        baseline_frame=x_train,
        scoring_frame=x_test,
        baseline_scores=train_scores,
        scoring_scores=test_scores,
        baseline=monitoring_baseline,
        monitoring_config=monitoring_config,
    )

    package = ModelPackage(
        config=config,
        business_config=business_config,
        score_config=score_config,
        monitoring_config=monitoring_config,
        preprocess_artifacts=preprocess_artifacts,
        model=best_model,
        model_name=best_model_name,
        threshold=best_threshold,
        feature_importance=feature_importance,
        monitoring_baseline=monitoring_baseline,
        train_reference_frame=x_train,
        train_reference_scores=train_scores,
    )
    outputs = {
        "metrics_summary_engineered": metrics_df,
        "business_threshold_optimized_metrics": business_df,
        "approval_decision_output": approval_output,
        "global_feature_importance": feature_importance,
        "monitoring_report": monitoring_report,
        "test_predictions": pd.DataFrame({
            config.id_col: test_processed[config.id_col].to_numpy(),
            "y_true": y_test.to_numpy(),
            "pd_score": test_scores,
        }),
    }
    return package, outputs


def score_with_package(
    df: pd.DataFrame,
    package: ModelPackage,
    include_monitoring: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    processed = transform_with_rules(
        df,
        package.preprocess_artifacts,
        package.config,
        require_target=package.config.target_col in df.columns,
    )
    x_score = processed[package.preprocess_artifacts.feature_cols]
    probs = package.model.predict_proba(x_score)[:, 1]
    reason_codes = generate_reason_codes(x_score, package.feature_importance, package.train_reference_frame)
    output = make_approval_output(
        ids=processed[package.config.id_col],
        probs=probs,
        threshold=package.threshold,
        model_name=package.model_name,
        score_config=package.score_config,
        y_true=processed[package.config.target_col] if package.config.target_col in processed.columns else None,
        reason_codes=reason_codes,
    )
    monitoring_report = None
    if include_monitoring:
        monitoring_report = monitor_scoring_batch(
            baseline_frame=package.train_reference_frame,
            scoring_frame=x_score,
            baseline_scores=package.train_reference_scores,
            scoring_scores=probs,
            baseline=package.monitoring_baseline,
            monitoring_config=package.monitoring_config,
        )
    return output, monitoring_report


def save_model_package(package: ModelPackage, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path)


def load_model_package(path: str | Path) -> ModelPackage:
    return joblib.load(path)
