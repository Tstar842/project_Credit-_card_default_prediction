"""Model explainability and monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MonitoringConfig


@dataclass
class MonitoringBaseline:
    feature_missing_rate: dict[str, float]
    feature_quantile_edges: dict[str, np.ndarray] = field(default_factory=dict)
    score_edges: np.ndarray | None = None
    approval_rate: float | None = None
    bad_rate: float | None = None


def calculate_psi_from_edges(
    expected_values: pd.Series | np.ndarray,
    actual_values: pd.Series | np.ndarray,
    edges: np.ndarray,
) -> float:
    expected = pd.Series(expected_values).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual_values).replace([np.inf, -np.inf], np.nan).dropna()
    if expected.empty or actual.empty or len(edges) < 3:
        return 0.0
    expected_bin = pd.cut(expected, bins=edges, include_lowest=True)
    actual_bin = pd.cut(actual, bins=edges, include_lowest=True)
    categories = expected_bin.cat.categories
    expected_counts = expected_bin.value_counts().reindex(categories, fill_value=0).to_numpy()
    actual_counts = actual_bin.value_counts().reindex(categories, fill_value=0).to_numpy()
    expected_ratio = (expected_counts + 1e-6) / (expected_counts.sum() + 1e-6)
    actual_ratio = (actual_counts + 1e-6) / (actual_counts.sum() + 1e-6)
    return float(np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio)))


def _quantile_edges(values: pd.Series | np.ndarray, bins: int) -> np.ndarray | None:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if series.nunique() < 3:
        return None
    edges = np.unique(np.quantile(series, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def build_monitoring_baseline(
    x_train_processed: pd.DataFrame,
    train_scores: pd.Series | np.ndarray,
    y_train: pd.Series | np.ndarray | None,
    approval_decisions: pd.Series | np.ndarray | None,
    monitoring_config: MonitoringConfig,
) -> MonitoringBaseline:
    feature_edges: dict[str, np.ndarray] = {}
    for col in x_train_processed.columns:
        edges = _quantile_edges(x_train_processed[col], monitoring_config.bins)
        if edges is not None:
            feature_edges[col] = edges

    score_edges = _quantile_edges(train_scores, monitoring_config.bins)
    approval_rate = None
    if approval_decisions is not None:
        approval_rate = float(pd.Series(approval_decisions).isin(["approve", "auto_approve"]).mean())
    bad_rate = None if y_train is None else float(pd.Series(y_train).mean())

    return MonitoringBaseline(
        feature_missing_rate=x_train_processed.isna().mean().to_dict(),
        feature_quantile_edges=feature_edges,
        score_edges=score_edges,
        approval_rate=approval_rate,
        bad_rate=bad_rate,
    )


def psi_status(value: float, monitoring_config: MonitoringConfig) -> str:
    if value < monitoring_config.psi_stable:
        return "stable"
    if value < monitoring_config.psi_warning:
        return "warning"
    return "alert"


def monitor_scoring_batch(
    baseline_frame: pd.DataFrame,
    scoring_frame: pd.DataFrame,
    baseline_scores: pd.Series | np.ndarray,
    scoring_scores: pd.Series | np.ndarray,
    baseline: MonitoringBaseline,
    monitoring_config: MonitoringConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if baseline.score_edges is not None:
        score_psi = calculate_psi_from_edges(baseline_scores, scoring_scores, baseline.score_edges)
        rows.append({
            "metric_type": "score_psi",
            "metric_name": "model_score",
            "value": score_psi,
            "status": psi_status(score_psi, monitoring_config),
            "threshold": monitoring_config.psi_warning,
        })

    for col, edges in baseline.feature_quantile_edges.items():
        if col not in scoring_frame.columns or col not in baseline_frame.columns:
            continue
        value = calculate_psi_from_edges(baseline_frame[col], scoring_frame[col], edges)
        rows.append({
            "metric_type": "feature_psi",
            "metric_name": col,
            "value": value,
            "status": psi_status(value, monitoring_config),
            "threshold": monitoring_config.psi_warning,
        })

    for col, baseline_rate in baseline.feature_missing_rate.items():
        if col not in scoring_frame.columns:
            continue
        scoring_rate = float(scoring_frame[col].isna().mean())
        delta = scoring_rate - float(baseline_rate)
        status = "warning" if abs(delta) >= monitoring_config.missing_rate_warning_delta else "stable"
        rows.append({
            "metric_type": "missing_rate_delta",
            "metric_name": col,
            "value": delta,
            "status": status,
            "threshold": monitoring_config.missing_rate_warning_delta,
        })

    return pd.DataFrame(rows)


def global_feature_importance(model: object, feature_names: list[str], model_name: str) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        return pd.DataFrame(columns=["model", "feature", "importance", "rank"])

    importance = pd.DataFrame({
        "model": model_name,
        "feature": feature_names,
        "importance": values,
    }).sort_values("importance", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance


def generate_reason_codes(
    x_processed: pd.DataFrame,
    feature_importance: pd.DataFrame,
    reference_frame: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    top_features = feature_importance.sort_values("rank").head(20)["feature"].tolist()
    stats = reference_frame[top_features].quantile([0.25, 0.75]).to_dict()
    records: list[dict[str, str]] = []
    for _, row in x_processed.iterrows():
        reasons: list[str] = []
        for feature in top_features:
            if feature not in row:
                continue
            value = row[feature]
            q25 = stats.get(feature, {}).get(0.25)
            q75 = stats.get(feature, {}).get(0.75)
            if pd.isna(q25) or pd.isna(q75):
                continue
            if value >= q75:
                reasons.append(f"{feature} >= train_P75")
            elif value <= q25:
                reasons.append(f"{feature} <= train_P25")
            if len(reasons) >= top_n:
                break
        while len(reasons) < top_n:
            reasons.append("")
        records.append({f"reason_code_{i + 1}": reasons[i] for i in range(top_n)})
    return pd.DataFrame(records)

