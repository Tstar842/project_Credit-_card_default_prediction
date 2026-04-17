"""Business threshold search and approval output helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import BusinessConfig, ScoreConfig


def threshold_grid(min_value: float, max_value: float, size: int) -> np.ndarray:
    return np.round(np.linspace(min_value, max_value, size), 4)


def business_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float,
    business_config: BusinessConfig,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    y_pred = (y_score_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    n = len(y_true_arr)

    approved_good = tn
    rejected_good = fp
    approved_bad = fn
    rejected_bad = tp
    expected_profit = (
        approved_good * business_config.profit_good_approved
        - approved_bad * business_config.loss_bad_approved
        - rejected_good * business_config.opportunity_loss_good_rejected
        + rejected_bad * business_config.benefit_bad_rejected
    )
    approved = approved_good + approved_bad

    return {
        "threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "Precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "approval_rate": float(approved / n) if n else 0.0,
        "reject_rate": float((rejected_good + rejected_bad) / n) if n else 0.0,
        "bad_approval_rate": float(approved_bad / approved) if approved else 0.0,
        "expected_profit": float(expected_profit),
        "expected_profit_per_user": float(expected_profit / n) if n else 0.0,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def search_best_business_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    business_config: BusinessConfig,
    grid: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:
    records = [business_metrics(y_true, y_score, float(thr), business_config) for thr in grid]
    search_df = pd.DataFrame(records)

    feasible = search_df[
        (search_df["Recall"] >= business_config.min_recall)
        & (search_df["approval_rate"] >= business_config.min_approval_rate)
        & (search_df["approval_rate"] <= business_config.max_approval_rate)
        & (search_df["bad_approval_rate"] <= business_config.max_bad_approval_rate)
    ].copy()
    if feasible.empty:
        feasible = search_df.copy()

    if business_config.optimize_mode == "max_approval_under_risk":
        best = feasible.sort_values(
            by=["approval_rate", "expected_profit_per_user"],
            ascending=[False, False],
        ).iloc[0]
    else:
        best = feasible.sort_values(
            by=["expected_profit_per_user", "Recall"],
            ascending=[False, False],
        ).iloc[0]
    return best, search_df


def probability_to_score(prob: pd.Series | np.ndarray, score_config: ScoreConfig) -> np.ndarray:
    prob_arr = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    odds = (1.0 - prob_arr) / prob_arr
    factor = score_config.pdo / np.log(2)
    offset = score_config.base_score - factor * np.log(score_config.base_odds)
    score = offset + factor * np.log(odds)
    return np.clip(np.round(score), score_config.min_score, score_config.max_score).astype(int)


def assign_risk_grade(prob: float, score_config: ScoreConfig) -> tuple[str, str]:
    for rule in score_config.probability_grade_rules:
        if float(rule["min_prob"]) <= prob < float(rule["max_prob"]):
            return str(rule["grade"]), str(rule["decision"])
    return "E", "reject"


def make_approval_output(
    ids: pd.Series | np.ndarray,
    probs: pd.Series | np.ndarray,
    threshold: float,
    model_name: str,
    score_config: ScoreConfig,
    y_true: pd.Series | np.ndarray | None = None,
    reason_codes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    probs_arr = np.asarray(probs, dtype=float)
    scores = probability_to_score(probs_arr, score_config)
    grades_decisions = [assign_risk_grade(prob, score_config) for prob in probs_arr]
    output = pd.DataFrame({
        "user_id": np.asarray(ids),
        "pd_score": probs_arr,
        "risk_score": scores,
        "risk_grade": [item[0] for item in grades_decisions],
        "decision": [item[1] for item in grades_decisions],
        "threshold_decision": np.where(probs_arr >= threshold, "reject", "approve"),
        "threshold": threshold,
        "model_name": model_name,
    })
    if y_true is not None:
        output.insert(1, "y_true", np.asarray(y_true))
    if reason_codes is not None:
        output = pd.concat([output.reset_index(drop=True), reason_codes.reset_index(drop=True)], axis=1)
    return output

