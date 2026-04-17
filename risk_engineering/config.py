"""Central configuration for the credit default risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DEFAULT_FINAL_COLS = (
    "user_id", "y", "x_001", "x_002", "x_003", "x_004", "x_005", "x_006",
    "x_019", "x_020", "x_021", "x_027", "x_033", "x_034", "x_035", "x_036",
    "x_037", "x_038", "x_041", "x_042", "x_044", "x_045", "x_048", "x_049",
    "x_052", "x_054", "x_055", "x_056", "x_074", "x_075", "x_077", "x_078",
    "x_088", "x_089", "x_121", "x_122", "x_124", "x_125", "x_131", "x_132",
    "x_134", "x_137", "x_142", "x_143", "x_144", "x_149", "x_154", "x_155",
    "x_157", "x_159", "x_162", "x_188", "x_189", "x_190", "x_196", "x_197",
    "x_198",
)


DEFAULT_BIN_COLS = (
    "x_002", "x_020", "x_021", "x_027", "x_033", "x_034", "x_035", "x_036",
    "x_037", "x_038", "x_041", "x_042", "x_044", "x_045", "x_048", "x_049",
    "x_052", "x_054", "x_055", "x_056", "x_074", "x_075", "x_077", "x_078",
    "x_088", "x_089", "x_121", "x_122", "x_124", "x_125", "x_131", "x_132",
    "x_134", "x_137", "x_142", "x_143", "x_144", "x_149", "x_154", "x_155",
    "x_157", "x_159", "x_162", "x_188", "x_189", "x_190", "x_196", "x_197",
    "x_198",
)


@dataclass(frozen=True)
class PipelineConfig:
    id_col: str = "user_id"
    target_col: str = "y"
    test_size: float = 0.30
    random_state: int = 42
    high_missing_threshold: float = 0.70
    moderate_missing_lower: float = 0.50
    moderate_missing_upper: float = 0.70
    rf_imputer_estimators: int = 80
    bin_quantiles: int = 6
    threshold_grid_min: float = 0.01
    threshold_grid_max: float = 0.99
    threshold_grid_size: int = 99
    best_model_metric: Literal["business_profit", "auc"] = "business_profit"
    final_cols: tuple[str, ...] = DEFAULT_FINAL_COLS
    bin_cols: tuple[str, ...] = DEFAULT_BIN_COLS


@dataclass(frozen=True)
class BusinessConfig:
    """Cost matrix assumptions.

    Prediction convention:
    - score/probability is the probability of default.
    - y_pred = 1 means reject/high risk.
    - y_pred = 0 means approve/low risk.
    """

    profit_good_approved: float = 1000.0
    loss_bad_approved: float = 5000.0
    opportunity_loss_good_rejected: float = 300.0
    benefit_bad_rejected: float = 0.0
    min_recall: float = 0.55
    min_approval_rate: float = 0.30
    max_approval_rate: float = 0.85
    max_bad_approval_rate: float = 0.20
    optimize_mode: Literal["max_profit", "max_approval_under_risk"] = "max_profit"


@dataclass(frozen=True)
class ScoreConfig:
    base_score: float = 600.0
    base_odds: float = 20.0
    pdo: float = 50.0
    min_score: int = 300
    max_score: int = 850
    # Probability bands are [min_prob, max_prob).
    probability_grade_rules: tuple[dict[str, object], ...] = field(
        default_factory=lambda: (
            {"grade": "A", "min_prob": 0.00, "max_prob": 0.10, "decision": "auto_approve"},
            {"grade": "B", "min_prob": 0.10, "max_prob": 0.20, "decision": "auto_approve"},
            {"grade": "C", "min_prob": 0.20, "max_prob": 0.35, "decision": "manual_review"},
            {"grade": "D", "min_prob": 0.35, "max_prob": 0.50, "decision": "manual_review"},
            {"grade": "E", "min_prob": 0.50, "max_prob": 1.01, "decision": "reject"},
        )
    )


@dataclass(frozen=True)
class MonitoringConfig:
    psi_stable: float = 0.10
    psi_warning: float = 0.25
    bins: int = 10
    missing_rate_warning_delta: float = 0.10
