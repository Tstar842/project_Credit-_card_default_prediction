"""Leakage-safe preprocessing for train and scoring data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import PipelineConfig


@dataclass
class PreprocessArtifacts:
    drop_cols: list[str]
    moderate_missing_cols: list[str]
    rf_imputers: dict[str, RandomForestRegressor] = field(default_factory=dict)
    rf_imputer_features: dict[str, list[str]] = field(default_factory=dict)
    median_values: dict[str, float] = field(default_factory=dict)
    final_cols: list[str] = field(default_factory=list)
    feature_cols: list[str] = field(default_factory=list)
    bin_cols: list[str] = field(default_factory=list)
    bin_edges: dict[str, np.ndarray] = field(default_factory=dict)


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def _numeric_feature_cols(df: pd.DataFrame, config: PipelineConfig, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    blocked = {config.id_col, config.target_col, *exclude}
    return [
        col for col in df.columns
        if col not in blocked and pd.api.types.is_numeric_dtype(df[col])
    ]


def _fit_numeric_medians(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for col in cols:
        median = df[col].median()
        medians[col] = 0.0 if pd.isna(median) else float(median)
    return medians


def _apply_numeric_medians(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col, value in medians.items():
        if col in out.columns:
            out[col] = out[col].fillna(value)
    return out


def build_binning_edges(df: pd.DataFrame, bin_cols: list[str], q: int) -> dict[str, np.ndarray]:
    edges: dict[str, np.ndarray] = {}
    for col in bin_cols:
        if col not in df.columns:
            continue
        non_null = pd.Series(df[col]).dropna()
        if non_null.nunique() < 3:
            continue
        try:
            _, col_edges = pd.qcut(non_null, q=q, duplicates="drop", retbins=True)
        except ValueError:
            continue
        col_edges = np.unique(col_edges)
        if len(col_edges) >= 3:
            col_edges[0] = -np.inf
            col_edges[-1] = np.inf
            edges[col] = col_edges
    return edges


def apply_binning_edges(df: pd.DataFrame, bin_edges: dict[str, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    for col, edges in bin_edges.items():
        if col not in out.columns:
            continue
        out[col] = pd.cut(out[col], bins=edges, labels=False, include_lowest=True)
    return out


def fit_preprocess_rules(
    train_df: pd.DataFrame,
    config: PipelineConfig,
    final_cols: list[str] | None = None,
    bin_cols: list[str] | None = None,
) -> PreprocessArtifacts:
    """Fit preprocessing rules on training data only."""

    train = replace_inf_with_nan(train_df.copy())
    missing_rate = train.isna().mean()
    protected = {config.id_col, config.target_col}
    drop_cols = [
        col for col, rate in missing_rate.items()
        if rate > config.high_missing_threshold and col not in protected
    ]

    train = train.drop(columns=drop_cols, errors="ignore")
    moderate_missing_cols = [
        col for col, rate in train.isna().mean().items()
        if config.moderate_missing_lower <= rate <= config.moderate_missing_upper
        and col not in protected
    ]

    artifacts = PreprocessArtifacts(
        drop_cols=drop_cols,
        moderate_missing_cols=moderate_missing_cols,
    )

    for target_col in moderate_missing_cols:
        train_mask = train[target_col].notna()
        test_mask = train[target_col].isna()
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        candidate_features = _numeric_feature_cols(train, config, exclude={target_col})
        feature_cols = [
            col for col in candidate_features
            if train.loc[train_mask, col].notna().all()
        ]
        if not feature_cols:
            continue

        model = RandomForestRegressor(
            n_estimators=config.rf_imputer_estimators,
            random_state=config.random_state,
            n_jobs=-1,
        )
        x_fit = train.loc[train_mask, feature_cols].copy()
        x_pred = train.loc[test_mask, feature_cols].copy()
        medians = _fit_numeric_medians(x_fit, feature_cols)
        x_fit = _apply_numeric_medians(x_fit, medians)
        x_pred = _apply_numeric_medians(x_pred, medians)
        model.fit(x_fit, train.loc[train_mask, target_col])
        train.loc[test_mask, target_col] = model.predict(x_pred)

        artifacts.rf_imputers[target_col] = model
        artifacts.rf_imputer_features[target_col] = feature_cols

    feature_cols_after_drop = _numeric_feature_cols(train, config)
    artifacts.median_values = _fit_numeric_medians(train, feature_cols_after_drop)
    train = _apply_numeric_medians(train, artifacts.median_values)

    if final_cols is None:
        final_cols = [config.id_col, config.target_col, *feature_cols_after_drop]
    else:
        final_cols = [col for col in final_cols if col in train.columns]

    artifacts.final_cols = final_cols
    artifacts.feature_cols = [col for col in final_cols if col not in {config.id_col, config.target_col}]
    artifacts.bin_cols = [col for col in (bin_cols or artifacts.feature_cols) if col in artifacts.feature_cols]
    artifacts.bin_edges = build_binning_edges(train[artifacts.feature_cols], artifacts.bin_cols, config.bin_quantiles)
    return artifacts


def transform_with_rules(
    df: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    config: PipelineConfig,
    require_target: bool = True,
) -> pd.DataFrame:
    """Apply training-fitted preprocessing rules to train, test, or scoring data."""

    out = replace_inf_with_nan(df.copy())
    out = out.drop(columns=artifacts.drop_cols, errors="ignore")

    for target_col, model in artifacts.rf_imputers.items():
        if target_col not in out.columns:
            out[target_col] = np.nan
        missing_mask = out[target_col].isna()
        if missing_mask.sum() == 0:
            continue
        feature_cols = artifacts.rf_imputer_features[target_col]
        for col in feature_cols:
            if col not in out.columns:
                out[col] = artifacts.median_values.get(col, 0.0)
        x_missing = out.loc[missing_mask, feature_cols].copy()
        x_missing = _apply_numeric_medians(x_missing, {
            col: artifacts.median_values.get(col, 0.0) for col in feature_cols
        })
        out.loc[missing_mask, target_col] = model.predict(x_missing)

    for col in artifacts.feature_cols:
        if col not in out.columns:
            out[col] = artifacts.median_values.get(col, 0.0)
    out = _apply_numeric_medians(out, artifacts.median_values)

    needed_cols = list(artifacts.final_cols)
    if not require_target and config.target_col in needed_cols and config.target_col not in out.columns:
        needed_cols.remove(config.target_col)
    missing_required = [col for col in needed_cols if col not in out.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after preprocessing: {missing_required}")

    out = out[needed_cols].copy()
    out.loc[:, artifacts.feature_cols] = apply_binning_edges(out[artifacts.feature_cols], artifacts.bin_edges)
    out.loc[:, artifacts.feature_cols] = _apply_numeric_medians(out[artifacts.feature_cols], artifacts.median_values)
    return out

