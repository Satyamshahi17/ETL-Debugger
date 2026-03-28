"""
server/reward.py
Computes the dense reward signal after each step.

Components (sum = 0.0 to 1.0 before penalty):
    schema_match    (weight 0.40) — column names + dtypes match ground truth
    row_correctness (weight 0.30) — % of rows matching ground truth values
    null_handling   (weight 0.20) — null distribution matches ground truth
    efficiency      (weight 0.10) — penalises wasted/redundant actions
    loop_penalty    (flat −0.30)  — applied when same action repeated 3+ times
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_reward(
    df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    action_history: List[str],
) -> float:
    """
    Returns a float in [0.0, 1.0] (possibly slightly negative due to loop penalty).
    Called inside Environment.step() before building the Observation.
    """
    schema  = _schema_match(df, ground_truth)
    rows    = _row_correctness(df, ground_truth)
    nulls   = _null_handling(df, ground_truth)
    eff     = _efficiency(action_history)
    penalty = _loop_penalty(action_history)

    raw = 0.40 * schema + 0.30 * rows + 0.20 * nulls + 0.10 * eff + penalty
    return float(round(max(-0.30, min(1.0, raw)), 4))


def compute_reward_breakdown(
    df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    action_history: List[str],
) -> dict:
    """Returns the individual sub-scores for diagnostics / metadata."""
    schema  = _schema_match(df, ground_truth)
    rows    = _row_correctness(df, ground_truth)
    nulls   = _null_handling(df, ground_truth)
    eff     = _efficiency(action_history)
    penalty = _loop_penalty(action_history)
    total   = 0.40*schema + 0.30*rows + 0.20*nulls + 0.10*eff + penalty
    return {
        "schema_match":    round(schema,  4),
        "row_correctness": round(rows,    4),
        "null_handling":   round(nulls,   4),
        "efficiency":      round(eff,     4),
        "loop_penalty":    round(penalty, 4),
        "total":           round(max(-0.30, min(1.0, total)), 4),
    }


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

def _schema_match(df: pd.DataFrame, gt: pd.DataFrame) -> float:
    """
    0.0 → no column names match
    0.5 → all names match but some dtypes differ
    1.0 → names and dtypes are identical
    """
    if len(df.columns) == 0:
        return 0.0

    df_cols = set(df.columns)
    gt_cols = set(gt.columns)

    name_score = len(df_cols & gt_cols) / len(gt_cols)
    if name_score == 0:
        return 0.0

    # Compare dtypes for columns that exist in both
    shared = df_cols & gt_cols
    dtype_matches = sum(
        1 for c in shared
        if _compatible_dtype(df[c].dtype, gt[c].dtype)
    )
    dtype_score = dtype_matches / len(gt_cols)

    return 0.5 * name_score + 0.5 * dtype_score


def _row_correctness(df: pd.DataFrame, gt: pd.DataFrame) -> float:
    """
    Rewards getting the right rows with the right values.
    Partial credit if row count differs from ground truth.
    """
    if len(df) == 0:
        return 0.0

    # Row count similarity (cap partial at 0.5 if count differs)
    count_ratio = min(len(df), len(gt)) / max(len(df), len(gt))
    if len(df) != len(gt):
        return 0.5 * count_ratio

    # Same row count — compare values cell-by-cell for shared columns
    shared_cols = [c for c in gt.columns if c in df.columns]
    if not shared_cols:
        return 0.0

    total_cells = 0
    matching_cells = 0
    for col in shared_cols:
        try:
            if pd.api.types.is_numeric_dtype(gt[col]) and pd.api.types.is_numeric_dtype(df[col]):
                matches = np.isclose(
                    df[col].fillna(0).values,
                    gt[col].fillna(0).values,
                    rtol=1e-3, atol=1e-3,
                )
            else:
                matches = (df[col].astype(str).values == gt[col].astype(str).values)
            matching_cells += matches.sum()
            total_cells += len(matches)
        except Exception:
            total_cells += len(df)

    return matching_cells / total_cells if total_cells > 0 else 0.0


def _null_handling(df: pd.DataFrame, gt: pd.DataFrame) -> float:
    """
    Penalises unexpected nulls (agent introduced NaNs) and missing nulls.
    """
    expected = int(gt.isnull().sum().sum())
    actual   = int(df.isnull().sum().sum())
    diff     = abs(actual - expected)
    denom    = max(1, expected, actual)
    return max(0.0, 1.0 - diff / denom)


def _efficiency(action_history: List[str]) -> float:
    """
    Rewards concise, non-repetitive action sequences.
    Each wasted (duplicate) action costs 0.02.
    """
    if not action_history:
        return 1.0
    useful = len(set(action_history))
    wasted = len(action_history) - useful
    return max(0.0, 1.0 - wasted * 0.02)


def _loop_penalty(action_history: List[str]) -> float:
    """
    Returns −0.30 if the last 3 actions are identical (agent is stuck in loop).
    """
    if len(action_history) >= 3:
        last3 = action_history[-3:]
        if len(set(last3)) == 1:
            return -0.30
    return 0.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _compatible_dtype(a, b) -> bool:
    """True if dtypes are the same broad kind (both numeric, both string, etc.)"""
    def kind(dt):
        if pd.api.types.is_integer_dtype(dt):   return "int"
        if pd.api.types.is_float_dtype(dt):     return "float"
        if pd.api.types.is_bool_dtype(dt):      return "bool"
        if pd.api.types.is_datetime64_any_dtype(dt): return "datetime"
        return "object"
    return kind(a) == kind(b)