"""
graders/grader_easy.py
Grader for Task 1 (easy): single-column type mismatch.

Ground truth: 'revenue' column is float64, values like 1234.56.
Agent must strip currency symbols and cast to float.

Score breakdown:
    0.10 — column exists
    0.30 — correct dtype (float64)
    0.50 — value correctness (% of rows matching to 2dp)
    0.10 — no unexpected nulls introduced
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def grade(final_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    score = 0.0

    # Check 1: column exists
    if "revenue" not in final_df.columns:
        return 0.0
    score += 0.10

    # Check 2: correct dtype
    if pd.api.types.is_float_dtype(final_df["revenue"]):
        score += 0.30
    elif pd.api.types.is_numeric_dtype(final_df["revenue"]):
        score += 0.10  # numeric but not float — partial credit

    # Check 3: value correctness
    try:
        gt_vals   = ground_truth_df["revenue"].round(2).values
        act_vals  = pd.to_numeric(final_df["revenue"], errors="coerce").round(2).values
        if len(act_vals) == len(gt_vals):
            row_match = np.isclose(act_vals, gt_vals, rtol=1e-2, atol=1e-2,
                                   equal_nan=False)
            score += 0.50 * float(row_match.mean())
        else:
            # partial credit for count proximity
            ratio = min(len(act_vals), len(gt_vals)) / max(len(act_vals), len(gt_vals))
            score += 0.25 * ratio
    except Exception:
        pass

    # Check 4: no nulls introduced
    if final_df["revenue"].isnull().sum() == 0:
        score += 0.10

    return round(min(1.0, score), 4)