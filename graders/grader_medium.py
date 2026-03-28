"""
graders/grader_medium.py
Grader for Task 2 (medium): join key mismatch (userId vs user_id).

Score breakdown:
    0.20 — 'user_id' column exists (not 'userId')
    0.20 — 'name' and 'tier' are not all-null (join fixed)
    0.25 — row count matches ground truth
    0.35 — value correctness on all shared columns
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EXPECTED_COLS = {"user_id", "name", "tier", "amount", "status"}


def grade(final_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    score = 0.0
    if len(final_df) == 0:
        return 0.0

    present = set(final_df.columns)

    # Check 1: user_id present (not broken userId)
    if "user_id" in present:
        score += 0.20

    # Check 2: join artifact fixed — name AND tier not majority-null
    null_fixed = 0
    for col in ["name", "tier"]:
        if col in final_df.columns:
            if final_df[col].isnull().mean() < 0.10:
                null_fixed += 1
    score += 0.20 * (null_fixed / 2)

    # Check 3: row count
    gt_rows = len(ground_truth_df)
    act_rows = len(final_df)
    if act_rows == gt_rows:
        score += 0.25
    else:
        score += 0.25 * 0.5 * min(act_rows, gt_rows) / max(act_rows, gt_rows)

    # Check 4: value correctness
    shared = [c for c in EXPECTED_COLS if c in final_df.columns and c in ground_truth_df.columns]
    if shared and act_rows == gt_rows:
        total_cells = matching_cells = 0
        for col in shared:
            try:
                if pd.api.types.is_numeric_dtype(ground_truth_df[col]):
                    m = np.isclose(
                        pd.to_numeric(final_df[col], errors="coerce").fillna(-999).values,
                        ground_truth_df[col].fillna(-999).values, rtol=1e-2)
                else:
                    m = (final_df[col].astype(str).values ==
                         ground_truth_df[col].astype(str).values)
                matching_cells += int(m.sum())
                total_cells += len(m)
            except Exception:
                total_cells += act_rows
        if total_cells > 0:
            score += 0.35 * (matching_cells / total_cells)

    return round(min(1.0, score), 4)