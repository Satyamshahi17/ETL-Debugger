"""
graders/grader_hard.py
Grader for Task 3 (hard): silent data corruption.

Ground truth: 200 rows, date range 2024-01-01 to 2024-07-18 inclusive,
category D rows have correct daily_total values.

Agent must: detect missing rows (date filter bug), fix category D totals.

Score breakdown:
    0.20 — row count within 2% of ground truth (198–202)
    0.25 — date range correctness (min/max dates)
    0.30 — category D daily_total distribution matches ground truth
    0.25 — overall amount distribution similarity (KS test proxy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def grade(final_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    score = 0.0

    if len(final_df) == 0:
        return 0.0

    gt = ground_truth_df.copy()

    # Check 1: row count
    gt_n  = len(gt)
    act_n = len(final_df)
    count_ratio = min(act_n, gt_n) / max(act_n, gt_n)
    if count_ratio >= 0.98:
        score += 0.20
    else:
        score += 0.20 * count_ratio

    # Check 2: date range (if date column exists)
    if "date" in final_df.columns and "date" in gt.columns:
        try:
            act_dates = pd.to_datetime(final_df["date"], errors="coerce")
            gt_dates  = pd.to_datetime(gt["date"], errors="coerce")
            act_min, act_max = act_dates.min(), act_dates.max()
            gt_min,  gt_max  = gt_dates.min(),  gt_dates.max()
            # Award partial based on date proximity
            days_tolerance = 2
            min_ok = abs((act_min - gt_min).days) <= days_tolerance
            max_ok = abs((act_max - gt_max).days) <= days_tolerance
            score += 0.25 * ((min_ok + max_ok) / 2)
        except Exception:
            pass

    # Check 3: category D daily_total fixed
    if "category" in final_df.columns and "daily_total" in final_df.columns:
        try:
            act_d = final_df[final_df["category"] == "D"]["daily_total"]
            gt_d  = gt[gt["category"] == "D"]["daily_total"]
            if len(act_d) > 0 and len(gt_d) > 0:
                # In the broken version, category D totals are all 0
                # Fixed: they should be > 0
                frac_nonzero = (act_d > 0).mean()
                gt_nonzero   = (gt_d > 0).mean()
                cat_d_score  = 1.0 - abs(frac_nonzero - gt_nonzero)
                score += 0.30 * max(0.0, cat_d_score)
        except Exception:
            pass

    # Check 4: overall amount distribution similarity
    if "amount" in final_df.columns and "amount" in gt.columns:
        try:
            act_amounts = pd.to_numeric(final_df["amount"], errors="coerce").dropna()
            gt_amounts  = pd.to_numeric(gt["amount"], errors="coerce").dropna()
            if len(act_amounts) > 0 and len(gt_amounts) > 0:
                act_mean = act_amounts.mean()
                gt_mean  = gt_amounts.mean()
                act_std  = act_amounts.std()
                gt_std   = gt_amounts.std()
                mean_sim = 1.0 - min(1.0, abs(act_mean - gt_mean) / max(1.0, abs(gt_mean)))
                std_sim  = 1.0 - min(1.0, abs(act_std  - gt_std)  / max(1.0, abs(gt_std) + 1e-9))
                score += 0.25 * (0.6 * mean_sim + 0.4 * std_sim)
        except Exception:
            pass

    return round(min(1.0, score), 4)