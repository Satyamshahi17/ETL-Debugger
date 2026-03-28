"""
tasks/fixtures.py
Generates the broken and ground-truth DataFrames for all 3 tasks.
Pure Python / pandas — no external CSV files required.
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class TaskFixture:
    task_id: str
    description: str
    broken_df: pd.DataFrame
    ground_truth_df: pd.DataFrame
    hint: str


# ---------------------------------------------------------------------------
# TASK: easy — single-column type mismatch
# Revenue loaded as currency string instead of float
# ---------------------------------------------------------------------------

def _make_easy() -> TaskFixture:
    rng = np.random.default_rng(42)
    n = 50

    raw_values = rng.uniform(100, 9999, n).round(2)
    # Ground truth: clean float column
    gt = pd.DataFrame({
        "order_id":  [f"ORD{i:04d}" for i in range(n)],
        "customer":  [f"Customer_{i % 10}" for i in range(n)],
        "revenue":   raw_values,
        "quantity":  rng.integers(1, 20, n),
        "region":    rng.choice(["North", "South", "East", "West"], n),
    })

    # Broken: revenue is stored as "$1,234.56" strings
    broken = gt.copy()
    broken["revenue"] = broken["revenue"].apply(lambda v: f"${v:,.2f}")

    return TaskFixture(
        task_id="easy",
        description=(
            "The 'revenue' column was loaded as a currency-formatted string "
            "(e.g. '$1,234.56') instead of float64. Strip currency symbols, "
            "remove commas, and cast to float."
        ),
        broken_df=broken,
        ground_truth_df=gt,
        hint="Use fix_column with transform='strip_currency', then cast_type to float64.",
    )


# ---------------------------------------------------------------------------
# TASK: medium — multi-step join gone wrong (key mismatch → duplicates)
# ---------------------------------------------------------------------------

def _make_medium() -> TaskFixture:
    rng = np.random.default_rng(7)
    n = 40

    # Ground truth: clean join result
    users = pd.DataFrame({
        "user_id":  [f"U{i:03d}" for i in range(n)],
        "name":     [f"User_{i}" for i in range(n)],
        "tier":     rng.choice(["free", "pro", "enterprise"], n),
    })
    orders = pd.DataFrame({
        "user_id":  [f"U{i % n:03d}" for i in range(n * 2)],
        "amount":   rng.uniform(10, 500, n * 2).round(2),
        "status":   rng.choice(["paid", "pending", "refunded"], n * 2),
    })
    gt = orders.merge(users, on="user_id", how="left")[
        ["user_id", "name", "tier", "amount", "status"]
    ].reset_index(drop=True)

    # Broken: pipeline used camelCase 'userId' key — join produced all-NULL
    # name/tier columns because the key never matched. The 'userId' column is
    # the visible artifact the agent must rename to 'user_id' to fix the join.
    broken = orders.copy()
    broken["name"] = pd.NA
    broken["tier"] = pd.NA
    broken.insert(0, "userId", broken["user_id"])
    broken = broken.drop(columns=["user_id"])
    broken = broken[["userId", "amount", "status", "name", "tier"]]

    return TaskFixture(
        task_id="medium",
        description=(
            "A left-join between 'orders' and 'users' failed because the key "
            "column is named 'user_id' in orders but 'userId' in users. "
            "The result has NaN-filled 'name' and 'tier' columns. "
            "Rename the column so the join key matches, then re-execute."
        ),
        broken_df=broken,
        ground_truth_df=gt,
        hint="Use rename_column on the 'userId' column to 'user_id', then reorder_columns.",
    )


# ---------------------------------------------------------------------------
# TASK: hard — silent data corruption (off-by-one date filter drops 8% rows)
# No exception, no stack trace — agent must detect statistical anomaly
# ---------------------------------------------------------------------------

def _make_hard() -> TaskFixture:
    rng = np.random.default_rng(99)
    n = 200

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    amounts = rng.uniform(50, 2000, n).round(2)
    categories = rng.choice(["A", "B", "C", "D"], n)

    # Ground truth: all 200 rows, correct totals
    gt = pd.DataFrame({
        "date":       dates,
        "amount":     amounts,
        "category":   categories,
        "daily_total": amounts,          # simplified — same as amount here
    })
    gt["month"] = gt["date"].dt.month
    gt["year"]  = gt["date"].dt.year

    # Broken: date range filter uses > instead of >= on Jan 1st
    # Drops the first row + last row → 198 rows, totals are wrong
    # The bug is silent — no error, output looks plausible
    mask = (gt["date"] > pd.Timestamp("2024-01-01")) & \
           (gt["date"] < pd.Timestamp("2024-07-18"))
    broken = gt[mask].copy().reset_index(drop=True)
    # Introduce aggregation error: daily_total is summed wrong (missing category D)
    broken.loc[broken["category"] == "D", "daily_total"] = 0.0

    return TaskFixture(
        task_id="hard",
        description=(
            "A date-range filter used strict '>' instead of '>=' on the start "
            "date, silently dropping rows. Additionally, category 'D' totals "
            "were zeroed out in the aggregation. There is no exception or stack "
            "trace — you must audit row counts and statistical distributions to "
            "find the corruption."
        ),
        broken_df=broken,
        ground_truth_df=gt,
        hint=(
            "Check row count vs expected. Inspect daily_total for category D. "
            "Use drop_rows to remove the corrupt filter, then fix_column on daily_total."
        ),
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

_FIXTURES: dict[str, TaskFixture] | None = None


def load_task(task_id: str) -> TaskFixture:
    global _FIXTURES
    if _FIXTURES is None:
        _FIXTURES = {
            "easy":   _make_easy(),
            "medium": _make_medium(),
            "hard":   _make_hard(),
        }
    if task_id not in _FIXTURES:
        raise ValueError(f"Unknown task_id '{task_id}'. Choose from: easy, medium, hard")
    return _FIXTURES[task_id]