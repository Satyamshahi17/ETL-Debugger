"""
server/actions.py
Applies a PipelineAction to a DataFrame, returning the mutated copy.
Exceptions are caught by the Environment and written into the error_log.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from models import PipelineAction


def apply_action(df: pd.DataFrame, action: PipelineAction) -> pd.DataFrame:
    """
    Dispatch to the correct handler and return a *new* DataFrame.
    Raises ValueError / KeyError on bad inputs — caller catches and logs.
    """
    df = df.copy()
    t  = action.action_type

    if t == "done":
        return df  # No-op; episode ends in environment.step()

    if t == "fix_column":
        return _fix_column(df, action.column, action.params)

    if t == "cast_type":
        return _cast_type(df, action.column, action.params)

    if t == "drop_rows":
        return _drop_rows(df, action.column, action.params)

    if t == "rename_column":
        return _rename_column(df, action.column, action.params)

    if t == "fill_nulls":
        return _fill_nulls(df, action.column, action.params)

    if t == "split_column":
        return _split_column(df, action.column, action.params)

    if t == "merge_columns":
        return _merge_columns(df, action.column, action.params)

    if t == "reorder_columns":
        return _reorder_columns(df, action.params)

    raise ValueError(f"Unknown action_type: '{t}'")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _require_col(df: pd.DataFrame, col: str | None) -> str:
    if col is None:
        raise ValueError("'column' is required for this action.")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")
    return col


def _fix_column(df: pd.DataFrame, col: str | None, params: dict) -> pd.DataFrame:
    """
    General-purpose string transform on a column.
    params["transform"] options:
        "strip_currency"  → remove $, commas, whitespace
        "strip_whitespace"→ strip leading/trailing whitespace
        "lowercase"       → str.lower()
        "uppercase"       → str.upper()
    """
    col = _require_col(df, col)
    transform = params.get("transform", "strip_whitespace")

    if transform == "strip_currency":
        df[col] = df[col].astype(str).str.replace(r"[\$,\s]", "", regex=True)
    elif transform == "strip_whitespace":
        df[col] = df[col].astype(str).str.strip()
    elif transform == "lowercase":
        df[col] = df[col].astype(str).str.lower()
    elif transform == "uppercase":
        df[col] = df[col].astype(str).str.upper()
    else:
        raise ValueError(f"Unknown transform '{transform}'")
    return df


def _cast_type(df: pd.DataFrame, col: str | None, params: dict) -> pd.DataFrame:
    """
    Cast column to a target dtype.
    params["dtype"]: "float64" | "int64" | "str" | "bool" | "datetime"
    """
    col   = _require_col(df, col)
    dtype = params.get("dtype", "float64")

    if dtype in ("float64", "float"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
    elif dtype in ("int64", "int"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
    elif dtype in ("str", "string", "object"):
        df[col] = df[col].astype(str)
    elif dtype == "bool":
        df[col] = df[col].astype(bool)
    elif dtype in ("datetime", "datetime64"):
        df[col] = pd.to_datetime(df[col], errors="raise")
    else:
        raise ValueError(f"Unsupported dtype '{dtype}'")
    return df


def _drop_rows(df: pd.DataFrame, col: str | None, params: dict) -> pd.DataFrame:
    """
    Drop rows based on a condition string or null check.
    params["condition"]: pandas query string, e.g. "revenue < 0"
    params["drop_nulls"]: bool — drop rows where `col` is null
    """
    if params.get("drop_nulls") and col:
        col = _require_col(df, col)
        return df.dropna(subset=[col]).reset_index(drop=True)

    condition = params.get("condition")
    if condition:
        try:
            mask = df.eval(condition)
            return df[~mask].reset_index(drop=True)
        except Exception as e:
            raise ValueError(f"Invalid condition '{condition}': {e}")

    raise ValueError("drop_rows requires 'condition' or 'drop_nulls' param.")


def _rename_column(df: pd.DataFrame, col: str | None, params: dict) -> pd.DataFrame:
    """
    Rename a column.
    params["new_name"]: target name string
    """
    col      = _require_col(df, col)
    new_name = params.get("new_name")
    if not new_name:
        raise ValueError("rename_column requires params['new_name'].")
    return df.rename(columns={col: new_name})


def _fill_nulls(df: pd.DataFrame, col: str | None, params: dict) -> pd.DataFrame:
    """
    Fill NaN values in a column.
    params["strategy"]: "mean" | "median" | "zero" | "forward" | "backward"
    params["value"]   : literal fill value (overrides strategy)
    """
    col      = _require_col(df, col)
    strategy = params.get("strategy", "zero")
    value    = params.get("value")

    if value is not None:
        df[col] = df[col].fillna(value)
    elif strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "zero":
        df[col] = df[col].fillna(0)
    elif strategy == "forward":
        df[col] = df[col].ffill()
    elif strategy == "backward":
        df[col] = df[col].bfill()
    else:
        raise ValueError(f"Unknown fill strategy '{strategy}'")
    return df


def _split_column(df: pd.DataFrame, col: str | None, params: dict) -> pd.DataFrame:
    """
    Split a string column into 2+ new columns and optionally drop the original.
    params["delimiter"] : split character, default "_"
    params["new_cols"]  : list of new column names
    params["keep_original"]: bool, default False
    """
    col       = _require_col(df, col)
    delim     = params.get("delimiter", "_")
    new_cols  = params.get("new_cols", [f"{col}_0", f"{col}_1"])
    keep_orig = params.get("keep_original", False)

    splits = df[col].astype(str).str.split(delim, expand=True)
    for i, name in enumerate(new_cols):
        if i < splits.shape[1]:
            df[name] = splits[i]
        else:
            df[name] = None

    if not keep_orig:
        df = df.drop(columns=[col])
    return df


def _merge_columns(df: pd.DataFrame, col: str | None, params: dict) -> pd.DataFrame:
    """
    Concatenate multiple string columns into one.
    params["other_cols"]   : list of additional column names to merge
    params["separator"]    : string separator, default " "
    params["new_col_name"] : name for merged column, default "merged"
    """
    col         = _require_col(df, col)
    other_cols  = params.get("other_cols", [])
    separator   = params.get("separator", " ")
    new_name    = params.get("new_col_name", "merged")

    cols_to_merge = [col] + [_require_col(df, c) for c in other_cols]
    df[new_name] = df[cols_to_merge].apply(
        lambda row: separator.join(row.astype(str)), axis=1
    )
    return df


def _reorder_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Reorder (and optionally subset) columns.
    params["order"]: list of column names in desired order
    """
    order = params.get("order")
    if not order:
        raise ValueError("reorder_columns requires params['order'] list.")
    missing = [c for c in order if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return df[order]