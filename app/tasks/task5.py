"""Task 5 — Cascading Pipeline Failure

A 4-stage pipeline: ingest → join → aggregate → features.
- Stage 2's join introduces nulls in revenue (left join instead of inner)
- Stage 3's aggregation totals are corrupted by those nulls
- Stage 4 double-counts customers (COUNT(*) instead of COUNT(DISTINCT))
- The agent must trace backwards to the root cause — Stage 2

4 stages. Hard difficulty. 40 max steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


TASK_INFO = {
    "id": 5,
    "name": "Cascading Pipeline Failure",
    "difficulty": "hard",
    "description": (
        "A 4-stage pipeline: ingest → join → aggregate → features. Stage 2's "
        "left join introduced nulls in revenue. Those nulls corrupt Stage 3's "
        "aggregation totals. Stage 4 double-counts customers. The agent must trace "
        "backwards to the root cause — Stage 2 — and fix each stage."
    ),
    "max_steps": 40,
    "hint": "Start by inspecting each stage. The root cause is Stage 2 (left join → nulls). Fix Stage 2 first, then fix Stage 4's double-counting.",
}

PROTECTED_COLUMNS = [
    "region", "total_revenue", "unique_customers", "avg_order_value",
]


class PipelineStages:
    """Holds the 4 pipeline stage DataFrames."""

    def __init__(self):
        self.stages: Dict[int, pd.DataFrame] = {}
        self.root_cause_identified: bool = False

    def get_stage(self, stage_id: int) -> Optional[pd.DataFrame]:
        return self.stages.get(stage_id)


def generate_data(seed: int = 42) -> pd.DataFrame:
    """Generate the full cascading pipeline with bugs. Returns final DataFrame."""
    rng = np.random.RandomState(seed)

    # ── Stage 1: Ingest ─────────────────────────────────────────────────────
    n_orders = 4000
    order_id = np.arange(1, n_orders + 1)
    customer_id = rng.randint(1, 801, n_orders)  # 800 unique customers
    region = rng.choice(["North", "South", "East", "West"], n_orders, p=[0.3, 0.25, 0.25, 0.2])
    order_amount = rng.uniform(10, 500, n_orders).round(2)

    stage1 = pd.DataFrame({
        "order_id": order_id,
        "customer_id": customer_id,
        "region": region,
        "order_amount": order_amount,
    })

    # ── Stage 2: Join with customer revenue (LEFT JOIN — BUG!) ──────────────
    # Only 700 of 800 customers have revenue data (introduces nulls)
    unique_customers = np.arange(1, 801)
    customers_with_revenue = rng.choice(unique_customers, size=700, replace=False)
    revenue_data = pd.DataFrame({
        "customer_id": customers_with_revenue,
        "annual_revenue": rng.uniform(1000, 50000, 700).round(2),
    })

    # BUG: Left join instead of inner join → nulls in annual_revenue
    stage2 = stage1.merge(revenue_data, on="customer_id", how="left")

    # ── Stage 3: Aggregate by region ────────────────────────────────────────
    # Nulls from stage 2 silently corrupt the sum (NaN propagation)
    stage3 = stage2.groupby("region").agg(
        total_revenue=("annual_revenue", "sum"),       # NaN-corrupted
        total_orders=("order_id", "count"),
        avg_order_value=("order_amount", "mean"),
    ).reset_index()

    # ── Stage 4: Feature engineering ────────────────────────────────────────
    # BUG: COUNT(*) instead of COUNT(DISTINCT customer_id) → double counts
    customer_counts = stage2.groupby("region")["customer_id"].count().reset_index()
    customer_counts.columns = ["region", "unique_customers"]  # Misleading name!

    stage4 = stage3.merge(customer_counts, on="region")

    # Store stages for inspection
    stage4.attrs["_pipeline_stages"] = {
        1: stage1.copy(),
        2: stage2.copy(),
        3: stage3.copy(),
        4: stage4.copy(),
    }
    stage4.attrs["_root_cause_identified"] = False

    return stage4


def get_stages(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Extract pipeline stages from the DataFrame's attrs."""
    return df.attrs.get("_pipeline_stages", {})


def set_stages(df: pd.DataFrame, stages: Dict[int, pd.DataFrame]):
    """Update pipeline stages in the DataFrame's attrs."""
    df.attrs["_pipeline_stages"] = stages


def get_stage_info(df: pd.DataFrame, stage_id: int) -> str:
    """Return a human-readable summary of a pipeline stage."""
    stages = get_stages(df)
    if stage_id not in stages:
        return f"Stage {stage_id} not found. Valid stages: 1, 2, 3, 4."

    stage_df = stages[stage_id]
    null_info = stage_df.isnull().sum()
    null_cols = {c: int(v) for c, v in null_info.items() if v > 0}

    info = (
        f"Stage {stage_id}: shape={stage_df.shape}, "
        f"columns={list(stage_df.columns)}, "
        f"dtypes={dict(stage_df.dtypes.astype(str))}, "
        f"null_counts={null_cols if null_cols else 'none'}, "
        f"sample=\n{stage_df.head(3).to_string()}"
    )
    return info


def apply_stage_fix(df: pd.DataFrame, stage_id: int, fix_type: str, **kwargs) -> str:
    """Apply a fix to a specific pipeline stage and re-run downstream stages."""
    stages = get_stages(df)
    if stage_id not in stages:
        return f"Invalid stage_id: {stage_id}"

    if stage_id == 2 and fix_type == "fix_join":
        # Fix: convert left join to inner join (drop nulls)
        stage2 = stages[2]
        stage2 = stage2.dropna(subset=["annual_revenue"])
        stages[2] = stage2
        df.attrs["_root_cause_identified"] = True

        # Re-run Stage 3
        stage3 = stage2.groupby("region").agg(
            total_revenue=("annual_revenue", "sum"),
            total_orders=("order_id", "count"),
            avg_order_value=("order_amount", "mean"),
        ).reset_index()
        stages[3] = stage3

        # Re-run Stage 4 (but keep the double-count bug if not fixed)
        customer_counts = stage2.groupby("region")["customer_id"].count().reset_index()
        customer_counts.columns = ["region", "unique_customers"]
        stage4 = stage3.merge(customer_counts, on="region")
        stages[4] = stage4

        # Update main df
        _update_main_df(df, stage4)
        set_stages(df, stages)
        df.attrs["_root_cause_identified"] = True

        return "Fixed Stage 2: converted left join to inner join. Dropped rows with null annual_revenue. Re-ran stages 3 and 4."

    elif stage_id == 4 and fix_type == "fix_double_count":
        # Fix: use nunique instead of count for customer_id
        stage2 = stages[2]
        customer_counts = stage2.groupby("region")["customer_id"].nunique().reset_index()
        customer_counts.columns = ["region", "unique_customers"]

        stage3 = stages[3]
        stage4 = stage3.merge(customer_counts, on="region", how="left",
                               suffixes=("_old", ""))
        # Drop old unique_customers if exists
        if "unique_customers_old" in stage4.columns:
            stage4 = stage4.drop(columns=["unique_customers_old"])
        stages[4] = stage4

        _update_main_df(df, stage4)
        set_stages(df, stages)

        return "Fixed Stage 4: replaced COUNT(*) with COUNT(DISTINCT customer_id) for unique_customers."

    elif stage_id == 3 and fix_type == "fix_aggregation":
        # Re-aggregate from current stage 2 data (skipping NaN)
        stage2 = stages[2]
        stage3 = stage2.groupby("region").agg(
            total_revenue=("annual_revenue", lambda x: x.dropna().sum()),
            total_orders=("order_id", "count"),
            avg_order_value=("order_amount", "mean"),
        ).reset_index()
        stages[3] = stage3

        # Re-run stage 4
        customer_counts = stage2.groupby("region")["customer_id"].count().reset_index()
        customer_counts.columns = ["region", "unique_customers"]
        stage4 = stage3.merge(customer_counts, on="region")
        if "unique_customers_old" in stage4.columns:
            stage4 = stage4.drop(columns=["unique_customers_old"])
        stages[4] = stage4

        _update_main_df(df, stage4)
        set_stages(df, stages)

        return "Fixed Stage 3: re-aggregated with NaN-safe sum. Re-ran stage 4."

    return f"Unknown fix_type '{fix_type}' for stage {stage_id}. Available: fix_join (stage 2), fix_aggregation (stage 3), fix_double_count (stage 4)."


def _update_main_df(df: pd.DataFrame, new_data: pd.DataFrame):
    """Update the main DataFrame in-place from new stage 4 output."""
    # Clear and rebuild columns
    for col in list(df.columns):
        if col not in new_data.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    # We need to resize if row counts differ
    if len(df) != len(new_data):
        # Can't truly resize in place — copy attrs, replace
        attrs_backup = dict(df.attrs)
        df.drop(df.index, inplace=True)
        for col in new_data.columns:
            df[col] = None
        for i, row in new_data.iterrows():
            df.loc[i] = row
        df.attrs.update(attrs_backup)
    else:
        for col in new_data.columns:
            df[col] = new_data[col].values


def grade(df: pd.DataFrame) -> List[bool]:
    """Return 5 sub-test booleans.

    Tests:
      1. No null values in total_revenue (Stage 2 join fixed)
      2. total_revenue values are positive and reasonable (Stage 3 aggregation valid)
      3. unique_customers <= total customers per region (no double-counting)
      4. Root cause correctly identified as Stage 2
      5. All required output columns present
    """
    results: List[bool] = []

    # Test 1: No nulls in total_revenue
    try:
        results.append(bool(
            "total_revenue" in df.columns and df["total_revenue"].notnull().all()
        ))
    except Exception:
        results.append(False)

    # Test 2: total_revenue is positive and reasonable
    try:
        col = df["total_revenue"]
        all_positive = bool((col > 0).all())
        # Reasonable range: each region should have some revenue
        results.append(bool(all_positive and len(col) >= 3))
    except Exception:
        results.append(False)

    # Test 3: unique_customers is reasonable (not double-counted)
    try:
        if "unique_customers" in df.columns:
            # After fix, unique_customers should be < total_orders per region
            uc = df["unique_customers"]
            to = df["total_orders"]
            # Unique customers should be less than total orders (many customers order multiple times)
            reasonable = bool((uc <= to).all() and (uc < to).any())
            results.append(reasonable)
        else:
            results.append(False)
    except Exception:
        results.append(False)

    # Test 4: Root cause identified as Stage 2
    try:
        root_cause = df.attrs.get("_root_cause_identified", False)
        results.append(root_cause is True)
    except Exception:
        results.append(False)

    # Test 5: All required columns present
    try:
        required = ["region", "total_revenue", "unique_customers", "avg_order_value"]
        results.append(bool(all(c in df.columns for c in required)))
    except Exception:
        results.append(False)

    return results
