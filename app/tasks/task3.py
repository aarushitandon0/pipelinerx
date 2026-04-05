"""Task 3 — Schema Drift

The model was trained on last quarter's schema. Since then:
- user_id was renamed to customer_id
- A new tier "enterprise" appeared that the model never saw
- Timestamps shifted from UTC to IST (+5.5h)
- Some rows silently dropped (3000 → ~2700)

3000 rows (drifted). Medium difficulty.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


TASK_INFO = {
    "id": 3,
    "name": "Schema Drift",
    "difficulty": "medium",
    "description": (
        "The model was trained on last quarter's schema. user_id was renamed to "
        "customer_id. A new tier 'enterprise' appeared. Timestamps shifted from UTC "
        "to IST (+5.5h). Some rows silently dropped. The agent must detect each drift "
        "and patch them."
    ),
    "max_steps": 30,
    "hint": "Rename customer_id back to user_id, filter out 'enterprise' rows, convert timestamps back from IST to UTC (subtract 5.5 hours).",
}

PROTECTED_COLUMNS = [
    "user_id", "tier", "event_timestamp", "revenue", "sessions", "converted",
]


def generate_data(seed: int = 42) -> pd.DataFrame:
    """Generate a 3000-row event DataFrame with schema-drift bugs."""
    rng = np.random.RandomState(seed)
    n = 3000

    user_id = np.arange(1, n + 1)
    tiers_original = ["free", "basic", "premium"]
    tier = rng.choice(tiers_original, n, p=[0.5, 0.3, 0.2]).tolist()

    base_ts = pd.Timestamp("2025-01-01", tz="UTC")
    random_hours = rng.randint(0, 8760, n)
    event_timestamps = [base_ts + pd.Timedelta(hours=int(h)) for h in random_hours]

    revenue = rng.uniform(0, 500, n).round(2)
    sessions = rng.randint(1, 100, n)
    converted = rng.choice([0, 1], n, p=[0.6, 0.4])

    df = pd.DataFrame({
        "user_id": user_id,
        "tier": tier,
        "event_timestamp": event_timestamps,
        "revenue": revenue,
        "sessions": sessions,
        "converted": converted,
    })

    # ── Plant bugs ──────────────────────────────────────────────────────────

    # Bug 1: Rename user_id → customer_id
    df = df.rename(columns={"user_id": "customer_id"})

    # Bug 2: Inject "enterprise" tier in ~10% of rows
    enterprise_mask = rng.choice(n, size=int(n * 0.1), replace=False)
    for idx in enterprise_mask:
        df.at[idx, "tier"] = "enterprise"

    # Bug 3: Shift timestamps from UTC to IST (+5.5h) — the data now claims IST
    df["event_timestamp"] = df["event_timestamp"] + pd.Timedelta(hours=5.5)
    # Remove timezone info to simulate "someone just shifted the numbers"
    df["event_timestamp"] = df["event_timestamp"].dt.tz_localize(None)

    # Bug 4: Drop ~10% of rows silently
    drop_mask = rng.choice(n, size=int(n * 0.1), replace=False)
    df = df.drop(index=drop_mask).reset_index(drop=True)

    return df


def grade(df: pd.DataFrame) -> List[bool]:
    """Return 5 sub-test booleans.

    Tests:
      1. Column is named 'user_id' (not 'customer_id')
      2. No 'enterprise' values in tier column
      3. Timestamps are back in UTC range (shifted back by 5.5h)
      4. Row count is roughly 2700 (enterprise rows filtered, but no extra drops)
      5. Core columns (revenue, sessions, converted) still exist and are numeric
    """
    results: List[bool] = []

    # Test 1: user_id column exists
    try:
        has_user_id = "user_id" in df.columns
        no_customer_id = "customer_id" not in df.columns
        results.append(has_user_id and no_customer_id)
    except Exception:
        results.append(False)

    # Test 2: No "enterprise" in tier
    try:
        col = df["tier"]
        no_enterprise = not col.astype(str).str.lower().eq("enterprise").any()
        results.append(no_enterprise)
    except Exception:
        results.append(False)

    # Test 3: Timestamps shifted back (check that mean timestamp is in UTC-ish range)
    try:
        col = pd.to_datetime(df["event_timestamp"])
        mean_ts = col.mean()
        # Original data centered around mid-2025 in UTC.
        # After filtering enterprise rows the mean shifts slightly.
        expected_center = pd.Timestamp("2025-07-01")
        diff_hours = abs((mean_ts - expected_center).total_seconds()) / 3600
        results.append(diff_hours < 48)  # Within 48 hours of expected center
    except Exception:
        results.append(False)

    # Test 4: Row count is in the right range (enterprise filtered)
    try:
        # After removing enterprise (~10%) from the already-dropped (~10%) data:
        # Original 3000, dropped 10% → ~2700, then enterprise ~10% of those removed
        # Expected: ~2400-2500
        row_count = len(df)
        results.append(2200 <= row_count <= 2600)
    except Exception:
        results.append(False)

    # Test 5: Core columns still exist and numeric
    try:
        required = {"revenue": np.floating, "sessions": np.integer, "converted": np.integer}
        all_ok = True
        for col_name in required:
            if col_name not in df.columns:
                all_ok = False
                break
            if not np.issubdtype(df[col_name].dtype, np.number):
                all_ok = False
                break
        results.append(all_ok)
    except Exception:
        results.append(False)

    return results
