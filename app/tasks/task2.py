"""Task 2 — Feature Encoding Bug

Problems:
- city column with 20 unique cities label-encoded as integers 1–20 (false ordinal)
- age and monthly_spending at wildly different scales (25 vs 50,000)
- signup_date column as a string, never converted to numeric

3000 rows. Easy-Medium difficulty.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


TASK_INFO = {
    "id": 2,
    "name": "Feature Encoding Bug",
    "difficulty": "easy-medium",
    "description": (
        "City is label-encoded as integers 1-20 (false ordinal ordering for nominal "
        "data). Age and monthly_spending are at wildly different scales. A date column "
        "was never converted to numeric. The agent must fix the encoding, normalize, "
        "and convert dates."
    ),
    "max_steps": 30,
    "hint": "One-hot encode or drop the city column, z-score normalize numeric columns, and convert signup_date to numeric.",
}

PROTECTED_COLUMNS = [
    "customer_id", "age", "monthly_spending", "signup_date",
    "num_purchases", "churned",
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "Indianapolis", "San Francisco", "Seattle", "Denver", "Nashville",
]


def generate_data(seed: int = 42) -> pd.DataFrame:
    """Generate a 3000-row customer DataFrame with encoding bugs."""
    rng = np.random.RandomState(seed)
    n = 3000

    customer_id = np.arange(1, n + 1)

    # Bug 1: city as ordinal integer 1–20 (should be nominal)
    city_encoded = rng.randint(1, 21, n)  # 1-based label encoding

    # Clean columns
    age = rng.randint(18, 75, n).astype(float)
    monthly_spending = rng.uniform(100, 80000, n).round(2)
    num_purchases = rng.randint(0, 200, n)
    churned = rng.choice([0, 1], n, p=[0.7, 0.3])

    # Bug 3: signup_date as string, not converted to numeric
    base_date = pd.Timestamp("2020-01-01")
    random_days = rng.randint(0, 1500, n)
    signup_dates = [(base_date + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in random_days]

    df = pd.DataFrame({
        "customer_id": customer_id,
        "city": city_encoded,                  # Bug 1: ordinal encoding of nominal
        "age": age,                            # Bug 2a: un-normalized
        "monthly_spending": monthly_spending,  # Bug 2b: un-normalized (different scale)
        "signup_date": signup_dates,           # Bug 3: string date
        "num_purchases": num_purchases,
        "churned": churned,
    })

    return df


def grade(df: pd.DataFrame) -> List[bool]:
    """Return 5 sub-test booleans.

    Tests:
      1. city is NOT an integer column (one-hot encoded or dropped)
      2. age is approximately z-score normalized (mean≈0, std≈1)
      3. monthly_spending is approximately z-score normalized (mean≈0, std≈1)
      4. signup_date has been converted to numeric (int64 or float64)
      5. Protected columns (customer_id, num_purchases, churned) still exist
    """
    results: List[bool] = []

    # Test 1: city is not ordinal integer anymore
    try:
        if "city" not in df.columns:
            # Dropped — acceptable
            results.append(True)
        else:
            # Should not be a single integer column
            is_int = df["city"].dtype in (np.int64, np.int32, np.float64)
            n_unique = df["city"].nunique()
            # If it's still a single column with ~20 integer values, it's still broken
            results.append(bool(not (is_int and n_unique <= 25)))
    except Exception:
        results.append(False)

    # Test 2: age is z-score normalized
    try:
        col = df["age"].astype(float)
        mean_ok = abs(col.mean()) < 0.5
        std_ok = abs(col.std() - 1.0) < 0.5
        results.append(bool(mean_ok and std_ok))
    except Exception:
        results.append(False)

    # Test 3: monthly_spending is z-score normalized
    try:
        col = df["monthly_spending"].astype(float)
        mean_ok = abs(col.mean()) < 0.5
        std_ok = abs(col.std() - 1.0) < 0.5
        results.append(bool(mean_ok and std_ok))
    except Exception:
        results.append(False)

    # Test 4: signup_date is numeric
    try:
        if "signup_date" in df.columns:
            col = df["signup_date"]
            is_numeric = col.dtype in (np.int64, np.float64)
            results.append(bool(is_numeric))
        else:
            results.append(False)
    except Exception:
        results.append(False)

    # Test 5: Protected columns still exist
    try:
        required = ["customer_id", "num_purchases", "churned"]
        all_present = all(c in df.columns for c in required)
        results.append(all_present)
    except Exception:
        results.append(False)

    return results
