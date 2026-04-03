"""Task 1 — Type Corruption

Numeric values stored as malformed strings.
- interest_rate: "8.5%" instead of 8.5
- annual_income: "$85,000" instead of 85000.0
- loan_amount stored as strings with commas
- credit_score stored as strings
- Null values as literal string "missing" instead of NaN

2000 rows. Easy difficulty.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


TASK_INFO = {
    "id": 1,
    "name": "Type Corruption",
    "difficulty": "easy",
    "description": (
        "Numeric values are stored as malformed strings. interest_rate is '8.5%' "
        "not 8.5. annual_income is '$85,000' not 85000.0. Null values are the "
        "literal string 'missing' instead of NaN. The model cannot do math on "
        "strings — accuracy collapses."
    ),
    "max_steps": 30,
    "hint": "Strip currency/percent characters, cast columns to float64, and replace 'missing' with NaN.",
}

# Columns that must NOT be dropped (used for over-fix penalty)
PROTECTED_COLUMNS = [
    "loan_id", "interest_rate", "annual_income", "loan_amount",
    "credit_score", "employment_years", "approved",
]


def generate_data(seed: int = 42) -> pd.DataFrame:
    """Generate a 2000-row loan-application DataFrame with type-corruption bugs."""
    rng = np.random.RandomState(seed)
    n = 2000

    loan_id = np.arange(1, n + 1)
    interest_rate_raw = rng.uniform(3.0, 18.0, n).round(2)
    annual_income_raw = rng.uniform(25000, 200000, n).round(0)
    loan_amount_raw = rng.uniform(5000, 500000, n).round(0)
    credit_score_raw = rng.randint(300, 850, n)
    employment_years = rng.randint(0, 40, n).astype(float)
    approved = rng.choice([0, 1], n, p=[0.4, 0.6])

    # ── Plant bugs ──────────────────────────────────────────────────────────

    # Bug 1: interest_rate as "8.5%" strings
    interest_rate = [f"{v}%" for v in interest_rate_raw]

    # Bug 2: annual_income as "$85,000" strings
    annual_income = [f"${v:,.0f}" for v in annual_income_raw]

    # Bug 3: loan_amount as strings with commas "250,000"
    loan_amount = [f"{v:,.0f}" for v in loan_amount_raw]

    # Bug 4: credit_score as plain strings
    credit_score = [str(v) for v in credit_score_raw]

    # Bug 5: Sprinkle "missing" as null sentinel in several columns
    missing_mask_interest = rng.choice(n, size=80, replace=False)
    for idx in missing_mask_interest:
        interest_rate[idx] = "missing"

    missing_mask_income = rng.choice(n, size=60, replace=False)
    for idx in missing_mask_income:
        annual_income[idx] = "missing"

    missing_mask_credit = rng.choice(n, size=50, replace=False)
    for idx in missing_mask_credit:
        credit_score[idx] = "missing"

    df = pd.DataFrame({
        "loan_id": loan_id,
        "interest_rate": interest_rate,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "approved": approved,
    })

    return df


def grade(df: pd.DataFrame) -> List[bool]:
    """Return 5 sub-test booleans (each worth 0.2 reward).

    Tests:
      1. interest_rate is float64 and contains no '%' characters
      2. annual_income is float64 and contains no '$' or ',' characters
      3. loan_amount is numeric (int64 or float64) with no commas
      4. credit_score is numeric (int64 or float64)
      5. No literal 'missing' strings remain in any column
    """
    results: List[bool] = []

    # Test 1: interest_rate
    try:
        col = df["interest_rate"]
        is_float = col.dtype == np.float64
        no_pct = not col.dropna().astype(str).str.contains("%").any()
        results.append(is_float and no_pct)
    except Exception:
        results.append(False)

    # Test 2: annual_income
    try:
        col = df["annual_income"]
        is_float = col.dtype == np.float64
        no_dollar = not col.dropna().astype(str).str.contains(r"[\$,]", regex=True).any()
        results.append(is_float and no_dollar)
    except Exception:
        results.append(False)

    # Test 3: loan_amount is numeric
    try:
        col = df["loan_amount"]
        is_numeric = col.dtype in (np.float64, np.int64)
        no_comma = not col.dropna().astype(str).str.contains(",").any()
        results.append(is_numeric and no_comma)
    except Exception:
        results.append(False)

    # Test 4: credit_score is numeric
    try:
        col = df["credit_score"]
        is_numeric = col.dtype in (np.float64, np.int64)
        results.append(is_numeric)
    except Exception:
        results.append(False)

    # Test 5: No literal "missing" anywhere
    try:
        has_missing = False
        for c in df.columns:
            if df[c].dtype == object:
                if df[c].astype(str).str.lower().eq("missing").any():
                    has_missing = True
                    break
            else:
                # If numeric, check if any NaN were not converted
                pass
        results.append(not has_missing)
    except Exception:
        results.append(False)

    return results
