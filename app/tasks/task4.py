"""Task 4 — Target Leakage

The model has 99% training accuracy but 52% on real data. The dataset includes
columns that leak the target variable:
- days_to_churn: derived from churn date (future data)
- churn_reason_code: only exists after the event
- cancellation_flag: literally the target under a new name

The agent must surgically remove these WITHOUT touching the 5 legitimate features.

4000 rows. Hard difficulty.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


TASK_INFO = {
    "id": 4,
    "name": "Target Leakage",
    "difficulty": "hard",
    "description": (
        "The model has 99% training accuracy but 52% on real data. The dataset "
        "includes days_to_churn (derived from future churn date), churn_reason_code "
        "(only exists after the event), and cancellation_flag (literally the target "
        "under a new name). The agent must identify and remove these leaky columns "
        "without touching the five legitimate feature columns."
    ),
    "max_steps": 30,
    "hint": "Drop days_to_churn, churn_reason_code, and cancellation_flag — they all leak the target 'churned'. Do NOT drop tenure_months, monthly_charges, total_charges, contract_type, or support_tickets.",
}

# Columns that must NOT be dropped
PROTECTED_COLUMNS = [
    "customer_id", "tenure_months", "monthly_charges", "total_charges",
    "contract_type", "support_tickets", "churned",
]

# Columns that MUST be dropped (leaky)
LEAKY_COLUMNS = [
    "days_to_churn", "churn_reason_code", "cancellation_flag",
]


def generate_data(seed: int = 42) -> pd.DataFrame:
    """Generate a 4000-row churn DataFrame with target leakage bugs."""
    rng = np.random.RandomState(seed)
    n = 4000

    customer_id = np.arange(1, n + 1)
    tenure_months = rng.randint(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n).round(2)
    total_charges = (tenure_months * monthly_charges + rng.normal(0, 50, n)).round(2)
    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.5, 0.3, 0.2])
    support_tickets = rng.randint(0, 10, n)
    churned = rng.choice([0, 1], n, p=[0.73, 0.27])

    # ── Leaky columns ──────────────────────────────────────────────────────

    # Leak 1: days_to_churn — directly derived from the target
    days_to_churn = np.where(churned == 1, rng.randint(1, 60, n), 0).astype(float)

    # Leak 2: churn_reason_code — only present for churned customers
    reasons = ["price", "service", "competitor", "relocation", "other"]
    churn_reason_code = np.where(
        churned == 1,
        rng.choice(reasons, n),
        "none"
    )

    # Leak 3: cancellation_flag — literally the target renamed
    cancellation_flag = churned.copy()

    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "support_tickets": support_tickets,
        "days_to_churn": days_to_churn,            # LEAK
        "churn_reason_code": churn_reason_code,     # LEAK
        "cancellation_flag": cancellation_flag,     # LEAK
        "churned": churned,                         # TARGET
    })

    return df


def grade(df: pd.DataFrame) -> List[bool]:
    """Return 5 sub-test booleans.

    Tests:
      1. days_to_churn column has been removed
      2. churn_reason_code column has been removed
      3. cancellation_flag column has been removed
      4. All 5 legitimate feature columns still exist
      5. Target column 'churned' still exists and is intact
    """
    results: List[bool] = []

    # Test 1: days_to_churn removed
    results.append("days_to_churn" not in df.columns)

    # Test 2: churn_reason_code removed
    results.append("churn_reason_code" not in df.columns)

    # Test 3: cancellation_flag removed
    results.append("cancellation_flag" not in df.columns)

    # Test 4: Legitimate feature columns still present
    try:
        legit = ["tenure_months", "monthly_charges", "total_charges",
                 "contract_type", "support_tickets"]
        results.append(all(c in df.columns for c in legit))
    except Exception:
        results.append(False)

    # Test 5: Target column exists and is binary
    try:
        has_target = "churned" in df.columns
        if has_target:
            unique_vals = set(df["churned"].dropna().unique())
            is_binary = unique_vals.issubset({0, 1, 0.0, 1.0})
            results.append(is_binary)
        else:
            results.append(False)
    except Exception:
        results.append(False)

    return results
