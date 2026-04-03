"""Environment state machine — holds the DataFrame, routes actions, computes rewards."""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.tasks import GENERATORS, GRADERS, TASK_INFOS
from app.tasks.task1 import PROTECTED_COLUMNS as TASK1_PROTECTED
from app.tasks.task2 import PROTECTED_COLUMNS as TASK2_PROTECTED
from app.tasks.task3 import PROTECTED_COLUMNS as TASK3_PROTECTED
from app.tasks.task4 import PROTECTED_COLUMNS as TASK4_PROTECTED, LEAKY_COLUMNS
from app.tasks.task5 import PROTECTED_COLUMNS as TASK5_PROTECTED
from app.tasks.task5 import get_stage_info, apply_stage_fix, get_stages

PROTECTED = {
    1: TASK1_PROTECTED,
    2: TASK2_PROTECTED,
    3: TASK3_PROTECTED,
    4: TASK4_PROTECTED,
    5: TASK5_PROTECTED,
}


class EnvironmentState:
    """Thread-safe state container for one episode at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self.df: Optional[pd.DataFrame] = None
        self.task_id: int = 1
        self.step_count: int = 0
        self.done: bool = False
        self.reward: float = 0.0
        self.session_id: Optional[str] = None
        self.max_steps: int = 30
        self._original_columns: List[str] = []
        self._cached_tests_passed: int = 0

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset(self, task_id: int = 1, seed: int = 42) -> Dict[str, Any]:
        with self._lock:
            self.task_id = task_id
            self.df = GENERATORS[task_id](seed=seed)
            self.step_count = 0
            self.done = False
            self.reward = 0.0
            self.session_id = str(uuid.uuid4())
            self.max_steps = TASK_INFOS[task_id]["max_steps"]
            self._original_columns = list(self.df.columns)
            self._cached_tests_passed = 0

            observation = self._get_reset_observation()

            return {
                "task_id": self.task_id,
                "observation": observation,
                "info": {**self._info_dict(), "session_id": self.session_id},
            }

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self.df is None:
                return self._error_response("No episode active. Call /reset first.")

            if self.done:
                return self._error_response("Episode is done. Call /reset to start a new one.")

            # Session protection: if the agent passed a session_id, ensure it matches
            provided_sid = None
            if isinstance(parameters, dict):
                provided_sid = parameters.get("session_id")
            if provided_sid is not None and provided_sid != self.session_id:
                return self._error_response("Stale session_id: this step belongs to a previous episode.")

            self.step_count += 1

            # Enforce step limit
            if self.step_count >= self.max_steps:
                self.done = True

            try:
                observation = self._execute_action(action, parameters)
            except Exception as e:
                observation = f"Invalid action: {str(e)}"

            # Note: reward is updated inside _run_tests() when action == "run_tests".
            # No extra grader call needed here.

            return {
                "observation": observation,
                "reward": round(self.reward, 4),
                "done": self.done,
                "info": self._info_dict(),
            }

    # ── State ───────────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            if self.df is None:
                return {
                    "task_id": self.task_id,
                    "step_count": 0,
                    "reward": 0.0,
                    "df_shape": [0, 0],
                    "column_types": {},
                    "max_steps": 30,
                    "done": False,
                }

            return {
                "task_id": self.task_id,
                "step_count": self.step_count,
                "reward": round(self.reward, 4),
                "df_shape": list(self.df.shape),
                "column_types": {c: str(self.df[c].dtype) for c in self.df.columns},
                "max_steps": self.max_steps,
                "done": self.done,
            }

    # ── Action execution ────────────────────────────────────────────────────

    def _execute_action(self, action: str, params: Dict[str, Any]) -> str:
        if action == "inspect_data":
            return self._inspect_data(params)
        elif action == "inspect_schema":
            return self._inspect_schema()
        elif action == "apply_fix":
            return self._apply_fix(params)
        elif action == "run_tests":
            return self._run_tests()
        elif action == "check_column_stats":
            return self._check_column_stats(params)
        elif action == "inspect_stage":
            return self._inspect_stage(params)
        else:
            return f"Invalid action: '{action}'. Valid actions: inspect_data, inspect_schema, apply_fix, run_tests, check_column_stats, inspect_stage."

    def _inspect_data(self, params: Dict[str, Any]) -> str:
        n_rows = params.get("n_rows", 5)
        n_rows = min(n_rows, 20)
        sample = self.df.head(n_rows)
        dtypes = {c: str(self.df[c].dtype) for c in self.df.columns}
        return (
            f"DataFrame shape: {self.df.shape}\n"
            f"Column dtypes: {dtypes}\n"
            f"First {n_rows} rows:\n{sample.to_string()}"
        )

    def _inspect_schema(self) -> str:
        schema_info = []
        for col in self.df.columns:
            null_count = int(self.df[col].isnull().sum())
            unique_count = int(self.df[col].nunique())
            dtype = str(self.df[col].dtype)
            schema_info.append(
                f"  {col}: dtype={dtype}, nulls={null_count}, unique={unique_count}"
            )
        return (
            f"DataFrame shape: {self.df.shape}\n"
            f"Columns:\n" + "\n".join(schema_info)
        )

    def _apply_fix(self, params: Dict[str, Any]) -> str:
        operation = params.get("operation", "")
        column = params.get("column", "")

        if operation == "strip_and_cast":
            return self._op_strip_and_cast(params)
        elif operation == "replace_value":
            return self._op_replace_value(params)
        elif operation == "normalize_zscore":
            return self._op_normalize_zscore(column)
        elif operation == "normalize_minmax":
            return self._op_normalize_minmax(column)
        elif operation == "rename_column":
            return self._op_rename_column(params)
        elif operation == "drop_column":
            return self._op_drop_column(column)
        elif operation == "encode_onehot":
            return self._op_encode_onehot(column)
        elif operation == "cast_datetime_to_numeric":
            return self._op_cast_datetime_to_numeric(column)
        elif operation == "filter_values":
            return self._op_filter_values(params)
        elif operation == "convert_timezone":
            return self._op_convert_timezone(params)
        elif operation == "fix_stage":
            return self._op_fix_stage(params)
        else:
            return (
                f"Invalid operation: '{operation}'. Valid operations: "
                "strip_and_cast, replace_value, normalize_zscore, normalize_minmax, "
                "rename_column, drop_column, encode_onehot, cast_datetime_to_numeric, "
                "filter_values, convert_timezone, fix_stage."
            )

    def _op_strip_and_cast(self, params: Dict[str, Any]) -> str:
        column = params.get("column", "")
        strip_chars = params.get("strip_chars", "")
        target_dtype = params.get("target_dtype", "float64")

        if column not in self.df.columns:
            return f"Column '{column}' not found."

        col = self.df[column].copy()
        # Replace 'missing' with NaN first
        col = col.replace("missing", np.nan)
        col = col.replace("Missing", np.nan)

        # Strip specified characters
        non_null_mask = col.notnull()
        if strip_chars:
            for ch in strip_chars:
                col[non_null_mask] = col[non_null_mask].astype(str).str.replace(ch, "", regex=False)

        # Cast to target dtype
        try:
            col = pd.to_numeric(col, errors="coerce")
            if target_dtype == "float64":
                col = col.astype("float64")
            elif target_dtype == "int64":
                col = col.astype("Int64")  # nullable int
        except Exception as e:
            return f"Failed to cast {column}: {str(e)}"

        self.df[column] = col
        non_null = int(col.notnull().sum())
        sample = col.dropna().head(3).tolist()
        return (
            f"Applied strip_and_cast to {column}. Column now has dtype {col.dtype}. "
            f"Sample values: {sample}. {non_null} non-null values."
        )

    def _op_replace_value(self, params: Dict[str, Any]) -> str:
        column = params.get("column", "")
        old_value = params.get("old_value", "")
        new_value = params.get("new_value", None)

        if column and column not in self.df.columns:
            return f"Column '{column}' not found."

        if new_value is None or (isinstance(new_value, str) and new_value.lower() == "nan"):
            new_value = np.nan

        if column:
            count = int((self.df[column] == old_value).sum())
            self.df[column] = self.df[column].replace(old_value, new_value)
        else:
            count = 0
            for c in self.df.columns:
                c_count = int((self.df[c] == old_value).sum())
                count += c_count
                self.df[c] = self.df[c].replace(old_value, new_value)

        return f"Replaced '{old_value}' with {repr(new_value)} in {'column ' + column if column else 'all columns'}. {count} values replaced."

    def _op_normalize_zscore(self, column: str) -> str:
        if column not in self.df.columns:
            return f"Column '{column}' not found."

        col = pd.to_numeric(self.df[column], errors="coerce")
        mean_val = col.mean()
        std_val = col.std()

        if std_val == 0 or pd.isna(std_val):
            return f"Column '{column}' has zero variance, cannot z-score normalize."

        self.df[column] = ((col - mean_val) / std_val).round(6)

        new_mean = self.df[column].mean()
        new_std = self.df[column].std()
        return (
            f"Z-score normalized {column}. New mean={new_mean:.4f}, std={new_std:.4f}. "
            f"Sample values: {self.df[column].dropna().head(3).tolist()}"
        )

    def _op_normalize_minmax(self, column: str) -> str:
        if column not in self.df.columns:
            return f"Column '{column}' not found."

        col = pd.to_numeric(self.df[column], errors="coerce")
        min_val = col.min()
        max_val = col.max()

        if max_val == min_val:
            return f"Column '{column}' has zero range, cannot min-max normalize."

        self.df[column] = ((col - min_val) / (max_val - min_val)).round(6)

        return (
            f"Min-max normalized {column} to [0, 1]. "
            f"Sample values: {self.df[column].dropna().head(3).tolist()}"
        )

    def _op_rename_column(self, params: Dict[str, Any]) -> str:
        old_name = params.get("old_name", "")
        new_name = params.get("new_name", "")

        if old_name not in self.df.columns:
            return f"Column '{old_name}' not found."

        self.df = self.df.rename(columns={old_name: new_name})
        return f"Renamed column '{old_name}' to '{new_name}'."

    def _op_drop_column(self, column: str) -> str:
        if column not in self.df.columns:
            return f"Column '{column}' not found."

        self.df = self.df.drop(columns=[column])
        return f"Dropped column '{column}'. Remaining columns: {list(self.df.columns)}"

    def _op_encode_onehot(self, column: str) -> str:
        if column not in self.df.columns:
            return f"Column '{column}' not found."

        n_unique = self.df[column].nunique()
        if n_unique > 50:
            return (
                f"Column '{column}' has {n_unique} unique values. "
                f"One-hot encoding would create {n_unique} new columns. "
                f"Consider dropping the column instead."
            )

        dummies = pd.get_dummies(self.df[column], prefix=column, dtype=int)
        self.df = self.df.drop(columns=[column])
        self.df = pd.concat([self.df, dummies], axis=1)

        return (
            f"One-hot encoded '{column}' into {len(dummies.columns)} columns: "
            f"{list(dummies.columns)[:10]}{'...' if len(dummies.columns) > 10 else ''}. "
            f"New DataFrame shape: {self.df.shape}"
        )

    def _op_cast_datetime_to_numeric(self, column: str) -> str:
        if column not in self.df.columns:
            return f"Column '{column}' not found."

        try:
            dt_col = pd.to_datetime(self.df[column], errors="coerce")
            epoch = pd.Timestamp("1970-01-01")
            # Convert to days since epoch
            if dt_col.dt.tz is not None:
                dt_col = dt_col.dt.tz_localize(None)
            self.df[column] = (dt_col - epoch).dt.days
            self.df[column] = self.df[column].astype("float64")
            return (
                f"Converted '{column}' to days-since-epoch (float64). "
                f"Sample values: {self.df[column].dropna().head(3).tolist()}"
            )
        except Exception as e:
            return f"Failed to convert '{column}': {str(e)}"

    def _op_filter_values(self, params: Dict[str, Any]) -> str:
        column = params.get("column", "")
        value = params.get("value", "")

        if column not in self.df.columns:
            return f"Column '{column}' not found."

        before = len(self.df)
        self.df = self.df[self.df[column] != value].reset_index(drop=True)
        after = len(self.df)
        removed = before - after

        return f"Filtered out rows where {column}=='{value}'. Removed {removed} rows. New shape: {self.df.shape}"

    def _op_convert_timezone(self, params: Dict[str, Any]) -> str:
        column = params.get("column", "")
        offset_hours = params.get("offset_hours", 0)

        if column not in self.df.columns:
            return f"Column '{column}' not found."

        try:
            offset_hours = float(offset_hours)
            dt_col = pd.to_datetime(self.df[column], errors="coerce")
            dt_col = dt_col - pd.Timedelta(hours=offset_hours)
            self.df[column] = dt_col
            return (
                f"Shifted '{column}' by {-offset_hours} hours (subtracted {offset_hours}h offset). "
                f"Sample values: {self.df[column].head(3).tolist()}"
            )
        except Exception as e:
            return f"Failed to convert timezone for '{column}': {str(e)}"

    def _op_fix_stage(self, params: Dict[str, Any]) -> str:
        """Task 5 specific: fix a pipeline stage."""
        if self.task_id != 5:
            return "fix_stage operation is only available for Task 5."

        stage_id = params.get("stage_id")
        fix_type = params.get("fix_type", "")

        if stage_id is None:
            return "Missing required parameter: stage_id"

        stage_id = int(stage_id)
        result = apply_stage_fix(self.df, stage_id, fix_type)
        return result

    def _inspect_stage(self, params: Dict[str, Any]) -> str:
        """Task 5 specific: inspect a pipeline stage."""
        if self.task_id != 5:
            return "inspect_stage is only available for Task 5."

        stage_id = params.get("stage_id")
        if stage_id is None:
            return "Missing required parameter: stage_id"

        stage_id = int(stage_id)
        return get_stage_info(self.df, stage_id)

    def _run_tests(self) -> str:
        """Run the grader and return results."""
        results = GRADERS[self.task_id](self.df)
        self._cached_tests_passed = sum(results)
        self.reward = self._compute_reward_from_results(results)

        passed = sum(results)
        total = len(results)

        details = []
        test_names = self._get_test_names()
        for i, (name, passed_flag) in enumerate(zip(test_names, results)):
            status = "✓ PASS" if passed_flag else "✗ FAIL"
            details.append(f"  Test {i+1}: {status} — {name}")

        if self.reward >= 0.95:
            self.done = True

        return (
            f"{passed}/{total} tests passed. Current reward: {self.reward:.4f}\n"
            + "\n".join(details)
        )

    def _check_column_stats(self, params: Dict[str, Any]) -> str:
        column = params.get("column", "")

        if column not in self.df.columns:
            return f"Column '{column}' not found. Available: {list(self.df.columns)}"

        col = self.df[column]
        stats = {
            "dtype": str(col.dtype),
            "null_count": int(col.isnull().sum()),
            "non_null_count": int(col.notnull().sum()),
            "unique_count": int(col.nunique()),
        }

        if np.issubdtype(col.dtype, np.number):
            stats.update({
                "min": float(col.min()) if col.notnull().any() else None,
                "max": float(col.max()) if col.notnull().any() else None,
                "mean": float(col.mean()) if col.notnull().any() else None,
                "std": float(col.std()) if col.notnull().any() else None,
            })
            sample = col.dropna().head(5).tolist()
        else:
            sample = col.dropna().head(5).tolist()
            # Show value counts for object columns
            vc = col.value_counts().head(5).to_dict()
            stats["top_values"] = vc

        stats["sample_values"] = sample
        return f"Column '{column}' stats: {stats}"

    # ── Reward computation ──────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        if self.df is None:
            return 0.0
        results = GRADERS[self.task_id](self.df)
        return self._compute_reward_from_results(results)

    def _compute_reward_from_results(self, results: List[bool]) -> float:
        if self.df is None:
            return 0.0

        base_reward = sum(results) / len(results)

        # Penalty for dropping protected columns
        if self._columns_dropped_incorrectly():
            base_reward *= 0.7

        # Gentle time pressure
        step_penalty = max(0, (self.step_count - 20) * 0.01)
        final = max(0.0, min(1.0, base_reward - step_penalty))

        return round(final, 4)

    def _columns_dropped_incorrectly(self) -> bool:
        """Check if any protected column was dropped."""
        protected = PROTECTED.get(self.task_id, [])
        for col in protected:
            if col not in self.df.columns:
                # Special case for Task 3: user_id may appear as customer_id initially
                if self.task_id == 3 and col == "user_id":
                    continue
                return True
        return False

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _info_dict(self) -> Dict[str, Any]:
        info = {
            "tests_passed": self._cached_tests_passed,
            "tests_total": 5,
            "step_count": self.step_count,
            "task_id": self.task_id,
            "warnings": [],
        }

        if self.task_id == 5:
            info["root_cause_stage"] = (
                2 if self.df is not None and self.df.attrs.get("_root_cause_identified", False) else None
            )

        return info

    def _get_reset_observation(self) -> str:
        task_info = TASK_INFOS[self.task_id]
        return (
            f"Task {self.task_id}: {task_info['name']} ({task_info['difficulty']})\n"
            f"{task_info['description']}\n"
            f"DataFrame shape: {self.df.shape}. "
            f"Columns: {list(self.df.columns)}. "
            f"Max steps: {self.max_steps}. "
            f"Hint: {task_info['hint']}"
        )

    def _get_test_names(self) -> List[str]:
        names = {
            1: [
                "interest_rate is float64 (no % chars)",
                "annual_income is float64 (no $ or , chars)",
                "loan_amount is numeric (no commas)",
                "credit_score is numeric",
                "No literal 'missing' strings in any column",
            ],
            2: [
                "city is not ordinal-encoded integer",
                "age is z-score normalized (mean≈0, std≈1)",
                "monthly_spending is z-score normalized",
                "signup_date is numeric (days-since-epoch)",
                "Protected columns still exist",
            ],
            3: [
                "Column renamed back to user_id",
                "No 'enterprise' values in tier",
                "Timestamps converted back to UTC",
                "Row count in expected range",
                "Core numeric columns still exist",
            ],
            4: [
                "days_to_churn column removed",
                "churn_reason_code column removed",
                "cancellation_flag column removed",
                "Legitimate feature columns preserved",
                "Target column 'churned' intact",
            ],
            5: [
                "No null values in total_revenue",
                "total_revenue is positive and valid",
                "unique_customers not double-counted",
                "Root cause identified (Stage 2)",
                "All required output columns present",
            ],
        }
        return names.get(self.task_id, [f"Test {i+1}" for i in range(5)])

    def _error_response(self, msg: str) -> Dict[str, Any]:
        return {
            "observation": msg,
            "reward": 0.0,
            "done": False,
            "info": {
                "tests_passed": 0,
                "tests_total": 5,
                "step_count": self.step_count,
                "task_id": self.task_id,
                "warnings": [msg],
            },
        }


# Global singleton
env_state = EnvironmentState()
