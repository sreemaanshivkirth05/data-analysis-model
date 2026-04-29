"""
analysis_plan.py
================
Unified entry point for the Universal Analysis Planner.

Combines:
  1. predict_plan()   — ML + rule-based intent/operation/chart prediction
  2. map_columns()    — dataset-agnostic column selection

Usage
-----
    from analysis_plan import create_analysis_plan

    plan = create_analysis_plan(question, df)

    # Ready-to-use by agents:
    plan["intent"]           # e.g. "diagnostic_analysis"
    plan["operation"]        # e.g. "groupby_target_rate"
    plan["best_chart"]       # e.g. "bar_chart"
    plan["metric_column"]    # e.g. "Sales"
    plan["category_column"]  # e.g. "OverTime"
    plan["target_column"]    # e.g. "Attrition"
    plan["driver_columns"]   # e.g. ["Department", "JobRole"]
    plan["groupby_columns"]  # e.g. ["OverTime"]

Integration pattern
-------------------
    plan     = create_analysis_plan(question, df)
    result   = analysis_agent.run(df=df, question=question, plan=plan)
    chart    = visualization_agent.run(result=result, plan=plan)
    story    = narrative_agent.run(result=result, plan=plan)
    review   = reviewer_agent.run(result=result, plan=plan)
"""

import os
import sys
import traceback
from typing import Dict, Any, Optional

# Support both direct execution and package import
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from predict import predict_plan
from column_mapper import map_columns


# ============================================================
# Constants
# ============================================================

PLANNER_VERSION = "1.0.0"


# ============================================================
# Core Public Function
# ============================================================

def create_analysis_plan(
    question: str,
    df,
    metadata: Optional[Dict[str, Any]] = None,
    source_type: str = "csv",
) -> Dict[str, Any]:
    """
    Create a fully structured analysis plan from a user question and dataset.

    Parameters
    ----------
    question : str
        The user's natural language question.
        Example: "Does overtime affect attrition?"

    df : pandas.DataFrame
        The uploaded dataset.

    metadata : dict, optional
        Pre-computed metadata from metadata_profiler.profile_dataframe().
        If None, it will be computed automatically from df.
        Pass it in if you've already profiled the dataset to avoid double work.

    source_type : str
        File source type hint ("csv", "excel", "json", "parquet").
        Only used if metadata is computed internally.

    Returns
    -------
    dict
        A flat, agent-ready plan dict. All keys are always present.

    Example output
    --------------
    {
        # ── Question ────────────────────────────────────────
        "question": "Does overtime affect attrition?",

        # ── Planner predictions ─────────────────────────────
        "intent":          "diagnostic_analysis",
        "answer_depth":    "visual_answer",
        "operation":       "groupby_target_rate",
        "best_chart":      "bar_chart",
        "chart_required":  true,
        "required_data_roles": {
            "needs_numeric":  false,
            "needs_category": true,
            "needs_datetime": false,
            "needs_text":     false
        },

        # ── Column mapping ───────────────────────────────────
        "target_column":   "Attrition",
        "metric_column":   null,
        "category_column": "OverTime",
        "date_column":     null,
        "text_column":     null,
        "driver_columns":  ["OverTime"],
        "groupby_columns": ["OverTime"],
        "time_grain":      "month",

        # ── Confidence ───────────────────────────────────────
        "confidence_scores": {
            "intent":           0.82,
            "operation":        0.79,
            "best_chart":       0.81,
            "chart_required":   0.95,
            "needs_numeric":    0.90,
            "needs_category":   0.87,
            "needs_datetime":   0.91,
            "needs_text":       0.93
        },
        "mapping_confidence": 0.91,
        "min_confidence":     0.79,

        # ── Agent guidance ───────────────────────────────────
        "recommended_charts":     ["bar_chart", "horizontal_bar_chart"],
        "fallback_chart":         "table",
        "planner_source":         "ml_plus_rules",
        "rule_override_applied":  true,
        "requires_llm_fallback":  false,
        "is_executable":          true,

        # ── Metadata ─────────────────────────────────────────
        "warnings":    [],
        "errors":      [],
        "planner_version": "1.0.0"
    }
    """

    errors: list = []
    warnings: list = []

    # -------------------------------------------------------
    # 0. Input validation
    # -------------------------------------------------------
    if not question or not question.strip():
        return _empty_plan(question, errors=["Question is empty."])

    if df is None or len(df) == 0:
        return _empty_plan(question, errors=["DataFrame is empty or None."])

    # -------------------------------------------------------
    # 1. Build metadata if not provided
    # -------------------------------------------------------
    if metadata is None:
        try:
            from metadata_profiler import profile_dataframe
            metadata = profile_dataframe(df, source_type=source_type)
        except Exception as exc:
            errors.append(f"Metadata profiling failed: {exc}")
            # Proceed without metadata — predict_plan handles None gracefully
            metadata = None

    # -------------------------------------------------------
    # 2. Run ML Planner
    # -------------------------------------------------------
    try:
        plan = predict_plan(question.strip(), metadata)
    except Exception as exc:
        errors.append(f"Planner failed: {exc}")
        return _empty_plan(question, errors=errors)

    # -------------------------------------------------------
    # 3. Run Column Mapper
    # -------------------------------------------------------
    try:
        mapping = map_columns(question.strip(), df, plan, metadata=metadata)
    except Exception as exc:
        errors.append(f"Column mapper failed: {exc}")
        mapping = _empty_mapping()

    # -------------------------------------------------------
    # 4. Determine executability
    # -------------------------------------------------------
    is_executable = _check_executability(plan, mapping)

    if not is_executable:
        warnings.append(
            "Plan may not be executable — required columns could not be identified. "
            "Check warnings for details."
        )

    # -------------------------------------------------------
    # 5. Merge warnings
    # -------------------------------------------------------
    warnings.extend(mapping.get("warnings", []))
    plan_warnings = plan.get("low_confidence", [])
    if plan_warnings:
        warnings.append(f"Low planner confidence on: {', '.join(plan_warnings)}")

    # -------------------------------------------------------
    # 6. Build the final flat agent-ready plan
    # -------------------------------------------------------
    result = {
        # Question
        "question": question.strip(),

        # Planner predictions
        "intent":              plan.get("intent"),
        "answer_depth":        plan.get("answer_depth"),
        "operation":           plan.get("operation"),
        "best_chart":          plan.get("best_chart"),
        "chart_required":      plan.get("chart_required", False),
        "required_data_roles": plan.get("required_data_roles", {}),

        # Column mapping (clean, agent-ready)
        "target_column":   mapping.get("target_column"),
        "metric_column":   mapping.get("metric_column"),
        "category_column": mapping.get("category_column"),
        "date_column":     mapping.get("date_column"),
        "text_column":     mapping.get("text_column"),
        "driver_columns":  mapping.get("driver_columns", []),
        "groupby_columns": mapping.get("groupby_columns", []),
        "time_grain":      mapping.get("time_grain", "month"),

        # Confidence
        "confidence_scores":  plan.get("confidence_scores", {}),
        "mapping_confidence": mapping.get("confidence", 0.0),
        "min_confidence":     plan.get("min_confidence", 0.0),

        # Agent guidance
        "recommended_charts":    plan.get("recommended_charts", []),
        "fallback_chart":        plan.get("fallback_chart"),
        "planner_source":        plan.get("planner_source", "unknown"),
        "rule_override_applied": plan.get("rule_override_applied", False),
        "requires_llm_fallback": plan.get("requires_llm_fallback", False),
        "is_executable":         is_executable,

        # Diagnostics
        "warnings":        _dedup(warnings),
        "errors":          errors,
        "planner_version": PLANNER_VERSION,
    }

    return result


# ============================================================
# Agent-Specific Getter Helpers
# ============================================================

def get_visualization_inputs(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract everything VisualizationAgent needs from a create_analysis_plan() result.

    Usage:
        plan = create_analysis_plan(question, df)
        viz_inputs = get_visualization_inputs(plan)
        chart = visualization_agent.run(**viz_inputs)
    """
    return {
        "best_chart":      plan.get("best_chart"),
        "chart_required":  plan.get("chart_required", False),
        "metric_column":   plan.get("metric_column"),
        "category_column": plan.get("category_column"),
        "date_column":     plan.get("date_column"),
        "target_column":   plan.get("target_column"),
        "groupby_columns": plan.get("groupby_columns", []),
        "time_grain":      plan.get("time_grain", "month"),
        "recommended_charts": plan.get("recommended_charts", []),
        "fallback_chart":  plan.get("fallback_chart"),
    }


def get_narrative_inputs(plan: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract everything NarrativeAgent needs.

    Usage:
        plan   = create_analysis_plan(question, df)
        result = analysis_agent.run(df=df, plan=plan)
        narrative_inputs = get_narrative_inputs(plan, result)
        story = narrative_agent.run(**narrative_inputs)
    """
    return {
        "question":        plan.get("question"),
        "intent":          plan.get("intent"),
        "operation":       plan.get("operation"),
        "metric_column":   plan.get("metric_column"),
        "category_column": plan.get("category_column"),
        "date_column":     plan.get("date_column"),
        "target_column":   plan.get("target_column"),
        "driver_columns":  plan.get("driver_columns", []),
        "analysis_result": analysis_result,
    }


def get_reviewer_inputs(
    plan: Dict[str, Any],
    analysis_result: Dict[str, Any],
    df,
) -> Dict[str, Any]:
    """
    Extract everything ReviewerAgent needs.

    The reviewer validates:
    - Was the question directly answered?
    - Were selected columns valid (no hallucination)?
    - Was the chart type appropriate?
    - Did the narrative match the computed results?

    Usage:
        reviewer_inputs = get_reviewer_inputs(plan, result, df)
        review = reviewer_agent.run(**reviewer_inputs)
    """
    existing_columns = set(df.columns.tolist())

    used_columns = [
        col for col in [
            plan.get("metric_column"),
            plan.get("category_column"),
            plan.get("date_column"),
            plan.get("target_column"),
            plan.get("text_column"),
        ]
        if col is not None
    ] + plan.get("driver_columns", [])

    hallucinated = [col for col in used_columns if col not in existing_columns]

    return {
        "question":           plan.get("question"),
        "intent":             plan.get("intent"),
        "operation":          plan.get("operation"),
        "best_chart":         plan.get("best_chart"),
        "metric_column":      plan.get("metric_column"),
        "category_column":    plan.get("category_column"),
        "date_column":        plan.get("date_column"),
        "target_column":      plan.get("target_column"),
        "driver_columns":     plan.get("driver_columns", []),
        "used_columns":       used_columns,
        "hallucinated_cols":  hallucinated,
        "warnings":           plan.get("warnings", []),
        "is_executable":      plan.get("is_executable", False),
        "analysis_result":    analysis_result,
        "available_columns":  list(existing_columns),
    }


# ============================================================
# Internal Helpers
# ============================================================

def _check_executability(plan: Dict[str, Any], mapping: Dict[str, Any]) -> bool:
    """
    Determine if the plan has enough column information to execute.
    """
    operation = plan.get("operation", "")

    metric   = mapping.get("metric_column")
    category = mapping.get("category_column")
    date_col = mapping.get("date_column")
    text_col = mapping.get("text_column")
    target   = mapping.get("target_column")

    # Data-quality operations always work (no column selection needed)
    if operation in [
        "count_rows", "list_columns", "null_check",
        "duplicate_check", "data_quality_summary",
    ]:
        return True

    # Scalar aggregations need a metric
    if operation in ["sum", "mean", "max", "min", "distribution", "outlier_check"]:
        return metric is not None

    # Group-by operations need both metric and category
    if operation in [
        "groupby_sum", "groupby_sum_sort_desc",
        "groupby_mean", "groupby_mean_sort_desc",
    ]:
        return metric is not None and category is not None

    # Target-rate operations need a category and a target
    if operation in ["groupby_target_rate", "groupby_target_rate_sort_desc"]:
        return category is not None and target is not None

    # Time-series needs metric + date
    if operation in ["time_groupby_sum", "forecast"]:
        return metric is not None and date_col is not None

    # Correlation needs at least a metric
    if operation in ["correlation", "correlation_heatmap"]:
        return metric is not None

    # Text operations need a text column
    if operation in ["text_summary", "sentiment_summary", "word_frequency"]:
        return text_col is not None

    # Diagnostic / full analysis need at least a metric or target
    if operation in ["diagnostic_analysis", "full_dataset_analysis"]:
        return metric is not None or target is not None

    return True


def _empty_mapping() -> Dict[str, Any]:
    return {
        "target_column":   None,
        "metric_column":   None,
        "category_column": None,
        "date_column":     None,
        "text_column":     None,
        "driver_columns":  [],
        "groupby_columns": [],
        "time_grain":      "month",
        "confidence":      0.0,
        "warnings":        [],
    }


def _empty_plan(question: str, errors: list = None) -> Dict[str, Any]:
    return {
        "question":          question,
        "intent":            None,
        "answer_depth":      None,
        "operation":         None,
        "best_chart":        None,
        "chart_required":    False,
        "required_data_roles": {},
        "target_column":     None,
        "metric_column":     None,
        "category_column":   None,
        "date_column":       None,
        "text_column":       None,
        "driver_columns":    [],
        "groupby_columns":   [],
        "time_grain":        "month",
        "confidence_scores": {},
        "mapping_confidence": 0.0,
        "min_confidence":    0.0,
        "recommended_charts": [],
        "fallback_chart":    None,
        "planner_source":    "failed",
        "rule_override_applied": False,
        "requires_llm_fallback": True,
        "is_executable":     False,
        "warnings":          [],
        "errors":            errors or [],
        "planner_version":   PLANNER_VERSION,
    }


def _dedup(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ============================================================
# CLI Quick Test
# ============================================================

if __name__ == "__main__":
    import json
    import pandas as pd

    print("\n===== Analysis Plan Quick Test =====\n")

    dataset_path = input("Enter dataset path: ").strip()

    try:
        if dataset_path.endswith(".csv"):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(dataset_path)
        elif dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path)
        else:
            print(f"Unsupported file type: {dataset_path}")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    print(f"\nDataset loaded: {len(df)} rows × {len(df.columns)} columns")
    print("Columns:", list(df.columns))

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        plan = create_analysis_plan(question, df)

        print("\n" + json.dumps(plan, indent=2, default=str))
