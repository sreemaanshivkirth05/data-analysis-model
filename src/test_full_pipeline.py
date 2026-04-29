import json
import numpy as np
import pandas as pd

from metadata_profiler import profile_file
from llm_router import route_question
from predict import predict_plan
from column_mapper import map_columns_to_plan
from operation_executor import load_dataset_for_execution, execute_operation
from chart_builder import build_chart_config


# ============================================================
# JSON Safety Helpers
# ============================================================

def make_json_safe(obj):
    """
    Convert NumPy/Pandas objects into JSON-safe Python objects.
    """

    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_safe(item) for item in obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)

    try:
        if pd.isna(obj) and not isinstance(obj, (dict, list, tuple, str)):
            return None
    except Exception:
        pass

    return obj


def print_json(title, data):
    """
    Pretty-print JSON-safe output.
    """

    print(f"\n================ {title} ================\n")
    safe_data = make_json_safe(data)
    print(json.dumps(safe_data, indent=2))


# ============================================================
# Router Fallback Helpers
# ============================================================

ANALYSIS_HINTS = [
    "missing",
    "null",
    "blank",
    "empty",
    "duplicate",
    "duplicates",
    "data quality",
    "sales",
    "revenue",
    "profit",
    "amount",
    "cost",
    "price",
    "orders",
    "clicks",
    "trend",
    "over time",
    "monthly",
    "daily",
    "weekly",
    "yearly",
    "compare",
    "comparison",
    "highest",
    "top",
    "lowest",
    "average",
    "mean",
    "total",
    "sum",
    "distribution",
    "outlier",
    "outliers",
    "correlation",
    "relationship",
    "heatmap",
    "matrix",
    "forecast",
    "predict",
    "analyze",
    "analysis",
    "insights",
    "summary",
    "reviews",
    "comments",
    "feedback",
    "sentiment",
    "keywords",
    "word frequency",
]


def normalize_text(text):
    return str(text).lower().strip()


def looks_like_analysis_question(question):
    """
    Lightweight fallback check.

    Used when the LLM router says conversation, but the normalized question
    clearly contains analysis intent.
    """

    q = normalize_text(question)

    if not q:
        return False

    return any(keyword in q for keyword in ANALYSIS_HINTS)


def should_override_router_to_analysis(route):
    """
    Decide whether to override a conversation route.

    Example:
    User typed: are there migginssgs valuse
    Router returned:
      route = conversation
      normalized_question = Are there missing values?

    Since the normalized question clearly asks about missing values,
    we should continue to the planner.
    """

    is_analysis_request = route.get("is_analysis_request", False)
    normalized_question = route.get("normalized_question", "")

    if is_analysis_request is True:
        return False

    return looks_like_analysis_question(normalized_question)


def get_router_conversation_response(route):
    """
    Return a clean message for non-analysis questions.
    """

    reason = route.get("reason", "This does not look like a dataset-analysis request.")

    return (
        f"{reason}\n"
        "Try asking something like: 'Which product has the highest sales?'"
    )


# ============================================================
# Main Test Runner
# ============================================================

def main():
    print("\n================ FULL ANALYSIS PIPELINE TEST ================\n")

    dataset_path = input("Enter dataset path: ").strip()

    try:
        df = load_dataset_for_execution(dataset_path)
        metadata = profile_file(dataset_path)
    except Exception as error:
        print("\nFailed to load dataset.")
        print(str(error))
        return

    print_json(
        "DATASET METADATA SUMMARY",
        {
            "source_type": metadata.get("source_type"),
            "row_count": metadata.get("row_count"),
            "column_count": metadata.get("column_count"),
            "has_numeric": metadata.get("has_numeric"),
            "has_category": metadata.get("has_category"),
            "has_datetime": metadata.get("has_datetime"),
            "has_text": metadata.get("has_text"),
            "has_boolean": metadata.get("has_boolean"),
            "has_geography": metadata.get("has_geography"),
            "has_identifier": metadata.get("has_identifier"),
            "has_currency": metadata.get("has_currency"),
            "has_percentage": metadata.get("has_percentage"),
        },
    )

    while True:
        question = input("\nAsk a question, or type 'exit': ").strip()

        if question.lower() == "exit":
            print("\nExiting full pipeline test.")
            break

        if not question:
            print("\nPlease enter a question.")
            continue

        # ----------------------------------------------------
        # 1. Router
        # ----------------------------------------------------

        try:
            route = route_question(question, metadata)
        except Exception as error:
            route = {
                "route": "analysis",
                "is_analysis_request": True,
                "confidence": 0.0,
                "reason": f"Router failed, falling back to analysis pipeline: {error}",
                "normalized_question": question,
                "router_error": str(error),
            }

        print_json("ROUTER DECISION", route)

        router_override = should_override_router_to_analysis(route)

        if route.get("is_analysis_request") is not True and not router_override:
            print("\nConversation Response:")
            print(get_router_conversation_response(route))
            continue

        normalized_question = route.get("normalized_question", question)

        if router_override:
            print("\nRouter Fallback Applied:")
            print(
                "The router marked this as conversation, but the normalized question "
                "looks like a dataset-analysis request. Continuing with planner."
            )

        print("\nNormalized Question:")
        print(normalized_question)

        # ----------------------------------------------------
        # 2. ML Planner + Rule Correction
        # ----------------------------------------------------

        try:
            plan = predict_plan(normalized_question, metadata)
        except Exception as error:
            print("\nPlanner failed.")
            print(str(error))
            continue

        print_json("ML PLANNER OUTPUT", plan)

        # ----------------------------------------------------
        # 3. Column Mapper
        # ----------------------------------------------------

        try:
            mapped_plan = map_columns_to_plan(
                question=normalized_question,
                metadata=metadata,
                plan=plan,
            )
        except Exception as error:
            print("\nColumn mapper failed.")
            print(str(error))
            continue

        print_json("MAPPED PLAN", mapped_plan)

        if mapped_plan.get("is_executable") is not True:
            print("\nPlan is not executable.")
            print_json(
                "VALIDATION MESSAGES",
                mapped_plan.get("validation_messages", []),
            )
            continue

        # ----------------------------------------------------
        # 4. Operation Executor
        # ----------------------------------------------------

        try:
            result = execute_operation(
                df=df,
                mapped_plan=mapped_plan,
            )
        except Exception as error:
            print("\nOperation execution failed.")
            print(str(error))
            continue

        print_json("EXECUTION RESULT", result)

        # ----------------------------------------------------
        # 5. Chart Builder
        # ----------------------------------------------------

        try:
            chart_output = build_chart_config(
                result=result,
                mapped_plan=mapped_plan,
            )
        except Exception as error:
            print("\nChart builder failed.")
            print(str(error))
            continue

        print_json("CHART BUILDER OUTPUT", chart_output)


if __name__ == "__main__":
    main()