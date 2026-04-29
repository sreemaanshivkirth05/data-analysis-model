import json
import os
import sys
from typing import Any, Dict, List, Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

EVAL_FILE = os.path.join(PROJECT_ROOT, "eval", "planner_eval_cases.jsonl")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "eval", "planner_eval_results.json")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


# =========================================================
# Planner connection
# =========================================================

def build_metadata_from_eval_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert eval case columns/dtypes into the metadata format
    your current predict.py expects.
    """

    dtypes = case.get("dtypes", {})

    has_numeric = False
    has_category = False
    has_datetime = False
    has_text = False

    for dtype in dtypes.values():
        dtype_str = str(dtype).lower()

        if any(x in dtype_str for x in ["int", "float", "double", "decimal", "number", "numeric"]):
            has_numeric = True

        elif any(x in dtype_str for x in ["date", "time", "datetime", "timestamp"]):
            has_datetime = True

        elif any(x in dtype_str for x in ["text", "string", "object"]):
            has_text = True
            has_category = True

        elif any(x in dtype_str for x in ["category", "categorical", "bool"]):
            has_category = True

    metadata = {
        "source_type": case.get("source_type", "csv"),
        "row_count": case.get("row_count", None),
        "column_count": len(case.get("columns", [])),
        "columns": case.get("columns", []),
        "dtypes": dtypes,
        "has_numeric": has_numeric,
        "has_category": has_category,
        "has_datetime": has_datetime,
        "has_text": has_text,
    }

    return metadata


def call_current_planner(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls your current predict_plan(question, metadata).
    """

    try:
        from predict import predict_plan

        question = case["question"]
        metadata = build_metadata_from_eval_case(case)

        prediction = predict_plan(question, metadata)

        if not isinstance(prediction, dict):
            return {
                "_error": "predict_plan did not return a dictionary.",
                "_raw_output": str(prediction)
            }

        return prediction

    except Exception as e:
        return {
            "_error": str(e),
            "_raw_output": None
        }


# =========================================================
# Load eval data
# =========================================================

def load_eval_cases(path: str) -> List[Dict[str, Any]]:
    cases = []

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in eval file on line {line_number}: {e}"
                )

    return cases


# =========================================================
# Normalization helpers
# =========================================================

def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def normalize_chart(chart: Optional[str]) -> Optional[str]:
    if chart is None:
        return None

    chart = str(chart).strip().lower()

    chart_aliases = {
        "bar": "bar_chart",
        "bar_chart": "bar_chart",
        "horizontal_bar": "horizontal_bar_chart",
        "horizontal_bar_chart": "horizontal_bar_chart",
        "line": "line_chart",
        "line_chart": "line_chart",
        "area": "area_chart",
        "area_chart": "area_chart",
        "scatter": "scatter_plot",
        "scatter_plot": "scatter_plot",
        "box": "box_plot",
        "boxplot": "box_plot",
        "box_plot": "box_plot",
        "histogram": "histogram",
        "heatmap": "heatmap",
        "correlation_heatmap": "correlation_heatmap",
        "table": "table",
        "kpi": "kpi_card",
        "kpi_card": "kpi_card",
        "kpi_cards": "kpi_cards",
        "multi_chart_dashboard": "multi_chart_dashboard",
        "none": "none",
    }

    return chart_aliases.get(chart, chart)


def normalize_operation(operation: Optional[str]) -> Optional[str]:
    if operation is None:
        return None

    operation = str(operation).strip().lower()

    operation_aliases = {
        "sum": "sum",
        "total": "sum",
        "mean": "mean",
        "average": "mean",
        "count": "count",
        "nunique": "nunique",
        "groupby_sum": "groupby_sum",
        "groupby_sum_sort_desc": "groupby_sum_sort_desc",
        "time_groupby_sum": "time_groupby_sum",
        "trend_sum_by_date": "time_groupby_sum",
        "correlation": "correlation",
        "correlation_heatmap": "correlation_heatmap",
        "distribution": "distribution",
        "outlier_check": "outlier_check",
        "null_check": "null_check",
        "duplicate_check": "duplicate_check",
        "data_quality_summary": "data_quality_summary",
        "full_dataset_analysis": "full_dataset_analysis",
        "forecast": "forecast",
        "text_summary": "text_summary",
        "word_frequency": "word_frequency",
        "sentiment_summary": "sentiment_summary",
        "list_columns": "list_columns",
        "count_rows": "count_rows",
    }

    return operation_aliases.get(operation, operation)


def exact_match(expected: Any, actual: Any) -> bool:
    return normalize_value(expected) == normalize_value(actual)


def operation_match(expected: Any, actual: Any) -> bool:
    return normalize_operation(expected) == normalize_operation(actual)


def chart_match(expected: Any, actual: Any) -> bool:
    return normalize_chart(expected) == normalize_chart(actual)


# =========================================================
# Scoring
# =========================================================

def score_case(case: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    expected = case.get("expected", {})

    result = {
        "id": case.get("id"),
        "question": case.get("question"),
        "json_valid": isinstance(prediction, dict) and "_error" not in prediction,
        "intent_correct": False,
        "operation_correct": False,
        "best_chart_correct": False,
        "chart_required_correct": False,
        "needs_numeric_correct": False,
        "needs_category_correct": False,
        "needs_datetime_correct": False,
        "needs_text_correct": False,
        "passed": False,
        "prediction": prediction,
        "expected": expected,
        "rules_fired": prediction.get("rules_fired", []) if isinstance(prediction, dict) else [],
        "planner_source": prediction.get("planner_source") if isinstance(prediction, dict) else None,
        "min_confidence": prediction.get("min_confidence") if isinstance(prediction, dict) else None,
    }

    if not result["json_valid"]:
        return result

    predicted_roles = prediction.get("required_data_roles", {})
    expected_roles = expected.get("required_data_roles", {})

    result["intent_correct"] = exact_match(
        expected.get("intent"),
        prediction.get("intent")
    )

    result["operation_correct"] = operation_match(
        expected.get("operation"),
        prediction.get("operation")
    )

    result["best_chart_correct"] = chart_match(
        expected.get("best_chart"),
        prediction.get("best_chart")
    )

    result["chart_required_correct"] = exact_match(
        expected.get("chart_required"),
        prediction.get("chart_required")
    )

    result["needs_numeric_correct"] = exact_match(
        expected_roles.get("needs_numeric"),
        predicted_roles.get("needs_numeric")
    )

    result["needs_category_correct"] = exact_match(
        expected_roles.get("needs_category"),
        predicted_roles.get("needs_category")
    )

    result["needs_datetime_correct"] = exact_match(
        expected_roles.get("needs_datetime"),
        predicted_roles.get("needs_datetime")
    )

    result["needs_text_correct"] = exact_match(
        expected_roles.get("needs_text"),
        predicted_roles.get("needs_text")
    )

    required_checks = [
        result["json_valid"],
        result["intent_correct"],
        result["operation_correct"],
        result["best_chart_correct"],
        result["chart_required_correct"],
        result["needs_numeric_correct"],
        result["needs_category_correct"],
        result["needs_datetime_correct"],
        result["needs_text_correct"],
    ]

    result["passed"] = all(required_checks)

    return result


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)

    def count_true(key: str) -> int:
        return sum(1 for r in results if r.get(key) is True)

    def percentage(count: int) -> float:
        if total == 0:
            return 0.0
        return round((count / total) * 100, 2)

    passed = count_true("passed")
    json_valid = count_true("json_valid")
    intent_correct = count_true("intent_correct")
    operation_correct = count_true("operation_correct")
    best_chart_correct = count_true("best_chart_correct")
    chart_required_correct = count_true("chart_required_correct")
    needs_numeric_correct = count_true("needs_numeric_correct")
    needs_category_correct = count_true("needs_category_correct")
    needs_datetime_correct = count_true("needs_datetime_correct")
    needs_text_correct = count_true("needs_text_correct")

    summary = {
        "total_cases": total,
        "passed": passed,
        "json_valid": json_valid,
        "intent_correct": intent_correct,
        "operation_correct": operation_correct,
        "best_chart_correct": best_chart_correct,
        "chart_required_correct": chart_required_correct,
        "needs_numeric_correct": needs_numeric_correct,
        "needs_category_correct": needs_category_correct,
        "needs_datetime_correct": needs_datetime_correct,
        "needs_text_correct": needs_text_correct,
        "accuracy_percentages": {
            "overall_pass_rate": percentage(passed),
            "json_valid": percentage(json_valid),
            "intent_correct": percentage(intent_correct),
            "operation_correct": percentage(operation_correct),
            "best_chart_correct": percentage(best_chart_correct),
            "chart_required_correct": percentage(chart_required_correct),
            "needs_numeric_correct": percentage(needs_numeric_correct),
            "needs_category_correct": percentage(needs_category_correct),
            "needs_datetime_correct": percentage(needs_datetime_correct),
            "needs_text_correct": percentage(needs_text_correct),
        }
    }

    return summary


# =========================================================
# Printing
# =========================================================

def print_summary(summary: Dict[str, Any]) -> None:
    print("\n================ PLANNER EVALUATION SUMMARY ================\n")

    print(f"Total cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed']}")

    print("\n---------------- Accuracy ----------------")
    for metric, value in summary["accuracy_percentages"].items():
        print(f"{metric}: {value}%")

    print("\n=============================================================\n")


def print_failed_cases(results: List[Dict[str, Any]]) -> None:
    failed_cases = [r for r in results if not r["passed"]]

    if not failed_cases:
        print("All test cases passed.")
        return

    print("\n================ FAILED CASES ================\n")

    for result in failed_cases:
        print(f"ID: {result['id']}")
        print(f"Question: {result['question']}")

        print("\nExpected:")
        print(json.dumps(result["expected"], indent=2))

        print("\nPrediction:")
        print(json.dumps(result["prediction"], indent=2))

        print("\nChecks:")
        print(f"JSON valid: {result['json_valid']}")
        print(f"Intent correct: {result['intent_correct']}")
        print(f"Operation correct: {result['operation_correct']}")
        print(f"Best chart correct: {result['best_chart_correct']}")
        print(f"Chart required correct: {result['chart_required_correct']}")
        print(f"Needs numeric correct: {result['needs_numeric_correct']}")
        print(f"Needs category correct: {result['needs_category_correct']}")
        print(f"Needs datetime correct: {result['needs_datetime_correct']}")
        print(f"Needs text correct: {result['needs_text_correct']}")
        print(f"Rules fired: {result['rules_fired']}")
        print(f"Planner source: {result['planner_source']}")
        print(f"Min confidence: {result['min_confidence']}")

        print("\n---------------------------------------------\n")


def print_recommendations(summary: Dict[str, Any]) -> None:
    acc = summary["accuracy_percentages"]

    print("\n================ RECOMMENDATIONS ================\n")

    if acc["json_valid"] < 100:
        print("- Fix planner runtime errors first. Some predictions failed or did not return valid dictionaries.")

    if acc["intent_correct"] < 85:
        print("- Improve intent labels in training data and rule corrections.")

    if acc["operation_correct"] < 85:
        print("- Improve operation labels. The planner may need more examples for aggregation, ranking, trend, correlation, and data-quality questions.")

    if acc["best_chart_correct"] < 85:
        print("- Improve chart labels and chart correction rules.")

    if acc["needs_numeric_correct"] < 85:
        print("- Improve required_data_roles.needs_numeric labels in training data.")

    if acc["needs_category_correct"] < 85:
        print("- Improve required_data_roles.needs_category labels in training data.")

    if acc["needs_datetime_correct"] < 85:
        print("- Improve required_data_roles.needs_datetime labels in training data.")

    if acc["needs_text_correct"] < 85:
        print("- Improve required_data_roles.needs_text labels in training data.")

    if acc["overall_pass_rate"] >= 85:
        print("- Your current ML + rules planner is strong enough. Fine-tuning is not urgent.")
    elif 70 <= acc["overall_pass_rate"] < 85:
        print("- Your planner is decent. Improve weak labels and rules before doing more fine-tuning.")
    else:
        print("- Your planner needs targeted improvement. Use failed cases to expand the training dataset and rule layer.")

    print("\n=================================================\n")


# =========================================================
# Main
# =========================================================

def main() -> None:
    cases = load_eval_cases(EVAL_FILE)

    results = []

    for case in cases:
        prediction = call_current_planner(case)
        result = score_case(case, prediction)
        results.append(result)

    summary = summarize_results(results)

    print_summary(summary)
    print_failed_cases(results)
    print_recommendations(summary)

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "results": results
            },
            file,
            indent=2
        )

    print(f"Detailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()