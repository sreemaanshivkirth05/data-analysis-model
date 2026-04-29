import os
import json
import datetime
from typing import Dict, Any, List

import pandas as pd

from predict import predict_plan
from column_mapper import map_columns_to_plan


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_FILE = os.path.join(BASE_DIR, "eval", "execution_eval_cases.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ============================================================
# JSON Safety Helpers
# ============================================================

def make_json_safe(obj):
    """
    Convert objects into JSON-safe Python objects.
    """

    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_safe(item) for item in obj)

    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, (np.floating,)):
            return float(obj)

        if isinstance(obj, (np.bool_,)):
            return bool(obj)

    except Exception:
        pass

    try:
        if pd.isna(obj) and not isinstance(obj, (dict, list, tuple, str)):
            return None
    except Exception:
        pass

    return obj


def print_json(title: str, data: Any):
    print(f"\n================ {title} ================\n")
    print(json.dumps(make_json_safe(data), indent=2))


# ============================================================
# File Loading
# ============================================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL evaluation cases.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Execution eval file not found: {file_path}\n"
            "Create eval/execution_eval_cases.jsonl first."
        )

    cases = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {file_path}: {error}"
                )

    return cases


# ============================================================
# Evaluation Helpers
# ============================================================

def normalize_expected_value(value):
    """
    Normalize expected values for comparison.
    """

    if value in ["", "null", "None", "none"]:
        return None

    return value


def compare_value(predicted, expected) -> bool:
    """
    Compare predicted and expected values.
    """

    expected = normalize_expected_value(expected)

    if predicted is None and expected is None:
        return True

    return str(predicted) == str(expected)


def get_selected_columns(mapped_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract selected columns from current mapper output.
    """

    selected = mapped_plan.get("selected_columns", {})

    return {
        "measure_column": selected.get("measure_column"),
        "dimension_column": selected.get("dimension_column"),
        "time_column": selected.get("time_column"),
        "text_column": selected.get("text_column"),
    }


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate one execution planning case.
    """

    case_id = case.get("id", "unknown")
    question = case["question"]
    metadata = case["metadata"]
    expected = case["expected"]

    plan = predict_plan(question, metadata)

    mapped_plan = map_columns_to_plan(
        question=question,
        metadata=metadata,
        plan=plan,
    )

    selected_columns = get_selected_columns(mapped_plan)

    predicted = {
        "measure_column": selected_columns.get("measure_column"),
        "dimension_column": selected_columns.get("dimension_column"),
        "time_column": selected_columns.get("time_column"),
        "text_column": selected_columns.get("text_column"),
        "is_executable": mapped_plan.get("is_executable"),
    }

    matches = {
        "measure_column": compare_value(
            predicted["measure_column"],
            expected.get("measure_column"),
        ),
        "dimension_column": compare_value(
            predicted["dimension_column"],
            expected.get("dimension_column"),
        ),
        "time_column": compare_value(
            predicted["time_column"],
            expected.get("time_column"),
        ),
        "text_column": compare_value(
            predicted["text_column"],
            expected.get("text_column"),
        ),
        "is_executable": bool(predicted["is_executable"]) == bool(expected.get("is_executable")),
    }

    overall_pass = all(matches.values())

    return {
        "id": case_id,
        "question": question,
        "overall_pass": overall_pass,
        "expected": expected,
        "predicted": predicted,
        "matches": matches,
        "planner_output": {
            "intent": plan.get("intent"),
            "operation": plan.get("operation"),
            "best_chart": plan.get("best_chart"),
            "chart_required": plan.get("chart_required"),
            "required_data_roles": plan.get("required_data_roles"),
            "confidence_status": plan.get("confidence_status"),
            "min_confidence": plan.get("min_confidence"),
            "requires_llm_fallback": plan.get("requires_llm_fallback"),
            "planner_source": plan.get("planner_source"),
            "rules_fired": plan.get("rules_fired", []),
            "raw_ml_prediction": plan.get("raw_ml_prediction"),
        },
        "mapped_plan": {
            "selected_columns": mapped_plan.get("selected_columns"),
            "mapping_confidence": mapped_plan.get("mapping_confidence"),
            "mapping_warnings": mapped_plan.get("mapping_warnings"),
            "is_executable": mapped_plan.get("is_executable"),
            "validation_messages": mapped_plan.get("validation_messages"),
        },
    }


# ============================================================
# Reporting
# ============================================================

def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)

    if total == 0:
        return {
            "total_cases": 0,
            "passed": 0,
            "failed": 0,
            "overall_pass_rate": 0.0,
        }

    passed = sum(1 for item in results if item["overall_pass"])
    failed = total - passed

    measure_correct = sum(1 for item in results if item["matches"]["measure_column"])
    dimension_correct = sum(1 for item in results if item["matches"]["dimension_column"])
    time_correct = sum(1 for item in results if item["matches"]["time_column"])
    text_correct = sum(1 for item in results if item["matches"]["text_column"])
    executable_correct = sum(1 for item in results if item["matches"]["is_executable"])

    return {
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "overall_pass_rate": round((passed / total) * 100, 2),
        "measure_column_accuracy": round((measure_correct / total) * 100, 2),
        "dimension_column_accuracy": round((dimension_correct / total) * 100, 2),
        "time_column_accuracy": round((time_correct / total) * 100, 2),
        "text_column_accuracy": round((text_correct / total) * 100, 2),
        "executable_accuracy": round((executable_correct / total) * 100, 2),
    }


def save_results(results: List[Dict[str, Any]], summary: Dict[str, Any]):
    """
    Save evaluation outputs to outputs folder.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    full_json_path = os.path.join(
        OUTPUT_DIR,
        f"{timestamp}_execution_plan_eval_results.json",
    )

    failures_json_path = os.path.join(
        OUTPUT_DIR,
        f"{timestamp}_execution_plan_eval_failures.json",
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{timestamp}_execution_plan_eval_summary.csv",
    )

    failures = [item for item in results if not item["overall_pass"]]

    with open(full_json_path, "w", encoding="utf-8") as file:
        json.dump(
            make_json_safe(
                {
                    "summary": summary,
                    "results": results,
                }
            ),
            file,
            indent=2,
        )

    with open(failures_json_path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(failures), file, indent=2)

    rows = []

    for item in results:
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "overall_pass": item["overall_pass"],
                "expected_measure_column": item["expected"].get("measure_column"),
                "predicted_measure_column": item["predicted"].get("measure_column"),
                "measure_match": item["matches"].get("measure_column"),
                "expected_dimension_column": item["expected"].get("dimension_column"),
                "predicted_dimension_column": item["predicted"].get("dimension_column"),
                "dimension_match": item["matches"].get("dimension_column"),
                "expected_time_column": item["expected"].get("time_column"),
                "predicted_time_column": item["predicted"].get("time_column"),
                "time_match": item["matches"].get("time_column"),
                "expected_text_column": item["expected"].get("text_column"),
                "predicted_text_column": item["predicted"].get("text_column"),
                "text_match": item["matches"].get("text_column"),
                "expected_is_executable": item["expected"].get("is_executable"),
                "predicted_is_executable": item["predicted"].get("is_executable"),
                "executable_match": item["matches"].get("is_executable"),
                "intent": item["planner_output"].get("intent"),
                "operation": item["planner_output"].get("operation"),
                "best_chart": item["planner_output"].get("best_chart"),
                "confidence_status": item["planner_output"].get("confidence_status"),
                "min_confidence": item["planner_output"].get("min_confidence"),
                "planner_source": item["planner_output"].get("planner_source"),
                "rules_fired": item["planner_output"].get("rules_fired"),
                "mapping_warnings": item["mapped_plan"].get("mapping_warnings"),
                "validation_messages": item["mapped_plan"].get("validation_messages"),
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\nSaved outputs:")
    print(f"- {full_json_path}")
    print(f"- {failures_json_path}")
    print(f"- {csv_path}")


def print_failures(results: List[Dict[str, Any]]):
    failures = [item for item in results if not item["overall_pass"]]

    if not failures:
        print("\nNo execution-plan failures found.")
        return

    print("\n================ FAILURES ================\n")

    for item in failures:
        print(f"ID: {item['id']}")
        print(f"Question: {item['question']}")
        print(f"Expected: {item['expected']}")
        print(f"Predicted: {item['predicted']}")
        print(f"Matches: {item['matches']}")
        print(f"Planner: {item['planner_output']}")
        print(f"Mapper: {item['mapped_plan']}")
        print("-" * 80)


# ============================================================
# Main
# ============================================================

def main():
    print("\n================ EXECUTION PLAN EVALUATION ================\n")

    cases = load_jsonl(EVAL_FILE)

    results = []

    for case in cases:
        result = evaluate_case(case)
        results.append(result)

    summary = summarize_results(results)

    print_json("SUMMARY", summary)
    print_failures(results)
    save_results(results, summary)


if __name__ == "__main__":
    main()