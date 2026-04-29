import os
import json
import datetime
from typing import Dict, Any, List

import pandas as pd

from predict import predict_plan, predict_plan_raw


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_FILE = os.path.join(BASE_DIR, "eval", "planner_stress_cases.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MAX_FAILURES_TO_PRINT = 30


# ============================================================
# JSON Helpers
# ============================================================

def make_json_safe(obj):
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
# Load Cases
# ============================================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Planner stress file not found: {file_path}\n"
            "Run python src/generate_planner_stress_cases.py first."
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
                    f"Invalid JSON on line {line_number}: {error}"
                )

    return cases


# ============================================================
# Evaluation
# ============================================================

def evaluate_case(case: Dict[str, Any], mode: str) -> Dict[str, Any]:
    question = case["question"]
    metadata = case["metadata"]
    expected = case["expected"]

    if mode == "raw_ml_only":
        predicted = predict_plan_raw(question, metadata)
    elif mode == "final_with_rules":
        predicted = predict_plan(question, metadata)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    expected_roles = expected.get("required_data_roles", {})
    predicted_roles = predicted.get("required_data_roles", {})

    matches = {
        "intent": predicted.get("intent") == expected.get("intent"),
        "answer_depth": predicted.get("answer_depth") == expected.get("answer_depth"),
        "operation": predicted.get("operation") == expected.get("operation"),
        "best_chart": predicted.get("best_chart") == expected.get("best_chart"),
        "chart_required": bool(predicted.get("chart_required")) == bool(expected.get("chart_required")),
        "needs_numeric": bool(predicted_roles.get("needs_numeric")) == bool(expected_roles.get("needs_numeric")),
        "needs_category": bool(predicted_roles.get("needs_category")) == bool(expected_roles.get("needs_category")),
        "needs_datetime": bool(predicted_roles.get("needs_datetime")) == bool(expected_roles.get("needs_datetime")),
        "needs_text": bool(predicted_roles.get("needs_text")) == bool(expected_roles.get("needs_text")),
    }

    overall_pass = all(matches.values())

    return {
        "id": case.get("id"),
        "question": question,
        "mode": mode,
        "overall_pass": overall_pass,
        "expected": expected,
        "predicted": {
            "intent": predicted.get("intent"),
            "answer_depth": predicted.get("answer_depth"),
            "operation": predicted.get("operation"),
            "best_chart": predicted.get("best_chart"),
            "chart_required": predicted.get("chart_required"),
            "required_data_roles": predicted.get("required_data_roles"),
            "confidence_status": predicted.get("confidence_status"),
            "min_confidence": predicted.get("min_confidence"),
            "planner_source": predicted.get("planner_source"),
            "rules_fired": predicted.get("rules_fired", []),
            "raw_ml_prediction": predicted.get("raw_ml_prediction"),
        },
        "matches": matches,
    }


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

    field_names = [
        "intent",
        "answer_depth",
        "operation",
        "best_chart",
        "chart_required",
        "needs_numeric",
        "needs_category",
        "needs_datetime",
        "needs_text",
    ]

    summary = {
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "overall_pass_rate": round((passed / total) * 100, 2),
    }

    for field in field_names:
        correct = sum(1 for item in results if item["matches"][field])
        summary[f"{field}_accuracy"] = round((correct / total) * 100, 2)

    return summary


def save_results(results: List[Dict[str, Any]], summary: Dict[str, Any], mode: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    full_json_path = os.path.join(
        OUTPUT_DIR,
        f"{timestamp}_{mode}_planner_stress_results.json",
    )

    failures_json_path = os.path.join(
        OUTPUT_DIR,
        f"{timestamp}_{mode}_planner_stress_failures.json",
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{timestamp}_{mode}_planner_stress_summary.csv",
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
        expected_roles = item["expected"].get("required_data_roles", {})
        predicted_roles = item["predicted"].get("required_data_roles", {})

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "overall_pass": item["overall_pass"],

                "expected_intent": item["expected"].get("intent"),
                "predicted_intent": item["predicted"].get("intent"),
                "intent_match": item["matches"].get("intent"),

                "expected_answer_depth": item["expected"].get("answer_depth"),
                "predicted_answer_depth": item["predicted"].get("answer_depth"),
                "answer_depth_match": item["matches"].get("answer_depth"),

                "expected_operation": item["expected"].get("operation"),
                "predicted_operation": item["predicted"].get("operation"),
                "operation_match": item["matches"].get("operation"),

                "expected_best_chart": item["expected"].get("best_chart"),
                "predicted_best_chart": item["predicted"].get("best_chart"),
                "best_chart_match": item["matches"].get("best_chart"),

                "expected_chart_required": item["expected"].get("chart_required"),
                "predicted_chart_required": item["predicted"].get("chart_required"),
                "chart_required_match": item["matches"].get("chart_required"),

                "expected_needs_numeric": expected_roles.get("needs_numeric"),
                "predicted_needs_numeric": predicted_roles.get("needs_numeric"),
                "needs_numeric_match": item["matches"].get("needs_numeric"),

                "expected_needs_category": expected_roles.get("needs_category"),
                "predicted_needs_category": predicted_roles.get("needs_category"),
                "needs_category_match": item["matches"].get("needs_category"),

                "expected_needs_datetime": expected_roles.get("needs_datetime"),
                "predicted_needs_datetime": predicted_roles.get("needs_datetime"),
                "needs_datetime_match": item["matches"].get("needs_datetime"),

                "expected_needs_text": expected_roles.get("needs_text"),
                "predicted_needs_text": predicted_roles.get("needs_text"),
                "needs_text_match": item["matches"].get("needs_text"),

                "confidence_status": item["predicted"].get("confidence_status"),
                "min_confidence": item["predicted"].get("min_confidence"),
                "planner_source": item["predicted"].get("planner_source"),
                "rules_fired": item["predicted"].get("rules_fired"),
                "raw_ml_prediction": item["predicted"].get("raw_ml_prediction"),
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {
        "full_json_path": full_json_path,
        "failures_json_path": failures_json_path,
        "csv_path": csv_path,
        "failure_count": len(failures),
    }


def print_failures(results: List[Dict[str, Any]], mode: str):
    failures = [item for item in results if not item["overall_pass"]]

    if not failures:
        print(f"\nNo planner stress failures found for mode: {mode}")
        return

    print(f"\n================ FAILURES: {mode} ================\n")
    print(f"Total failures: {len(failures)}")
    print(f"Showing first {min(MAX_FAILURES_TO_PRINT, len(failures))} only.\n")

    for item in failures[:MAX_FAILURES_TO_PRINT]:
        print(f"ID: {item['id']}")
        print(f"Question: {item['question']}")
        print(f"Expected: {item['expected']}")
        print(f"Predicted: {item['predicted']}")
        print(f"Matches: {item['matches']}")
        print("-" * 80)


def run_mode(mode: str):
    print(f"\n================ PLANNER STRESS EVALUATION: {mode} ================\n")

    cases = load_jsonl(EVAL_FILE)
    results = [evaluate_case(case, mode) for case in cases]

    summary = summarize_results(results)

    print_json(f"SUMMARY: {mode}", summary)
    print_failures(results, mode)

    saved = save_results(results, summary, mode)

    print("\nSaved outputs:")
    print(f"- {saved['full_json_path']}")
    print(f"- {saved['failures_json_path']}")
    print(f"- {saved['csv_path']}")

    return summary


def main():
    final_summary = run_mode("final_with_rules")
    raw_summary = run_mode("raw_ml_only")

    print("\n================ COMPARISON ================\n")
    print("Final with rules pass rate:", final_summary["overall_pass_rate"])
    print("Raw ML only pass rate:", raw_summary["overall_pass_rate"])
    print("Rule layer improvement:", round(final_summary["overall_pass_rate"] - raw_summary["overall_pass_rate"], 2), "points")


if __name__ == "__main__":
    main()