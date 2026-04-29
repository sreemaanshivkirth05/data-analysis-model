import os
import json
import pandas as pd

from predict import predict_plan_raw


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILE = os.path.join(BASE_DIR, "data", "test_questions.csv")


def build_metadata_from_row(row):
    return {
        "source_type": row.get("source_type", "unknown"),
        "has_numeric": str(row.get("has_numeric", "unknown")).lower() == "true",
        "has_category": str(row.get("has_category", "unknown")).lower() == "true",
        "has_datetime": str(row.get("has_datetime", "unknown")).lower() == "true",
        "has_text": str(row.get("has_text", "unknown")).lower() == "true",
    }


def main():
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    df = pd.read_csv(TEST_FILE)

    total = len(df)
    intent_correct = 0
    operation_correct = 0
    chart_correct = 0

    detailed_results = []

    for _, row in df.iterrows():
        question = row["question"]
        metadata = build_metadata_from_row(row)

        prediction = predict_plan_raw(question, metadata)

        expected_intent = row["expected_intent"]
        expected_operation = row["expected_operation"]
        expected_best_chart = row["expected_best_chart"]

        intent_match = prediction["intent"] == expected_intent
        operation_match = prediction["operation"] == expected_operation
        chart_match = prediction["best_chart"] == expected_best_chart

        intent_correct += int(intent_match)
        operation_correct += int(operation_match)
        chart_correct += int(chart_match)

        detailed_results.append(
            {
                "question": question,
                "expected": {
                    "intent": expected_intent,
                    "operation": expected_operation,
                    "best_chart": expected_best_chart,
                },
                "predicted_raw_ml": {
                    "intent": prediction["intent"],
                    "operation": prediction["operation"],
                    "best_chart": prediction["best_chart"],
                },
                "matches": {
                    "intent": intent_match,
                    "operation": operation_match,
                    "best_chart": chart_match,
                },
            }
        )

    print("\n================ RAW ML-ONLY EVALUATION ================\n")
    print("Evaluation type: raw ML model only, no rule correction")
    print(f"Total test questions: {total}")
    print(f"Intent accuracy: {intent_correct}/{total} = {intent_correct / total:.2%}")
    print(f"Operation accuracy: {operation_correct}/{total} = {operation_correct / total:.2%}")
    print(f"Chart accuracy: {chart_correct}/{total} = {chart_correct / total:.2%}")

    print("\n================ DETAILED RESULTS ================\n")
    print(json.dumps(detailed_results, indent=2))


if __name__ == "__main__":
    main()