import os
import json
import datetime
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from predict import predict_plan_raw, predict_plan, CONFIDENCE_THRESHOLD


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILE = os.path.join(BASE_DIR, "data", "test_questions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


REQUIRED_COLUMNS = [
    "question",
    "source_type",
    "has_numeric",
    "has_category",
    "has_datetime",
    "has_text",
    "expected_intent",
    "expected_operation",
    "expected_best_chart",
]


def load_test_questions() -> pd.DataFrame:
    """
    Load test_questions.csv and validate required columns.

    This prevents errors like:
    KeyError: 'question'

    That usually happens when:
    - the CSV header is missing
    - the header is misspelled
    - the header has extra spaces
    - the file was pasted without the first header row
    """

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    df = pd.read_csv(TEST_FILE)

    # Clean column names:
    # - remove UTF-8 BOM if present
    # - remove leading/trailing spaces
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("\n================ CSV HEADER ERROR ================\n")
        print("Your data/test_questions.csv file does not have the expected columns.")
        print("\nMissing columns:")
        print(missing_columns)

        print("\nColumns found in your file:")
        print(df.columns.tolist())

        print("\nThe first line of data/test_questions.csv must be exactly:\n")
        print(
            "question,source_type,has_numeric,has_category,has_datetime,"
            "has_text,expected_intent,expected_operation,expected_best_chart"
        )

        print("\nExample:\n")
        print(
            'question,source_type,has_numeric,has_category,has_datetime,'
            'has_text,expected_intent,expected_operation,expected_best_chart'
        )
        print(
            '"What are the sales of the company?",csv,true,true,true,false,'
            'aggregation,sum,kpi_card'
        )

        raise ValueError(
            f"Missing required columns in test_questions.csv: {missing_columns}"
        )

    return df


def build_metadata_from_row(row: pd.Series) -> dict:
    """
    Build metadata dictionary from one row of test_questions.csv.
    """

    return {
        "source_type": row.get("source_type", "unknown"),
        "has_numeric": str(row.get("has_numeric", "unknown")).lower().strip() == "true",
        "has_category": str(row.get("has_category", "unknown")).lower().strip() == "true",
        "has_datetime": str(row.get("has_datetime", "unknown")).lower().strip() == "true",
        "has_text": str(row.get("has_text", "unknown")).lower().strip() == "true",
    }


def collect_predictions(use_rules: bool = False):
    """
    Collect expected and predicted labels.

    use_rules=False:
        Evaluates raw ML model only.

    use_rules=True:
        Evaluates final planner output after rule correction.
    """

    df = load_test_questions()

    y_true = {
        "intent": [],
        "operation": [],
        "best_chart": [],
    }

    y_pred = {
        "intent": [],
        "operation": [],
        "best_chart": [],
    }

    detailed_rows = []

    for _, row in df.iterrows():
        question = str(row["question"])
        metadata = build_metadata_from_row(row)

        if use_rules:
            prediction = predict_plan(question, metadata)
        else:
            prediction = predict_plan_raw(question, metadata)

        expected_intent = str(row["expected_intent"]).strip()
        expected_operation = str(row["expected_operation"]).strip()
        expected_best_chart = str(row["expected_best_chart"]).strip()

        predicted_intent = prediction["intent"]
        predicted_operation = prediction["operation"]
        predicted_best_chart = prediction["best_chart"]

        y_true["intent"].append(expected_intent)
        y_true["operation"].append(expected_operation)
        y_true["best_chart"].append(expected_best_chart)

        y_pred["intent"].append(predicted_intent)
        y_pred["operation"].append(predicted_operation)
        y_pred["best_chart"].append(predicted_best_chart)

        detailed_rows.append(
            {
                "question": question,
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
                "expected_operation": expected_operation,
                "predicted_operation": predicted_operation,
                "expected_best_chart": expected_best_chart,
                "predicted_best_chart": predicted_best_chart,
                "intent_match": expected_intent == predicted_intent,
                "operation_match": expected_operation == predicted_operation,
                "chart_match": expected_best_chart == predicted_best_chart,
                "confidence_scores": prediction.get("confidence_scores", {}),
                "ml_flag": prediction.get("ml_flag", "unknown"),
                "rules_fired": prediction.get("rules_fired", []),
            }
        )

    return y_true, y_pred, detailed_rows


def get_class_support(y_true_values, labels):
    """
    Count how many true examples exist for each class.
    """

    support = {}

    for label in labels:
        support[label] = sum(1 for value in y_true_values if value == label)

    return support


def save_confusion_matrix(
    y_true,
    y_pred,
    label_name: str,
    output_file: str,
):
    """
    Save a confusion matrix image for one target label.

    Adds support count to each class label, for example:
    aggregation (n=11)
    """

    labels = sorted(
        list(set(y_true[label_name]) | set(y_pred[label_name]))
    )

    cm = confusion_matrix(
        y_true[label_name],
        y_pred[label_name],
        labels=labels,
    )

    support = get_class_support(y_true[label_name], labels)

    display_labels = [
        f"{label}\n(n={support[label]})"
        for label in labels
    ]

    figure_width = max(11, len(labels) * 1.1)
    figure_height = max(9, len(labels) * 0.9)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )

    ax.set_title(f"Confusion Matrix: {label_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved: {output_file}")


def save_misclassifications(detailed_rows, output_file: str):
    """
    Save all wrong predictions into a CSV file.
    """

    df = pd.DataFrame(detailed_rows)

    mistakes = df[
        (df["intent_match"] == False)
        | (df["operation_match"] == False)
        | (df["chart_match"] == False)
    ]

    mistakes.to_csv(output_file, index=False)

    print(f"Saved mistakes CSV: {output_file}")
    print(f"Total mistakes: {len(mistakes)}")


def save_run_summary(detailed_rows: list, timestamp: str, mode: str):
    """
    Save a JSON file summarising confidence scores and rule firing counts for the run.
    """
    total = len(detailed_rows)
    high_conf = sum(1 for r in detailed_rows if r.get("ml_flag") == "high_confidence")
    low_conf = sum(1 for r in detailed_rows if r.get("ml_flag") == "low_confidence_fallback")

    low_conf_field_counts: Dict[str, int] = {}
    for row in detailed_rows:
        for field, score in row.get("confidence_scores", {}).items():
            if score < CONFIDENCE_THRESHOLD:
                low_conf_field_counts[field] = low_conf_field_counts.get(field, 0) + 1

    rule_fire_counts: Dict[str, int] = {}
    for row in detailed_rows:
        for rule in row.get("rules_fired", []):
            rule_fire_counts[rule] = rule_fire_counts.get(rule, 0) + 1

    summary = {
        "timestamp": timestamp,
        "mode": mode,
        "total_predictions": total,
        "high_confidence": high_conf,
        "low_confidence_routed_to_llm": low_conf,
        "low_confidence_pct": round(low_conf / total * 100, 1) if total else 0,
        "low_confidence_fields": dict(
            sorted(low_conf_field_counts.items(), key=lambda x: -x[1])
        ),
        "rule_fire_counts": dict(
            sorted(rule_fire_counts.items(), key=lambda x: -x[1])
        ),
    }

    output_file = os.path.join(OUTPUT_DIR, f"{timestamp}_{mode}_run_summary.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved run summary: {output_file}")
    print(f"  High confidence: {high_conf}/{total}")
    print(f"  Low confidence (→ LLM): {low_conf}/{total}")
    print(f"  Rules fired: {rule_fire_counts}")


def save_full_predictions(detailed_rows, output_file: str):
    """
    Save every prediction, not only mistakes.
    Useful for comparing evaluation runs.
    """

    df = pd.DataFrame(detailed_rows)
    df.to_csv(output_file, index=False)

    print(f"Saved full predictions CSV: {output_file}")


def print_class_support_summary(y_true, mode: str):
    """
    Print sample counts for each class so weak evaluation areas are visible.
    """

    print(f"\n================ CLASS SUPPORT SUMMARY: {mode} ================\n")

    for label_name in ["intent", "operation", "best_chart"]:
        labels = sorted(set(y_true[label_name]))
        support = get_class_support(y_true[label_name], labels)

        print(f"\n--- {label_name} ---")

        for label, count in sorted(support.items(), key=lambda item: item[1]):
            warning = "  <-- low support" if count < 10 else ""
            print(f"{label}: {count}{warning}")


def run_confusion_matrix_evaluation(use_rules: bool = False):
    """
    Main function.

    Generates confusion matrices and CSVs for either:
    - raw ML-only predictions
    - final planner predictions after rule correction
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mode = "final_with_rules" if use_rules else "raw_ml_only"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n================ CONFUSION MATRIX EVALUATION ================\n")
    print(f"Mode: {mode}")
    print(f"Timestamp: {timestamp}")

    y_true, y_pred, detailed_rows = collect_predictions(use_rules=use_rules)

    print_class_support_summary(y_true, mode)

    save_confusion_matrix(
        y_true,
        y_pred,
        label_name="intent",
        output_file=os.path.join(
            OUTPUT_DIR,
            f"{timestamp}_{mode}_intent_confusion_matrix.png",
        ),
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        label_name="operation",
        output_file=os.path.join(
            OUTPUT_DIR,
            f"{timestamp}_{mode}_operation_confusion_matrix.png",
        ),
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        label_name="best_chart",
        output_file=os.path.join(
            OUTPUT_DIR,
            f"{timestamp}_{mode}_chart_confusion_matrix.png",
        ),
    )

    save_misclassifications(
        detailed_rows,
        output_file=os.path.join(
            OUTPUT_DIR,
            f"{timestamp}_{mode}_misclassifications.csv",
        ),
    )

    save_full_predictions(
        detailed_rows,
        output_file=os.path.join(
            OUTPUT_DIR,
            f"{timestamp}_{mode}_all_predictions.csv",
        ),
    )

    save_run_summary(detailed_rows, timestamp, mode)

    print("\nDone.")


if __name__ == "__main__":
    run_confusion_matrix_evaluation(use_rules=False)
    run_confusion_matrix_evaluation(use_rules=True)