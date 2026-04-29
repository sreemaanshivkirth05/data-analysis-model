import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_FILE = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "planner_model.pkl")


TARGET_COLUMNS = [
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


METADATA_COLUMNS = [
    "source_type",
    "has_numeric",
    "has_category",
    "has_datetime",
    "has_text",
]


def normalize_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    bool_columns = [
        "chart_required",
        "needs_numeric",
        "needs_category",
        "needs_datetime",
        "needs_text",
        "has_numeric",
        "has_category",
        "has_datetime",
        "has_text",
    ]

    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    return df


def add_missing_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the code backward-compatible.
    If metadata columns are missing, it fills them with 'unknown'.
    """

    for col in METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = "unknown"

    return df


def build_training_input(row: pd.Series) -> str:
    """
    Build the same style of input used during prediction.

    This helps the model learn not only from the question, but also from
    dataset-level metadata such as whether numeric/date/text columns exist.
    """

    return (
        str(row["question"])
        + f" source_type:{row.get('source_type', 'unknown')}"
        + f" has_numeric:{row.get('has_numeric', 'unknown')}"
        + f" has_category:{row.get('has_category', 'unknown')}"
        + f" has_datetime:{row.get('has_datetime', 'unknown')}"
        + f" has_text:{row.get('has_text', 'unknown')}"
    )


def validate_training_data(df: pd.DataFrame) -> None:
    required_columns = ["question"] + TARGET_COLUMNS
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if len(df) < 30:
        raise ValueError("Training data is too small. Add more examples before training.")


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\n================ TRAINING DATA SUMMARY ================\n")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nIntent distribution:")
    print(df["intent"].value_counts())

    print("\nBest chart distribution:")
    print(df["best_chart"].value_counts())


def train_model():
    if not os.path.exists(TRAINING_FILE):
        raise FileNotFoundError(f"Training file not found: {TRAINING_FILE}")

    df = pd.read_csv(TRAINING_FILE)
    df = add_missing_metadata_columns(df)
    df = normalize_bool_columns(df)

    validate_training_data(df)
    print_dataset_summary(df)

    X = df.apply(build_training_input, axis=1)
    y = df[TARGET_COLUMNS].astype(str)

    stratify_col = df["intent"] if df["intent"].value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 3),
                    max_features=10000,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                MultiOutputClassifier(
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=2.0,
                    )
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n================ MODEL EVALUATION ================\n")

    for index, target in enumerate(TARGET_COLUMNS):
        print(f"\n--- {target} ---")
        print(classification_report(y_test[target], y_pred[:, index], zero_division=0))

    artifact = {
        "model": model,
        "target_columns": TARGET_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "model_version": "v2_metadata_aware_tfidf_logreg",
    }

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(artifact, file)

    print("\n================ TRAINING COMPLETE ================\n")
    print(f"Model saved at: {MODEL_FILE}")
    print("Model version: v2_metadata_aware_tfidf_logreg")


if __name__ == "__main__":
    train_model()