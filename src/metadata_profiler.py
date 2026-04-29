import os
import json
import warnings
from typing import Dict, Any, List

import pandas as pd


# -----------------------------
# Dataset Loading
# -----------------------------

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from CSV, Excel, JSON, or Parquet into a pandas DataFrame.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    if ext == ".json":
        return pd.read_json(file_path)

    if ext == ".parquet":
        return pd.read_parquet(file_path)

    raise ValueError(
        f"Unsupported file type: {ext}. "
        "Supported formats: .csv, .xlsx, .xls, .json, .parquet"
    )


def get_source_type(file_path: str) -> str:
    """
    Convert file extension into a source type label.
    """

    ext = os.path.splitext(file_path)[1].lower()

    source_map = {
        ".csv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".json": "json",
        ".parquet": "parquet",
    }

    return source_map.get(ext, "unknown")


# -----------------------------
# Basic Helpers
# -----------------------------

def normalize_name(name: str) -> str:
    """
    Normalize a column name for keyword matching.
    """

    return str(name).lower().replace("_", " ").replace("-", " ").strip()


def get_sample_values(series: pd.Series, limit: int = 5) -> List[str]:
    """
    Return clean sample values from a column.
    """

    return series.dropna().astype(str).head(limit).tolist()


def safe_unique_count(series: pd.Series) -> int:
    """
    Safely count unique values.
    """

    try:
        return int(series.nunique(dropna=True))
    except Exception:
        return 0


def get_cardinality_type(series: pd.Series, row_count: int, semantic_type: str) -> str:
    """
    Classify column cardinality.

    Useful for avoiding bad chart dimensions like:
    - customer_id
    - order_id
    - transaction_id
    """

    unique_count = safe_unique_count(series)

    if row_count == 0:
        return "unknown"

    unique_ratio = unique_count / row_count

    if semantic_type == "numeric":
        if unique_count <= 2:
            return "binary_numeric"
        if unique_ratio >= 0.9:
            return "high_unique_numeric"
        return "continuous_or_measure"

    if semantic_type == "datetime":
        return "time"

    if semantic_type == "text":
        return "free_text"

    if semantic_type == "boolean":
        return "boolean"

    # category
    if unique_count <= 2:
        return "binary_category"
    if unique_count <= 20:
        return "low_cardinality_category"
    if unique_ratio >= 0.8:
        return "high_cardinality_category"

    return "medium_cardinality_category"


# -----------------------------
# Semantic Type Detection
# -----------------------------

def looks_like_boolean(series: pd.Series) -> bool:
    """
    Detect boolean-like columns.
    """

    sample = series.dropna().astype(str).str.lower().str.strip()

    if len(sample) == 0:
        return False

    unique_values = set(sample.unique())

    boolean_sets = [
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
        {"0", "1"},
        {"active", "inactive"},
        {"passed", "failed"},
        {"success", "failure"},
    ]

    return any(unique_values.issubset(valid_set) for valid_set in boolean_sets)


def looks_like_datetime(series: pd.Series, column_name: str = "") -> bool:
    """
    Detect date-like columns while reducing pandas warning noise.

    Strategy:
    1. Prefer date-like column names.
    2. Try strict/common date formats first.
    3. Use fallback parser silently only if needed.
    """

    name = normalize_name(column_name)

    date_name_keywords = [
        "date",
        "time",
        "timestamp",
        "created",
        "updated",
        "month",
        "year",
        "day",
        "datetime",
    ]

    name_suggests_date = any(keyword in name for keyword in date_name_keywords)

    sample = series.dropna().astype(str).head(50)

    if len(sample) == 0:
        return False

    # Avoid treating short numeric IDs as dates.
    if pd.api.types.is_numeric_dtype(series) and not name_suggests_date:
        return False

    common_formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]

    for fmt in common_formats:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if parsed.notna().mean() >= 0.7:
            return True

    # Fallback parser, but suppress noisy warnings.
    if name_suggests_date:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")

        return parsed.notna().mean() >= 0.7

    return False


def looks_like_text(series: pd.Series) -> bool:
    """
    Detect free-text columns such as reviews, comments, messages, feedback.
    """

    sample = series.dropna().astype(str).head(50)

    if len(sample) == 0:
        return False

    avg_length = sample.str.len().mean()
    avg_words = sample.str.split().str.len().mean()

    return avg_length >= 50 or avg_words >= 8


def detect_semantic_type(series: pd.Series, column_name: str = "") -> str:
    """
    Detect the semantic type of a column.

    Returns:
    - numeric
    - datetime
    - text
    - category
    - boolean
    """

    if looks_like_boolean(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if looks_like_datetime(series, column_name):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if looks_like_text(series):
        return "text"

    return "category"


# -----------------------------
# Business Type Detection
# -----------------------------

def detect_business_type(column_name: str, series: pd.Series, semantic_type: str) -> str:
    """
    Detect richer business meaning of a column.

    Examples:
    - currency_or_amount
    - percentage
    - identifier
    - geography
    - rating
    - quantity
    - date
    - free_text
    """

    name = normalize_name(column_name)

    id_keywords = [
        "id",
        "uuid",
        "key",
        "code",
        "number",
        "no",
        "identifier",
        "order id",
        "customer id",
        "user id",
        "transaction id",
        "invoice id",
        "product id",
        "employee id",
    ]

    geography_keywords = [
        "country",
        "state",
        "city",
        "region",
        "location",
        "zip",
        "zipcode",
        "postal",
        "territory",
        "province",
        "latitude",
        "longitude",
        "lat",
        "lon",
    ]

    currency_keywords = [
        "sales",
        "revenue",
        "amount",
        "profit",
        "price",
        "cost",
        "income",
        "expense",
        "balance",
        "margin",
        "spend",
        "budget",
        "salary",
        "wage",
        "payment",
        "invoice",
    ]

    percentage_keywords = [
        "percent",
        "percentage",
        "rate",
        "%",
        "ratio",
        "conversion",
        "ctr",
        "cvr",
        "margin rate",
        "attrition rate",
    ]

    quantity_keywords = [
        "quantity",
        "qty",
        "count",
        "orders",
        "units",
        "items",
        "volume",
        "number of",
        "num",
    ]

    rating_keywords = [
        "rating",
        "score",
        "stars",
        "grade",
        "rank",
        "satisfaction",
        "nps",
    ]

    time_keywords = [
        "date",
        "time",
        "timestamp",
        "month",
        "year",
        "day",
        "created",
        "updated",
    ]

    text_keywords = [
        "review",
        "comment",
        "feedback",
        "message",
        "description",
        "ticket",
        "notes",
        "summary",
        "text",
    ]

    if semantic_type == "datetime" or any(keyword in name for keyword in time_keywords):
        return "date_or_time"

    if any(keyword in name for keyword in id_keywords):
        return "identifier"

    if any(keyword in name for keyword in geography_keywords):
        return "geography"

    if any(keyword in name for keyword in percentage_keywords):
        return "percentage"

    if any(keyword in name for keyword in currency_keywords):
        return "currency_or_amount"

    if any(keyword in name for keyword in quantity_keywords):
        return "quantity_or_count"

    if any(keyword in name for keyword in rating_keywords):
        return "rating_or_score"

    if semantic_type == "text" or any(keyword in name for keyword in text_keywords):
        return "free_text"

    if semantic_type == "boolean":
        return "boolean_flag"

    if semantic_type == "numeric":
        return "numeric_measure"

    if semantic_type == "category":
        return "categorical_dimension"

    return "unknown"


# -----------------------------
# Column Role Detection
# -----------------------------

def infer_column_role(
    column_name: str,
    semantic_type: str,
    business_type: str,
    cardinality_type: str,
    row_count: int,
    unique_count: int,
) -> str:
    """
    Infer the role of a column for analysis.

    Returns:
    - measure
    - dimension
    - time
    - text
    - id
    - boolean
    """

    name = normalize_name(column_name)

    if semantic_type == "datetime" or business_type == "date_or_time":
        return "time"

    if business_type == "identifier":
        return "id"

    # High-cardinality categories are often IDs/names.
    # But some names like Customer or Product can still be useful dimensions.
    if cardinality_type == "high_cardinality_category":
        if any(keyword in name for keyword in ["customer", "product", "employee", "vendor", "supplier"]):
            return "dimension"
        return "id"

    if semantic_type == "numeric":
        # Numeric IDs should not be treated as measures.
        if business_type == "identifier":
            return "id"

        return "measure"

    if semantic_type == "text":
        return "text"

    if semantic_type == "boolean":
        return "boolean"

    return "dimension"


# -----------------------------
# Dataset-Level Profiling
# -----------------------------

def profile_dataframe(df: pd.DataFrame, source_type: str = "unknown") -> Dict[str, Any]:
    """
    Create metadata for a pandas DataFrame.
    This metadata is passed into your ML planner and column mapper.
    """

    columns = []

    has_numeric = False
    has_category = False
    has_datetime = False
    has_text = False
    has_boolean = False
    has_geography = False
    has_identifier = False
    has_currency = False
    has_percentage = False

    row_count = int(len(df))

    for column in df.columns:
        series = df[column]

        semantic_type = detect_semantic_type(series, str(column))
        unique_count = safe_unique_count(series)
        cardinality_type = get_cardinality_type(series, row_count, semantic_type)
        business_type = detect_business_type(str(column), series, semantic_type)

        role = infer_column_role(
            column_name=str(column),
            semantic_type=semantic_type,
            business_type=business_type,
            cardinality_type=cardinality_type,
            row_count=row_count,
            unique_count=unique_count,
        )

        if semantic_type == "numeric":
            has_numeric = True
        elif semantic_type == "datetime":
            has_datetime = True
        elif semantic_type == "text":
            has_text = True
        elif semantic_type == "category":
            has_category = True
        elif semantic_type == "boolean":
            has_boolean = True

        if business_type == "geography":
            has_geography = True
        elif business_type == "identifier":
            has_identifier = True
        elif business_type == "currency_or_amount":
            has_currency = True
        elif business_type == "percentage":
            has_percentage = True

        column_metadata = {
            "name": str(column),
            "dtype": str(series.dtype),
            "semantic_type": semantic_type,
            "role": role,
            "business_type": business_type,
            "cardinality_type": cardinality_type,
            "null_count": int(series.isna().sum()),
            "null_percent": round(float(series.isna().mean() * 100), 2),
            "unique_count": unique_count,
            "unique_ratio": round(float(unique_count / row_count), 4) if row_count > 0 else 0.0,
            "sample_values": get_sample_values(series),
        }

        columns.append(column_metadata)

    metadata = {
        "source_type": source_type,
        "row_count": row_count,
        "column_count": int(len(df.columns)),
        "has_numeric": has_numeric,
        "has_category": has_category,
        "has_datetime": has_datetime,
        "has_text": has_text,
        "has_boolean": has_boolean,
        "has_geography": has_geography,
        "has_identifier": has_identifier,
        "has_currency": has_currency,
        "has_percentage": has_percentage,
        "columns": columns,
    }

    return metadata


def profile_file(file_path: str) -> Dict[str, Any]:
    """
    Load a file and return metadata.
    """

    df = load_dataset(file_path)
    source_type = get_source_type(file_path)
    metadata = profile_dataframe(df, source_type=source_type)

    return metadata


# -----------------------------
# Printing Helpers
# -----------------------------

def print_metadata_summary(metadata: Dict[str, Any]) -> None:
    """
    Print a readable summary of dataset metadata.
    """

    print("\n================ DATASET METADATA SUMMARY ================\n")

    print(f"Source Type: {metadata['source_type']}")
    print(f"Rows: {metadata['row_count']}")
    print(f"Columns: {metadata['column_count']}")
    print(f"Has Numeric: {metadata['has_numeric']}")
    print(f"Has Category: {metadata['has_category']}")
    print(f"Has Datetime: {metadata['has_datetime']}")
    print(f"Has Text: {metadata['has_text']}")
    print(f"Has Boolean: {metadata.get('has_boolean')}")
    print(f"Has Geography: {metadata.get('has_geography')}")
    print(f"Has Identifier: {metadata.get('has_identifier')}")
    print(f"Has Currency/Amount: {metadata.get('has_currency')}")
    print(f"Has Percentage: {metadata.get('has_percentage')}")

    print("\n---------------- COLUMN DETAILS ----------------\n")

    for col in metadata["columns"]:
        print(f"Column: {col['name']}")
        print(f"  dtype: {col['dtype']}")
        print(f"  semantic_type: {col['semantic_type']}")
        print(f"  role: {col['role']}")
        print(f"  business_type: {col['business_type']}")
        print(f"  cardinality_type: {col['cardinality_type']}")
        print(f"  null_percent: {col['null_percent']}%")
        print(f"  unique_count: {col['unique_count']}")
        print(f"  unique_ratio: {col['unique_ratio']}")
        print(f"  sample_values: {col['sample_values']}")
        print()


if __name__ == "__main__":
    file_path = input("Enter dataset path: ").strip()

    metadata = profile_file(file_path)

    print_metadata_summary(metadata)

    print("\n================ RAW METADATA JSON ================\n")
    print(json.dumps(metadata, indent=2))