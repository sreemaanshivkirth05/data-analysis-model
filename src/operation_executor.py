import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


def load_dataset_for_execution(file_path: str) -> pd.DataFrame:
    """
    Load a dataset file for operation execution.

    Supports:
    - CSV
    - Excel .xlsx / .xls
    - JSON

    This function is used by test_full_pipeline.py.
    """

    if not file_path:
        raise ValueError("Dataset path is empty.")

    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".csv"):
        return pd.read_csv(file_path)

    if file_path_lower.endswith(".xlsx") or file_path_lower.endswith(".xls"):
        return pd.read_excel(file_path)

    if file_path_lower.endswith(".json"):
        return pd.read_json(file_path)

    raise ValueError(
        f"Unsupported file type for execution: {file_path}. "
        "Supported formats are .csv, .xlsx, .xls, and .json."
    )

# ============================================================
# Helpers
# ============================================================

def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric safely.
    Non-numeric values become NaN.
    """

    return pd.to_numeric(series, errors="coerce")


def ensure_column_exists(df: pd.DataFrame, column: Optional[str], role: str) -> None:
    """
    Raise a clear error if a required column is missing.
    """

    if not column:
        raise ValueError(f"Missing required {role} column.")

    if column not in df.columns:
        raise ValueError(f"{role} column '{column}' not found in dataframe.")


def safe_round(value, digits: int = 2):
    """
    Round numeric values safely for JSON output.
    """

    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return value


def records_to_json_safe(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert NaN / numpy values to JSON-safe values.
    """

    cleaned = []

    for row in records:
        clean_row = {}

        for key, value in row.items():
            if pd.isna(value):
                clean_row[key] = None
            elif isinstance(value, (np.integer,)):
                clean_row[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean_row[key] = round(float(value), 2)
            else:
                clean_row[key] = value

        cleaned.append(clean_row)

    return cleaned


def get_selected_columns(mapped_plan: Dict[str, Any]) -> Dict[str, Any]:
    return mapped_plan.get("selected_columns", {})


def get_measure_column(mapped_plan: Dict[str, Any]) -> Optional[str]:
    return get_selected_columns(mapped_plan).get("measure_column")


def get_dimension_column(mapped_plan: Dict[str, Any]) -> Optional[str]:
    return get_selected_columns(mapped_plan).get("dimension_column")


def get_time_column(mapped_plan: Dict[str, Any]) -> Optional[str]:
    return get_selected_columns(mapped_plan).get("time_column")


def get_text_column(mapped_plan: Dict[str, Any]) -> Optional[str]:
    return get_selected_columns(mapped_plan).get("text_column")


# ============================================================
# Scalar Operations
# ============================================================

def execute_sum(df: pd.DataFrame, measure_column: str) -> Dict[str, Any]:
    ensure_column_exists(df, measure_column, "measure")

    values = clean_numeric_series(df[measure_column])
    result = safe_round(values.sum())

    return {
        "result_type": "scalar",
        "value": result,
        "measure_column": measure_column,
        "operation": "sum",
        "summary": f"The total {measure_column} is {result}."
    }


def execute_mean(df: pd.DataFrame, measure_column: str) -> Dict[str, Any]:
    ensure_column_exists(df, measure_column, "measure")

    values = clean_numeric_series(df[measure_column])
    result = safe_round(values.mean())

    return {
        "result_type": "scalar",
        "value": result,
        "measure_column": measure_column,
        "operation": "mean",
        "summary": f"The average {measure_column} is {result}."
    }


def execute_max(df: pd.DataFrame, measure_column: str) -> Dict[str, Any]:
    ensure_column_exists(df, measure_column, "measure")

    values = clean_numeric_series(df[measure_column])
    result = safe_round(values.max())

    return {
        "result_type": "scalar",
        "value": result,
        "measure_column": measure_column,
        "operation": "max",
        "summary": f"The maximum {measure_column} is {result}."
    }


def execute_min(df: pd.DataFrame, measure_column: str) -> Dict[str, Any]:
    ensure_column_exists(df, measure_column, "measure")

    values = clean_numeric_series(df[measure_column])
    result = safe_round(values.min())

    return {
        "result_type": "scalar",
        "value": result,
        "measure_column": measure_column,
        "operation": "min",
        "summary": f"The minimum {measure_column} is {result}."
    }


# ============================================================
# Group-by Operations
# ============================================================

def execute_groupby_sum(
    df: pd.DataFrame,
    measure_column: str,
    dimension_column: str,
    sort_desc: bool = False,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Sum a numeric measure by a categorical dimension.

    Used for:
    - sales by region
    - total revenue by country
    - quantity by product
    """

    ensure_column_exists(df, measure_column, "measure")
    ensure_column_exists(df, dimension_column, "dimension")

    working_df = df[[dimension_column, measure_column]].copy()
    working_df[measure_column] = clean_numeric_series(working_df[measure_column])

    grouped = (
        working_df
        .groupby(dimension_column, dropna=False)[measure_column]
        .sum()
        .reset_index()
    )

    grouped[measure_column] = grouped[measure_column].round(2)

    if sort_desc:
        grouped = grouped.sort_values(by=measure_column, ascending=False)

    grouped = grouped.head(top_n)

    records = records_to_json_safe(grouped.to_dict(orient="records"))

    if records:
        top = records[0]
        summary = (
            f"The top {dimension_column} by total {measure_column} "
            f"is {top[dimension_column]} with {top[measure_column]}."
        )
    else:
        summary = f"Calculated total {measure_column} by {dimension_column}."

    return {
        "result_type": "table",
        "dimension_column": dimension_column,
        "measure_column": measure_column,
        "aggregation": "sum",
        "records": records,
        "summary": summary,
    }


def execute_groupby_mean(
    df: pd.DataFrame,
    measure_column: str,
    dimension_column: str,
    sort_desc: bool = False,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Average a numeric measure by a categorical dimension.

    Used for:
    - average MonthlyIncome by Department
    - average MonthlyIncome by JobRole
    - average DailyRate by Hotel
    - average Rating by Product

    This is important because salary/income comparisons should usually
    use average, not sum.
    """

    ensure_column_exists(df, measure_column, "measure")
    ensure_column_exists(df, dimension_column, "dimension")

    working_df = df[[dimension_column, measure_column]].copy()
    working_df[measure_column] = clean_numeric_series(working_df[measure_column])

    grouped = (
        working_df
        .groupby(dimension_column, dropna=False)[measure_column]
        .mean()
        .reset_index()
    )

    grouped[measure_column] = grouped[measure_column].round(2)

    if sort_desc:
        grouped = grouped.sort_values(by=measure_column, ascending=False)

    grouped = grouped.head(top_n)

    records = records_to_json_safe(grouped.to_dict(orient="records"))

    if records:
        top = records[0]
        summary = (
            f"The top {dimension_column} by average {measure_column} "
            f"is {top[dimension_column]} with {top[measure_column]}."
        )
    else:
        summary = f"Calculated average {measure_column} by {dimension_column}."

    return {
        "result_type": "table",
        "dimension_column": dimension_column,
        "measure_column": measure_column,
        "aggregation": "mean",
        "records": records,
        "summary": summary,
    }


# ============================================================
# Time Series Operations
# ============================================================

def execute_time_groupby_sum(
    df: pd.DataFrame,
    measure_column: str,
    time_column: str,
    time_grain: str = "month",
) -> Dict[str, Any]:
    ensure_column_exists(df, measure_column, "measure")
    ensure_column_exists(df, time_column, "time")

    working_df = df[[time_column, measure_column]].copy()
    working_df[time_column] = pd.to_datetime(working_df[time_column], errors="coerce")
    working_df[measure_column] = clean_numeric_series(working_df[measure_column])

    working_df = working_df.dropna(subset=[time_column])

    if working_df.empty:
        raise ValueError(
            f"Could not parse valid datetime values from time column '{time_column}'."
        )

    if time_grain == "hour":
        working_df["period"] = working_df[time_column].dt.to_period("H").astype(str)
    elif time_grain == "day":
        working_df["period"] = working_df[time_column].dt.to_period("D").astype(str)
    elif time_grain == "week":
        working_df["period"] = working_df[time_column].dt.to_period("W").astype(str)
    elif time_grain == "year":
        working_df["period"] = working_df[time_column].dt.to_period("Y").astype(str)
    elif time_grain == "quarter":
        working_df["period"] = working_df[time_column].dt.to_period("Q").astype(str)
    else:
        working_df["period"] = working_df[time_column].dt.to_period("M").astype(str)
        time_grain = "month"

    grouped = (
        working_df
        .groupby("period", dropna=False)[measure_column]
        .sum()
        .reset_index()
        .sort_values("period")
    )

    grouped[measure_column] = grouped[measure_column].round(2)

    records = records_to_json_safe(grouped.to_dict(orient="records"))

    return {
        "result_type": "time_series",
        "time_column": time_column,
        "measure_column": measure_column,
        "time_grain": time_grain,
        "records": records,
        "summary": f"Generated {time_grain}-level trend for {measure_column}.",
    }


# ============================================================
# Distribution Operations
# ============================================================

def execute_distribution(
    df: pd.DataFrame,
    measure_column: str,
    bins: int = 10,
) -> Dict[str, Any]:
    ensure_column_exists(df, measure_column, "measure")

    values = clean_numeric_series(df[measure_column]).dropna()

    if values.empty:
        raise ValueError(f"No valid numeric values found in {measure_column}.")

    stats = {
        "count": int(values.count()),
        "mean": safe_round(values.mean()),
        "median": safe_round(values.median()),
        "std": safe_round(values.std()),
        "min": safe_round(values.min()),
        "max": safe_round(values.max()),
        "q1": safe_round(values.quantile(0.25)),
        "q3": safe_round(values.quantile(0.75)),
    }

    counts, bin_edges = np.histogram(values, bins=bins)

    histogram_bins = []

    for i in range(len(counts)):
        start = safe_round(bin_edges[i])
        end = safe_round(bin_edges[i + 1])
        label = f"{start} - {end}"

        histogram_bins.append(
            {
                "bin": label,
                "start": start,
                "end": end,
                "count": int(counts[i]),
            }
        )

    return {
        "result_type": "distribution",
        "measure_column": measure_column,
        "statistics": stats,
        "bins": histogram_bins,
        "summary": f"Generated distribution statistics for {measure_column}.",
    }


def execute_outlier_check(df: pd.DataFrame, measure_column: str) -> Dict[str, Any]:
    ensure_column_exists(df, measure_column, "measure")

    values = clean_numeric_series(df[measure_column]).dropna()

    if values.empty:
        raise ValueError(f"No valid numeric values found in {measure_column}.")

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = values[(values < lower_bound) | (values > upper_bound)]

    return {
        "result_type": "outlier_check",
        "measure_column": measure_column,
        "outlier_count": int(outliers.count()),
        "lower_bound": safe_round(lower_bound),
        "upper_bound": safe_round(upper_bound),
        "summary": (
            f"Found {int(outliers.count())} potential outliers in {measure_column} "
            f"using the IQR method."
        ),
    }


# ============================================================
# Data Quality Operations
# ============================================================

def execute_null_check(df: pd.DataFrame) -> Dict[str, Any]:
    records = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_percent = round((missing_count / len(df)) * 100, 2) if len(df) else 0

        records.append(
            {
                "column": col,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }
        )

    records = sorted(records, key=lambda row: row["missing_count"], reverse=True)

    return {
        "result_type": "data_quality",
        "operation": "null_check",
        "records": records,
        "summary": "Generated missing value summary for all columns.",
    }


def execute_duplicate_check(df: pd.DataFrame) -> Dict[str, Any]:
    duplicate_count = int(df.duplicated().sum())

    return {
        "result_type": "data_quality",
        "operation": "duplicate_check",
        "duplicate_count": duplicate_count,
        "row_count": int(len(df)),
        "summary": f"Found {duplicate_count} duplicate rows out of {len(df)} total rows.",
    }


def execute_data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    missing_records = execute_null_check(df)["records"]
    duplicate_count = int(df.duplicated().sum())

    return {
        "result_type": "data_quality",
        "operation": "data_quality_summary",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_count": duplicate_count,
        "missing_values": missing_records,
        "summary": "Generated data quality summary including missing values and duplicate rows.",
    }


# ============================================================
# Correlation Operations
# ============================================================

def execute_correlation(df: pd.DataFrame, measure_column: str) -> Dict[str, Any]:
    ensure_column_exists(df, measure_column, "measure")

    numeric_df = df.select_dtypes(include=["number"]).copy()

    if measure_column not in numeric_df.columns:
        numeric_df[measure_column] = clean_numeric_series(df[measure_column])

    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for correlation analysis.")

    correlations = (
        numeric_df
        .corr(numeric_only=True)[measure_column]
        .drop(labels=[measure_column], errors="ignore")
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    records = [
        {
            "column": col,
            "correlation": safe_round(value, 4),
        }
        for col, value in correlations.items()
    ]

    return {
        "result_type": "correlation",
        "measure_column": measure_column,
        "records": records,
        "summary": f"Generated correlation summary for {measure_column}.",
    }


def execute_correlation_heatmap(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for correlation heatmap.")

    corr = numeric_df.corr(numeric_only=True).round(4)

    records = []

    for row_col in corr.index:
        for col in corr.columns:
            records.append(
                {
                    "x": col,
                    "y": row_col,
                    "value": safe_round(corr.loc[row_col, col], 4),
                }
            )

    return {
        "result_type": "correlation_heatmap",
        "columns": list(corr.columns),
        "records": records,
        "summary": "Generated correlation heatmap for numeric columns.",
    }


# ============================================================
# Text Operations
# ============================================================

def execute_text_summary(df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
    ensure_column_exists(df, text_column, "text")

    sample_text = (
        df[text_column]
        .dropna()
        .astype(str)
        .head(10)
        .tolist()
    )

    return {
        "result_type": "text_summary",
        "text_column": text_column,
        "sample_text": sample_text,
        "summary": f"Collected sample text from {text_column} for summarization.",
    }


def execute_word_frequency(df: pd.DataFrame, text_column: str, top_n: int = 20) -> Dict[str, Any]:
    ensure_column_exists(df, text_column, "text")

    text = " ".join(df[text_column].dropna().astype(str).tolist()).lower()

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)

    stopwords = {
        "the", "and", "for", "with", "this", "that", "from", "are",
        "was", "were", "you", "your", "have", "has", "had", "but",
        "not", "all", "can", "our", "out", "use", "used"
    }

    filtered_words = [word for word in words if word not in stopwords]

    counts = pd.Series(filtered_words).value_counts().head(top_n)

    records = [
        {
            "word": word,
            "count": int(count),
        }
        for word, count in counts.items()
    ]

    return {
        "result_type": "word_frequency",
        "text_column": text_column,
        "records": records,
        "summary": f"Generated word frequency summary for {text_column}.",
    }


def execute_sentiment_summary(df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
    ensure_column_exists(df, text_column, "text")

    return {
        "result_type": "sentiment_summary",
        "text_column": text_column,
        "summary": (
            f"Sentiment summary requires an NLP sentiment model. "
            f"Text column selected: {text_column}."
        ),
    }


# ============================================================
# Full Dataset / Diagnostic Placeholder Operations
# ============================================================

def execute_full_dataset_analysis(
    df: pd.DataFrame,
    measure_column: Optional[str] = None,
    dimension_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> Dict[str, Any]:
    profile = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=["number"]).columns.tolist(),
    }

    result = {
        "result_type": "full_dataset_analysis",
        "profile": profile,
        "measure_column": measure_column,
        "dimension_column": dimension_column,
        "time_column": time_column,
        "summary": "Generated high-level dataset analysis profile.",
    }

    if measure_column and measure_column in df.columns:
        result["measure_summary"] = execute_distribution(df, measure_column)["statistics"]

    if measure_column and dimension_column and measure_column in df.columns and dimension_column in df.columns:
        result["category_breakdown"] = execute_groupby_mean(
            df,
            measure_column=measure_column,
            dimension_column=dimension_column,
            sort_desc=True,
            top_n=10,
        )["records"]

    return result


def execute_diagnostic_analysis(
    df: pd.DataFrame,
    measure_column: Optional[str] = None,
    dimension_column: Optional[str] = None,
) -> Dict[str, Any]:
    result = {
        "result_type": "diagnostic_analysis",
        "measure_column": measure_column,
        "dimension_column": dimension_column,
        "summary": "Generated diagnostic analysis placeholder.",
    }

    if measure_column and dimension_column:
        result["breakdown"] = execute_groupby_mean(
            df,
            measure_column=measure_column,
            dimension_column=dimension_column,
            sort_desc=True,
            top_n=10,
        )["records"]

    return result


# ============================================================
# Main Executor
# ============================================================

def execute_operation(df: pd.DataFrame, mapped_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main operation executor.

    Expected inputs:
    - df: pandas DataFrame
    - mapped_plan: planner output after column mapping
    """

    operation = mapped_plan.get("operation")
    time_grain = mapped_plan.get("time_grain", "month")

    measure_column = get_measure_column(mapped_plan)
    dimension_column = get_dimension_column(mapped_plan)
    time_column = get_time_column(mapped_plan)
    text_column = get_text_column(mapped_plan)

    if not mapped_plan.get("is_executable", True):
        return {
            "result_type": "not_executable",
            "operation": operation,
            "summary": "This plan is not executable.",
            "validation_messages": mapped_plan.get("validation_messages", []),
        }

    if operation == "sum":
        return execute_sum(df, measure_column)

    if operation == "mean":
        return execute_mean(df, measure_column)

    if operation == "max":
        return execute_max(df, measure_column)

    if operation == "min":
        return execute_min(df, measure_column)

    if operation == "groupby_sum":
        return execute_groupby_sum(
            df,
            measure_column=measure_column,
            dimension_column=dimension_column,
            sort_desc=False,
        )

    if operation == "groupby_sum_sort_desc":
        return execute_groupby_sum(
            df,
            measure_column=measure_column,
            dimension_column=dimension_column,
            sort_desc=True,
        )

    if operation == "groupby_mean":
        return execute_groupby_mean(
            df,
            measure_column=measure_column,
            dimension_column=dimension_column,
            sort_desc=False,
        )

    if operation == "groupby_mean_sort_desc":
        return execute_groupby_mean(
            df,
            measure_column=measure_column,
            dimension_column=dimension_column,
            sort_desc=True,
        )

    if operation == "time_groupby_sum":
        return execute_time_groupby_sum(
            df,
            measure_column=measure_column,
            time_column=time_column,
            time_grain=time_grain,
        )

    if operation == "distribution":
        return execute_distribution(df, measure_column)

    if operation == "outlier_check":
        return execute_outlier_check(df, measure_column)

    if operation == "null_check":
        return execute_null_check(df)

    if operation == "duplicate_check":
        return execute_duplicate_check(df)

    if operation == "data_quality_summary":
        return execute_data_quality_summary(df)

    if operation == "correlation":
        return execute_correlation(df, measure_column)

    if operation == "correlation_heatmap":
        return execute_correlation_heatmap(df)

    if operation == "text_summary":
        return execute_text_summary(df, text_column)

    if operation == "word_frequency":
        return execute_word_frequency(df, text_column)

    if operation == "sentiment_summary":
        return execute_sentiment_summary(df, text_column)

    if operation == "full_dataset_analysis":
        return execute_full_dataset_analysis(
            df,
            measure_column=measure_column,
            dimension_column=dimension_column,
            time_column=time_column,
        )

    if operation == "diagnostic_analysis":
        return execute_diagnostic_analysis(
            df,
            measure_column=measure_column,
            dimension_column=dimension_column,
        )

    return {
        "result_type": "unsupported_operation",
        "operation": operation,
        "summary": f"Unsupported operation: {operation}",
    }


# ============================================================
# Compatibility Aliases
# ============================================================

def execute_plan(df: pd.DataFrame, mapped_plan: Dict[str, Any]) -> Dict[str, Any]:
    return execute_operation(df, mapped_plan)


def run_operation(df: pd.DataFrame, mapped_plan: Dict[str, Any]) -> Dict[str, Any]:
    return execute_operation(df, mapped_plan)


def execute(df: pd.DataFrame, mapped_plan: Dict[str, Any]) -> Dict[str, Any]:
    return execute_operation(df, mapped_plan)