import os
import json
import random
from copy import deepcopy


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "eval", "planner_stress_cases.jsonl")

RANDOM_SEED = 42
TARGET_CASE_COUNT = 3500


# ============================================================
# Metadata Templates
# ============================================================

def metadata_template(source_type="csv", has_numeric=True, has_category=True, has_datetime=True, has_text=False):
    return {
        "source_type": source_type,
        "row_count": 10000,
        "column_count": 8,
        "has_numeric": has_numeric,
        "has_category": has_category,
        "has_datetime": has_datetime,
        "has_text": has_text,
    }


METADATA_VARIANTS = [
    metadata_template("csv", True, True, True, False),
    metadata_template("excel", True, True, True, False),
    metadata_template("json", True, True, True, True),
    metadata_template("parquet", True, True, True, False),
    metadata_template("csv", True, True, False, False),
    metadata_template("json", True, True, True, True),
]


# ============================================================
# Case Helper
# ============================================================

def make_case(
    case_id,
    question,
    metadata,
    expected_intent,
    expected_answer_depth,
    expected_operation,
    expected_best_chart,
    expected_chart_required,
    needs_numeric,
    needs_category,
    needs_datetime,
    needs_text,
):
    return {
        "id": f"planner_{case_id:05d}",
        "question": question,
        "metadata": deepcopy(metadata),
        "expected": {
            "intent": expected_intent,
            "answer_depth": expected_answer_depth,
            "operation": expected_operation,
            "best_chart": expected_best_chart,
            "chart_required": expected_chart_required,
            "required_data_roles": {
                "needs_numeric": needs_numeric,
                "needs_category": needs_category,
                "needs_datetime": needs_datetime,
                "needs_text": needs_text,
            },
        },
    }


def add_case(cases, counter, *args, **kwargs):
    cases.append(make_case(counter, *args, **kwargs))
    return counter + 1


# ============================================================
# Template Groups
# ============================================================

def aggregation_questions():
    metrics = [
        "sales", "revenue", "profit", "quantity", "salary", "bonus",
        "treatment cost", "patient count", "spend", "clicks",
        "impressions", "conversion rate", "temperature", "humidity",
        "pressure", "score", "attendance", "expense", "income",
        "budget", "resolution time", "latency", "error count",
        "request count", "rating"
    ]

    templates = [
        ("What is the total {metric}?", "sum"),
        ("Show total {metric}", "sum"),
        ("How much {metric} did we have?", "sum"),
        ("Give me overall {metric}", "sum"),
        ("What is the average {metric}?", "mean"),
        ("Show average {metric}", "mean"),
        ("Mean {metric}", "mean"),
        ("Highest {metric} value", "max"),
        ("Maximum {metric}", "max"),
        ("Lowest {metric} recorded", "min"),
        ("Minimum {metric}", "min"),
    ]

    rows = []

    for metric in metrics:
        for template, operation in templates:
            rows.append(
                {
                    "question": template.format(metric=metric),
                    "intent": "aggregation",
                    "answer_depth": "direct_answer",
                    "operation": operation,
                    "best_chart": "kpi_card",
                    "chart_required": False,
                    "needs_numeric": True,
                    "needs_category": False,
                    "needs_datetime": False,
                    "needs_text": False,
                }
            )

    return rows


def ranking_questions():
    dimensions = [
        "product", "country", "region", "department", "employee",
        "customer", "channel", "campaign", "hospital", "diagnosis",
        "doctor", "service", "endpoint", "status code", "device type",
        "location", "priority", "job role", "class", "subject",
        "student", "account", "expense category"
    ]

    metrics = [
        "sales", "revenue", "profit", "quantity", "salary", "bonus",
        "treatment cost", "patient count", "spend", "clicks",
        "conversion rate", "temperature", "humidity", "pressure",
        "score", "attendance", "expense", "income", "budget",
        "resolution time", "latency", "error count", "request count",
        "rating"
    ]

    templates = [
        "Which {dimension} has the highest {metric}?",
        "Top {dimension} by {metric}",
        "Rank {dimension} by {metric}",
        "Best {dimension} based on {metric}",
        "Which {dimension} generated the most {metric}?",
    ]

    rows = []

    for dimension in dimensions:
        for metric in metrics:
            for template in templates:
                rows.append(
                    {
                        "question": template.format(dimension=dimension, metric=metric),
                        "intent": "ranking",
                        "answer_depth": "small_summary",
                        "operation": "groupby_sum_sort_desc",
                        "best_chart": "horizontal_bar_chart",
                        "chart_required": True,
                        "needs_numeric": True,
                        "needs_category": True,
                        "needs_datetime": False,
                        "needs_text": False,
                    }
                )

    return rows


def comparison_questions():
    dimensions = [
        "product", "country", "region", "department", "channel",
        "campaign", "hospital", "diagnosis", "service", "endpoint",
        "status code", "device type", "location", "priority",
        "job role", "class", "subject", "expense category"
    ]

    metrics = [
        "sales", "revenue", "profit", "quantity", "salary", "bonus",
        "treatment cost", "patient count", "spend", "clicks",
        "impressions", "temperature", "humidity", "pressure",
        "score", "attendance", "expense", "income", "budget",
        "resolution time", "latency", "error count", "request count",
        "rating"
    ]

    templates = [
        "Compare {metric} by {dimension}",
        "Show {metric} by {dimension}",
        "Break down {metric} by {dimension}",
        "{metric} across {dimension}",
        "Group {metric} by {dimension}",
    ]

    rows = []

    for dimension in dimensions:
        for metric in metrics:
            for template in templates:
                rows.append(
                    {
                        "question": template.format(metric=metric, dimension=dimension),
                        "intent": "comparison",
                        "answer_depth": "visual_answer",
                        "operation": "groupby_sum",
                        "best_chart": "bar_chart",
                        "chart_required": True,
                        "needs_numeric": True,
                        "needs_category": True,
                        "needs_datetime": False,
                        "needs_text": False,
                    }
                )

    return rows


def trend_questions():
    metrics = [
        "sales", "revenue", "profit", "quantity", "salary", "bonus",
        "treatment cost", "patient count", "spend", "clicks",
        "impressions", "conversion rate", "temperature", "humidity",
        "pressure", "score", "attendance", "expense", "income",
        "budget", "resolution time", "latency", "error count",
        "request count", "rating"
    ]

    templates = [
        "Show {metric} trend over time",
        "How has {metric} changed over time?",
        "Monthly {metric} trend",
        "Daily {metric} movement",
        "{metric} by month",
        "{metric} over time",
        "Plot {metric} by day",
        "Track {metric} weekly",
        "Show {metric} by hour",
        "Plot {metric} per hour",
    ]

    rows = []

    for metric in metrics:
        for template in templates:
            rows.append(
                {
                    "question": template.format(metric=metric),
                    "intent": "trend_analysis",
                    "answer_depth": "visual_answer",
                    "operation": "time_groupby_sum",
                    "best_chart": "line_chart",
                    "chart_required": True,
                    "needs_numeric": True,
                    "needs_category": False,
                    "needs_datetime": True,
                    "needs_text": False,
                }
            )

    return rows


def correlation_questions():
    metric_pairs = [
        ("sales", "profit"),
        ("revenue", "profit"),
        ("spend", "revenue"),
        ("conversion rate", "revenue"),
        ("clicks", "conversions"),
        ("temperature", "pressure"),
        ("temperature", "humidity"),
        ("humidity", "pressure"),
        ("attendance", "score"),
        ("discount", "profit"),
        ("price", "quantity"),
        ("latency", "error count"),
        ("request count", "latency"),
        ("expense", "income"),
        ("budget", "expense"),
    ]

    templates = [
        "Compare {x} and {y}",
        "Does {x} affect {y}?",
        "Does {x} relate to {y}?",
        "Is {x} related to {y}?",
        "Relationship between {x} and {y}",
        "Show correlation between {x} and {y}",
    ]

    rows = []

    for x, y in metric_pairs:
        for template in templates:
            rows.append(
                {
                    "question": template.format(x=x, y=y),
                    "intent": "correlation",
                    "answer_depth": "visual_answer",
                    "operation": "correlation",
                    "best_chart": "scatter_plot",
                    "chart_required": True,
                    "needs_numeric": True,
                    "needs_category": False,
                    "needs_datetime": False,
                    "needs_text": False,
                }
            )

    return rows


def correlation_heatmap_questions():
    questions = [
        "Show correlation heatmap for numeric columns",
        "Create a heatmap of numeric relationships",
        "Show all numeric correlations",
        "Build a correlation matrix",
        "Show relationships among all metrics",
        "Correlation matrix for financial metrics",
        "Heatmap for sales profit and discount",
        "Show numeric feature correlation",
        "Compare all numeric columns together",
        "Create a numeric correlation matrix",
        "Build heatmap for all measures",
        "Show metric-to-metric relationship heatmap",
    ]

    rows = []

    for question in questions:
        rows.append(
            {
                "question": question,
                "intent": "correlation",
                "answer_depth": "visual_answer",
                "operation": "correlation_heatmap",
                "best_chart": "heatmap",
                "chart_required": True,
                "needs_numeric": True,
                "needs_category": False,
                "needs_datetime": False,
                "needs_text": False,
            }
        )

    return rows


def data_quality_questions():
    questions = [
        "Are there missing values?",
        "Check missing entries",
        "Find incomplete records",
        "Which columns have missing values?",
        "Check for nulls in this file",
        "Any blanks in the dataset?",
        "Find empty values",
        "Show null value counts",
        "Are any fields missing data?",
        "Tell me if the dataset has missing entries",
        "Which columns are incomplete?",
        "Are there duplicate rows?",
        "Check duplicate records",
        "Find duplicate entries",
        "Show data quality summary",
        "Are there errors in the data?",
    ]

    rows = []

    for question in questions:
        if "duplicate" in question.lower():
            operation = "duplicate_check"
        elif "quality" in question.lower() or "errors in the data" in question.lower():
            operation = "data_quality_summary"
        else:
            operation = "null_check"

        rows.append(
            {
                "question": question,
                "intent": "data_quality",
                "answer_depth": "data_quality_answer",
                "operation": operation,
                "best_chart": "table",
                "chart_required": False,
                "needs_numeric": False,
                "needs_category": False,
                "needs_datetime": False,
                "needs_text": False,
            }
        )

    return rows


def distribution_questions():
    metrics = [
        "sales", "revenue", "profit", "salary", "bonus", "treatment cost",
        "spend", "clicks", "temperature", "humidity", "pressure", "score",
        "attendance", "expense", "income", "budget", "latency",
        "resolution time", "rating"
    ]

    rows = []

    for metric in metrics:
        rows.append(
            {
                "question": f"Show distribution of {metric}",
                "intent": "distribution",
                "answer_depth": "visual_answer",
                "operation": "distribution",
                "best_chart": "histogram",
                "chart_required": True,
                "needs_numeric": True,
                "needs_category": False,
                "needs_datetime": False,
                "needs_text": False,
            }
        )

        rows.append(
            {
                "question": f"Are there outliers in {metric}?",
                "intent": "distribution",
                "answer_depth": "visual_answer",
                "operation": "outlier_check",
                "best_chart": "box_plot",
                "chart_required": True,
                "needs_numeric": True,
                "needs_category": False,
                "needs_datetime": False,
                "needs_text": False,
            }
        )

    return rows


def text_analysis_questions():
    text_fields = [
        "reviews",
        "comments",
        "feedback",
        "customer feedback",
        "support tickets",
        "ticket descriptions",
        "log messages",
        "open responses",
        "complaints",
    ]

    rows = []

    summary_templates = [
        "Summarize {field}",
        "What do the {field} say?",
        "Give me a summary of {field}",
        "Read the {field} and summarize them",
        "What is the main theme in the {field}?",
    ]

    sentiment_templates = [
        "Analyze sentiment in {field}",
        "Are {field} mostly positive or negative?",
        "Show sentiment breakdown for {field}",
        "Classify {field} sentiment",
        "Check whether {field} sound negative",
    ]

    word_templates = [
        "What words appear most often in {field}?",
        "Show common words in {field}",
        "Find frequent terms in {field}",
        "What are the top keywords in {field}?",
        "Show word frequency for {field}",
        "Find repeated words in {field}",
        "Show most mentioned topics in {field}",
    ]

    for field in text_fields:
        for template in summary_templates:
            rows.append(
                {
                    "question": template.format(field=field),
                    "intent": "text_analysis",
                    "answer_depth": "small_summary",
                    "operation": "text_summary",
                    "best_chart": "bar_chart",
                    "chart_required": True,
                    "needs_numeric": False,
                    "needs_category": False,
                    "needs_datetime": False,
                    "needs_text": True,
                }
            )

        for template in sentiment_templates:
            rows.append(
                {
                    "question": template.format(field=field),
                    "intent": "text_analysis",
                    "answer_depth": "visual_answer",
                    "operation": "sentiment_summary",
                    "best_chart": "bar_chart",
                    "chart_required": True,
                    "needs_numeric": False,
                    "needs_category": False,
                    "needs_datetime": False,
                    "needs_text": True,
                }
            )

        for template in word_templates:
            rows.append(
                {
                    "question": template.format(field=field),
                    "intent": "text_analysis",
                    "answer_depth": "visual_answer",
                    "operation": "word_frequency",
                    "best_chart": "bar_chart",
                    "chart_required": True,
                    "needs_numeric": False,
                    "needs_category": False,
                    "needs_datetime": False,
                    "needs_text": True,
                }
            )

    return rows


def summary_questions():
    questions = [
        "Do a full analysis of this dataset",
        "Analyze this dataset and give business insights",
        "Give me a full summary of the file",
        "Analyze this file and explain the main patterns",
        "Give me a business overview of this dataset",
        "What are the main insights in this data?",
        "Summarize everything important in this dataset",
        "Create an executive summary from this data",
        "Give me a dashboard-style analysis",
        "Tell me what this dataset is showing",
        "Give me a quick but complete analysis",
    ]

    rows = []

    for question in questions:
        rows.append(
            {
                "question": question,
                "intent": "summary_analysis",
                "answer_depth": "deep_analysis",
                "operation": "full_dataset_analysis",
                "best_chart": "multi_chart_dashboard",
                "chart_required": True,
                "needs_numeric": True,
                "needs_category": True,
                "needs_datetime": True,
                "needs_text": False,
            }
        )

    return rows


def forecasting_questions():
    metrics = [
        "sales", "revenue", "profit", "orders", "clicks", "spend",
        "temperature", "demand", "expense", "income", "request count",
        "patient count"
    ]

    templates = [
        "Forecast future {metric}",
        "Predict next month {metric}",
        "What will {metric} look like in the future?",
        "Forecast upcoming {metric}",
    ]

    rows = []

    for metric in metrics:
        for template in templates:
            rows.append(
                {
                    "question": template.format(metric=metric),
                    "intent": "forecasting",
                    "answer_depth": "deep_analysis",
                    "operation": "forecast",
                    "best_chart": "line_chart",
                    "chart_required": True,
                    "needs_numeric": True,
                    "needs_category": False,
                    "needs_datetime": True,
                    "needs_text": False,
                }
            )

    return rows


def diagnostic_questions():
    questions = [
        "Why did sales drop?",
        "Why did revenue decrease?",
        "Why did profit decline?",
        "Why did request failures spike?",
        "Explain abnormal latency changes",
        "Investigate abnormal error spikes",
        "Why did traffic suddenly increase?",
        "Find anomalies in traffic logs",
        "Detect anomalies in server logs",
        "Find unusual traffic patterns",
        "Why did conversion rate drop?",
        "Why did expenses increase?",
        "Why did patient count decrease?",
    ]

    rows = []

    for question in questions:
        rows.append(
            {
                "question": question,
                "intent": "diagnostic_analysis",
                "answer_depth": "deep_analysis",
                "operation": "diagnostic_analysis",
                "best_chart": "multi_chart_dashboard",
                "chart_required": True,
                "needs_numeric": True,
                "needs_category": True,
                "needs_datetime": True,
                "needs_text": False,
            }
        )

    return rows


# ============================================================
# Build Stress Cases
# ============================================================

def build_cases():
    random.seed(RANDOM_SEED)

    all_rows = []
    all_rows.extend(aggregation_questions())
    all_rows.extend(ranking_questions())
    all_rows.extend(comparison_questions())
    all_rows.extend(trend_questions())
    all_rows.extend(correlation_questions())
    all_rows.extend(correlation_heatmap_questions())
    all_rows.extend(data_quality_questions())
    all_rows.extend(distribution_questions())
    all_rows.extend(text_analysis_questions())
    all_rows.extend(summary_questions())
    all_rows.extend(forecasting_questions())
    all_rows.extend(diagnostic_questions())

    random.shuffle(all_rows)

    cases = []
    counter = 1

    while len(cases) < TARGET_CASE_COUNT:
        row = random.choice(all_rows)
        metadata = random.choice(METADATA_VARIANTS)

        # Make sure text questions get text-capable metadata.
        if row["needs_text"]:
            metadata = metadata_template("json", True, True, True, True)

        # Make sure trend/forecast questions get datetime-capable metadata.
        if row["needs_datetime"]:
            metadata["has_datetime"] = True

        cases.append(
            make_case(
                counter,
                row["question"],
                metadata,
                row["intent"],
                row["answer_depth"],
                row["operation"],
                row["best_chart"],
                row["chart_required"],
                row["needs_numeric"],
                row["needs_category"],
                row["needs_datetime"],
                row["needs_text"],
            )
        )
        counter += 1

    return cases


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    cases = build_cases()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case) + "\n")

    print("\nGenerated planner stress cases.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total cases: {len(cases)}")


if __name__ == "__main__":
    main()