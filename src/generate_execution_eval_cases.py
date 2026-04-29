import os
import json
import random
from copy import deepcopy


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "eval", "execution_eval_cases.jsonl")


RANDOM_SEED = 42
TARGET_CASE_COUNT = 1200


# ============================================================
# Metadata Templates
# ============================================================

def sales_metadata():
    return {
        "source_type": "csv",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": False,
        "columns": [
            {"name": "Order ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Date", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.9},
            {"name": "Country", "semantic_type": "category", "role": "dimension", "business_type": "geography", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Region", "semantic_type": "category", "role": "dimension", "business_type": "geography", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.25},
            {"name": "Product", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.3},
            {"name": "Category", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Customer Name", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "high_cardinality_category", "unique_ratio": 0.9},
            {"name": "Sales", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.8},
            {"name": "Profit", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Quantity", "semantic_type": "numeric", "role": "measure", "business_type": "quantity_or_count", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.6},
            {"name": "Discount", "semantic_type": "numeric", "role": "measure", "business_type": "percentage", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.5},
        ],
    }


def hr_metadata():
    return {
        "source_type": "excel",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": False,
        "columns": [
            {"name": "Employee ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Employee Name", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "high_cardinality_category", "unique_ratio": 0.9},
            {"name": "Department", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.1},
            {"name": "Job Role", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "medium_cardinality_category", "unique_ratio": 0.4},
            {"name": "Location", "semantic_type": "category", "role": "dimension", "business_type": "geography", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Salary", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Bonus", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.5},
            {"name": "Performance Score", "semantic_type": "numeric", "role": "measure", "business_type": "rating_or_score", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.4},
            {"name": "Attrition Rate", "semantic_type": "numeric", "role": "measure", "business_type": "percentage", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.3},
            {"name": "Hire Date", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.9},
        ],
    }


def marketing_metadata():
    return {
        "source_type": "csv",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": False,
        "columns": [
            {"name": "Campaign ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Campaign", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "medium_cardinality_category", "unique_ratio": 0.5},
            {"name": "Channel", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.1},
            {"name": "Region", "semantic_type": "category", "role": "dimension", "business_type": "geography", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Spend", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Revenue", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.8},
            {"name": "Clicks", "semantic_type": "numeric", "role": "measure", "business_type": "quantity_or_count", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Impressions", "semantic_type": "numeric", "role": "measure", "business_type": "quantity_or_count", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.8},
            {"name": "Conversion Rate", "semantic_type": "numeric", "role": "measure", "business_type": "percentage", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.5},
            {"name": "Date", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.9},
        ],
    }


def healthcare_metadata():
    return {
        "source_type": "csv",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": False,
        "columns": [
            {"name": "Patient ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Hospital", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Diagnosis", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "medium_cardinality_category", "unique_ratio": 0.4},
            {"name": "Doctor", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "high_cardinality_category", "unique_ratio": 0.85},
            {"name": "Treatment Cost", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.8},
            {"name": "Patient Count", "semantic_type": "numeric", "role": "measure", "business_type": "quantity_or_count", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.6},
            {"name": "Length of Stay", "semantic_type": "numeric", "role": "measure", "business_type": "numeric_measure", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.5},
            {"name": "Admission Date", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.9},
        ],
    }


def logs_metadata():
    return {
        "source_type": "json",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": True,
        "columns": [
            {"name": "Request ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Service", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Endpoint", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "medium_cardinality_category", "unique_ratio": 0.5},
            {"name": "Status Code", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "medium_cardinality_category", "unique_ratio": 0.4},
            {"name": "Error Count", "semantic_type": "numeric", "role": "measure", "business_type": "quantity_or_count", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.6},
            {"name": "Request Count", "semantic_type": "numeric", "role": "measure", "business_type": "quantity_or_count", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Latency", "semantic_type": "numeric", "role": "measure", "business_type": "numeric_measure", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.8},
            {"name": "Timestamp", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.9},
            {"name": "Log Message", "semantic_type": "text", "role": "text", "business_type": "free_text", "cardinality_type": "free_text", "unique_ratio": 1.0},
        ],
    }


def reviews_metadata():
    return {
        "source_type": "json",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": True,
        "columns": [
            {"name": "Review ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Product", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Customer Segment", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Rating", "semantic_type": "numeric", "role": "measure", "business_type": "rating_or_score", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.3},
            {"name": "Review Text", "semantic_type": "text", "role": "text", "business_type": "free_text", "cardinality_type": "free_text", "unique_ratio": 1.0},
            {"name": "Review Date", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.9},
        ],
    }


def support_metadata():
    return {
        "source_type": "json",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": True,
        "columns": [
            {"name": "Ticket ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Priority", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Category", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Resolution Time", "semantic_type": "numeric", "role": "measure", "business_type": "numeric_measure", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Ticket Description", "semantic_type": "text", "role": "text", "business_type": "free_text", "cardinality_type": "free_text", "unique_ratio": 1.0},
            {"name": "Created At", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.9},
        ],
    }


def iot_metadata():
    return {
        "source_type": "parquet",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": False,
        "columns": [
            {"name": "Device ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Device Type", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Location", "semantic_type": "category", "role": "dimension", "business_type": "geography", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Temperature", "semantic_type": "numeric", "role": "measure", "business_type": "numeric_measure", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.8},
            {"name": "Humidity", "semantic_type": "numeric", "role": "measure", "business_type": "numeric_measure", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Pressure", "semantic_type": "numeric", "role": "measure", "business_type": "numeric_measure", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Timestamp", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.9},
        ],
    }


def finance_metadata():
    return {
        "source_type": "parquet",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": False,
        "columns": [
            {"name": "Transaction ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Account Name", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "high_cardinality_category", "unique_ratio": 0.85},
            {"name": "Expense Category", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Department", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Expense", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Income", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.7},
            {"name": "Budget", "semantic_type": "numeric", "role": "measure", "business_type": "currency_or_amount", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.6},
            {"name": "Transaction Date", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.9},
        ],
    }


def education_metadata():
    return {
        "source_type": "csv",
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": False,
        "columns": [
            {"name": "Student ID", "semantic_type": "category", "role": "id", "business_type": "identifier", "cardinality_type": "high_cardinality_category", "unique_ratio": 1.0},
            {"name": "Student Name", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "high_cardinality_category", "unique_ratio": 0.9},
            {"name": "Class", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.1},
            {"name": "Subject", "semantic_type": "category", "role": "dimension", "business_type": "categorical_dimension", "cardinality_type": "low_cardinality_category", "unique_ratio": 0.2},
            {"name": "Score", "semantic_type": "numeric", "role": "measure", "business_type": "rating_or_score", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.6},
            {"name": "Attendance", "semantic_type": "numeric", "role": "measure", "business_type": "percentage", "cardinality_type": "continuous_or_measure", "unique_ratio": 0.4},
            {"name": "Exam Date", "semantic_type": "datetime", "role": "time", "business_type": "date_or_time", "cardinality_type": "time", "unique_ratio": 0.8},
        ],
    }


# ============================================================
# Case Creation Helpers
# ============================================================

def make_case(case_id, question, metadata, measure=None, dimension=None, time=None, text=None, executable=True):
    return {
        "id": f"exec_{case_id:04d}",
        "question": question,
        "metadata": deepcopy(metadata),
        "expected": {
            "measure_column": measure,
            "dimension_column": dimension,
            "time_column": time,
            "text_column": text,
            "is_executable": executable,
        },
    }


def add_case(cases, counter, question, metadata, measure=None, dimension=None, time=None, text=None, executable=True):
    cases.append(make_case(counter, question, metadata, measure, dimension, time, text, executable))
    return counter + 1


# ============================================================
# Template Builders
# ============================================================

def add_numeric_cases(cases, counter, metadata, measure):
    questions = [
        f"What is the total {measure.lower()}?",
        f"Show total {measure.lower()}",
        f"How much {measure.lower()} did we have?",
        f"What is the average {measure.lower()}?",
        f"Average {measure.lower()}",
        f"Highest {measure.lower()} value",
        f"Lowest {measure.lower()} recorded",
        f"Maximum {measure.lower()}",
        f"Minimum {measure.lower()}",
    ]

    for q in questions:
        counter = add_case(cases, counter, q, metadata, measure=measure)

    return counter


def add_groupby_cases(cases, counter, metadata, measure, dimension):
    questions = [
        f"Compare {measure.lower()} by {dimension.lower()}",
        f"Show {measure.lower()} by {dimension.lower()}",
        f"Break down {measure.lower()} by {dimension.lower()}",
        f"{measure} across {dimension}",
        f"Which {dimension.lower()} has the highest {measure.lower()}?",
        f"Top {dimension.lower()} by {measure.lower()}",
        f"Rank {dimension.lower()} by {measure.lower()}",
    ]

    for q in questions:
        counter = add_case(cases, counter, q, metadata, measure=measure, dimension=dimension)

    return counter


def add_trend_cases(cases, counter, metadata, measure, time_col):
    questions = [
        f"Show {measure.lower()} trend over time",
        f"How has {measure.lower()} changed over time?",
        f"Monthly {measure.lower()} trend",
        f"Daily {measure.lower()} movement",
        f"{measure} by month",
        f"{measure} over time",
        f"Plot {measure.lower()} by day",
        f"Track {measure.lower()} weekly",
    ]

    for q in questions:
        counter = add_case(cases, counter, q, metadata, measure=measure, time=time_col)

    return counter


def add_correlation_cases(cases, counter, metadata, measure_x, measure_y):
    questions = [
        f"Does {measure_x.lower()} affect {measure_y.lower()}?",
        f"Does {measure_x.lower()} relate to {measure_y.lower()}?",
        f"Is {measure_x.lower()} related to {measure_y.lower()}?",
        f"Compare {measure_x.lower()} and {measure_y.lower()}",
        f"Relationship between {measure_x.lower()} and {measure_y.lower()}",
    ]

    for q in questions:
        # For this evaluator, expected primary measure is first-mentioned variable.
        counter = add_case(cases, counter, q, metadata, measure=measure_x)

    return counter


def add_text_cases(cases, counter, metadata, text_col):
    questions = [
        f"Summarize {text_col.lower()}",
        f"What words appear in {text_col.lower()}?",
        f"Frequent words in {text_col.lower()}",
        f"Find common keywords in {text_col.lower()}",
        f"What topics are mentioned most in {text_col.lower()}?",
        f"Analyze sentiment in {text_col.lower()}",
        f"Are {text_col.lower()} mostly positive or negative?",
    ]

    for q in questions:
        counter = add_case(cases, counter, q, metadata, text=text_col)

    return counter


def add_text_grouping_cases(cases, counter, metadata, text_col, dimension):
    questions = [
        f"Analyze sentiment in {text_col.lower()} by {dimension.lower()}",
        f"Show topics in {text_col.lower()} by {dimension.lower()}",
        f"Common words in {text_col.lower()} by {dimension.lower()}",
    ]

    for q in questions:
        counter = add_case(cases, counter, q, metadata, dimension=dimension, text=text_col)

    return counter


def add_data_quality_cases(cases, counter, metadata):
    questions = [
        "Are there missing values?",
        "Check missing entries",
        "Find incomplete records",
        "Are there duplicate rows?",
        "Check duplicate records",
        "Show data quality summary",
    ]

    for q in questions:
        counter = add_case(cases, counter, q, metadata)

    return counter


def add_domain_special_cases(cases, counter):
    # Sales
    sales = sales_metadata()
    counter = add_case(cases, counter, "Give me company-wide sales", sales, measure="Sales")
    counter = add_case(cases, counter, "Top products by revenue", sales, measure="Sales", dimension="Product")
    counter = add_case(cases, counter, "Which customer has the highest revenue?", sales, measure="Sales", dimension="Customer Name")
    counter = add_case(cases, counter, "Does discount affect profit?", sales, measure="Discount")

    # HR
    hr = hr_metadata()
    counter = add_case(cases, counter, "Which employee has the highest bonus?", hr, measure="Bonus", dimension="Employee Name")
    counter = add_case(cases, counter, "Which employee has the highest salary?", hr, measure="Salary", dimension="Employee Name")
    counter = add_case(cases, counter, "Salary by department", hr, measure="Salary", dimension="Department")
    counter = add_case(cases, counter, "Performance score by job role", hr, measure="Performance Score", dimension="Job Role")

    # Marketing
    marketing = marketing_metadata()
    counter = add_case(cases, counter, "Does spend affect revenue?", marketing, measure="Spend")
    counter = add_case(cases, counter, "Does conversion rate relate to revenue?", marketing, measure="Conversion Rate")
    counter = add_case(cases, counter, "Which channel drove the most revenue?", marketing, measure="Revenue", dimension="Channel")

    # Logs
    logs = logs_metadata()
    counter = add_case(cases, counter, "Find anomalies in traffic logs", logs, measure="Request Count", dimension="Service", time="Timestamp")
    counter = add_case(cases, counter, "Errors by hour", logs, measure="Error Count", time="Timestamp")
    counter = add_case(cases, counter, "Plot latency by hour", logs, measure="Latency", time="Timestamp")
    counter = add_case(cases, counter, "Request count by endpoint", logs, measure="Request Count", dimension="Endpoint")
    counter = add_case(cases, counter, "What words appear in log messages?", logs, text="Log Message")

    return counter


# ============================================================
# Build Cases
# ============================================================

def build_cases():
    random.seed(RANDOM_SEED)

    cases = []
    counter = 1

    datasets = [
        {
            "metadata": sales_metadata(),
            "measures": ["Sales", "Profit", "Quantity", "Discount"],
            "dimensions": ["Country", "Region", "Product", "Category", "Customer Name"],
            "time": "Date",
            "text": None,
        },
        {
            "metadata": hr_metadata(),
            "measures": ["Salary", "Bonus", "Performance Score", "Attrition Rate"],
            "dimensions": ["Department", "Job Role", "Location", "Employee Name"],
            "time": "Hire Date",
            "text": None,
        },
        {
            "metadata": marketing_metadata(),
            "measures": ["Spend", "Revenue", "Clicks", "Impressions", "Conversion Rate"],
            "dimensions": ["Campaign", "Channel", "Region"],
            "time": "Date",
            "text": None,
        },
        {
            "metadata": healthcare_metadata(),
            "measures": ["Treatment Cost", "Patient Count", "Length of Stay"],
            "dimensions": ["Hospital", "Diagnosis", "Doctor"],
            "time": "Admission Date",
            "text": None,
        },
        {
            "metadata": logs_metadata(),
            "measures": ["Error Count", "Request Count", "Latency"],
            "dimensions": ["Service", "Endpoint", "Status Code"],
            "time": "Timestamp",
            "text": "Log Message",
        },
        {
            "metadata": reviews_metadata(),
            "measures": ["Rating"],
            "dimensions": ["Product", "Customer Segment"],
            "time": "Review Date",
            "text": "Review Text",
        },
        {
            "metadata": support_metadata(),
            "measures": ["Resolution Time"],
            "dimensions": ["Priority", "Category"],
            "time": "Created At",
            "text": "Ticket Description",
        },
        {
            "metadata": iot_metadata(),
            "measures": ["Temperature", "Humidity", "Pressure"],
            "dimensions": ["Device Type", "Location"],
            "time": "Timestamp",
            "text": None,
        },
        {
            "metadata": finance_metadata(),
            "measures": ["Expense", "Income", "Budget"],
            "dimensions": ["Account Name", "Expense Category", "Department"],
            "time": "Transaction Date",
            "text": None,
        },
        {
            "metadata": education_metadata(),
            "measures": ["Score", "Attendance"],
            "dimensions": ["Student Name", "Class", "Subject"],
            "time": "Exam Date",
            "text": None,
        },
    ]

    # Systematic coverage
    for dataset in datasets:
        metadata = dataset["metadata"]
        measures = dataset["measures"]
        dimensions = dataset["dimensions"]
        time_col = dataset["time"]
        text_col = dataset["text"]

        for measure in measures:
            counter = add_numeric_cases(cases, counter, metadata, measure)
            counter = add_trend_cases(cases, counter, metadata, measure, time_col)

        for measure in measures:
            for dimension in dimensions:
                counter = add_groupby_cases(cases, counter, metadata, measure, dimension)

        for idx, measure_x in enumerate(measures):
            for measure_y in measures[idx + 1:]:
                counter = add_correlation_cases(cases, counter, metadata, measure_x, measure_y)

        if text_col:
            counter = add_text_cases(cases, counter, metadata, text_col)

            for dimension in dimensions:
                counter = add_text_grouping_cases(cases, counter, metadata, text_col, dimension)

        counter = add_data_quality_cases(cases, counter, metadata)

    # Known hard examples
    counter = add_domain_special_cases(cases, counter)

    # Add randomized paraphrase cases until target is reached.
    while len(cases) < TARGET_CASE_COUNT:
        dataset = random.choice(datasets)
        metadata = dataset["metadata"]
        measures = dataset["measures"]
        dimensions = dataset["dimensions"]
        time_col = dataset["time"]
        text_col = dataset["text"]

        case_type = random.choice([
            "numeric",
            "groupby",
            "trend",
            "correlation",
            "data_quality",
            "text" if text_col else "groupby",
        ])

        if case_type == "numeric":
            measure = random.choice(measures)
            template = random.choice([
                "What is the total {measure}?",
                "Show average {measure}",
                "Lowest {measure} recorded",
                "Highest {measure} value",
                "Maximum {measure}",
                "Minimum {measure}",
            ])
            question = template.format(measure=measure.lower())
            counter = add_case(cases, counter, question, metadata, measure=measure)

        elif case_type == "groupby":
            measure = random.choice(measures)
            dimension = random.choice(dimensions)
            template = random.choice([
                "Show {measure} by {dimension}",
                "Compare {measure} by {dimension}",
                "Break down {measure} by {dimension}",
                "Which {dimension} has the highest {measure}?",
                "Top {dimension} by {measure}",
            ])
            question = template.format(
                measure=measure.lower(),
                dimension=dimension.lower(),
            )
            counter = add_case(cases, counter, question, metadata, measure=measure, dimension=dimension)

        elif case_type == "trend":
            measure = random.choice(measures)
            template = random.choice([
                "Show {measure} over time",
                "Monthly {measure} trend",
                "Plot {measure} by day",
                "{measure} trend by month",
                "Track {measure} weekly",
            ])
            question = template.format(measure=measure.lower())
            counter = add_case(cases, counter, question, metadata, measure=measure, time=time_col)

        elif case_type == "correlation":
            if len(measures) >= 2:
                measure_x, measure_y = random.sample(measures, 2)
                template = random.choice([
                    "Does {x} affect {y}?",
                    "Does {x} relate to {y}?",
                    "Compare {x} and {y}",
                    "Relationship between {x} and {y}",
                ])
                question = template.format(x=measure_x.lower(), y=measure_y.lower())
                counter = add_case(cases, counter, question, metadata, measure=measure_x)

        elif case_type == "data_quality":
            question = random.choice([
                "Are there missing values?",
                "Find incomplete records",
                "Check duplicate rows",
                "Show data quality summary",
            ])
            counter = add_case(cases, counter, question, metadata)

        elif case_type == "text" and text_col:
            template = random.choice([
                "Summarize {text_col}",
                "What words appear in {text_col}?",
                "Frequent words in {text_col}",
                "Analyze sentiment in {text_col}",
                "Common topics in {text_col}",
            ])
            question = template.format(text_col=text_col.lower())
            counter = add_case(cases, counter, question, metadata, text=text_col)

    return cases[:TARGET_CASE_COUNT]


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    cases = build_cases()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case) + "\n")

    print("\nGenerated execution evaluation cases.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total cases: {len(cases)}")


if __name__ == "__main__":
    main()