import os
import csv
import random


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "training_data.csv")


DOMAINS = {
    "sales": {
        "measures": ["sales", "revenue", "amount", "profit"],
        "dimensions": ["product", "country", "region", "customer", "sales person"],
        "time_terms": ["month", "date", "year", "quarter"],
        "source_type": "csv",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "false",
    },
    "hr": {
        "measures": ["salary", "bonus", "performance score", "attrition rate"],
        "dimensions": ["department", "job role", "gender", "location", "manager"],
        "time_terms": ["hire date", "month", "year"],
        "source_type": "excel",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "false",
    },
    "healthcare": {
        "measures": ["patient count", "age", "treatment cost", "length of stay"],
        "dimensions": ["diagnosis", "department", "doctor", "hospital", "gender"],
        "time_terms": ["admission date", "month", "year"],
        "source_type": "csv",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "false",
    },
    "education": {
        "measures": ["score", "grade", "attendance", "pass rate"],
        "dimensions": ["student", "class", "subject", "teacher", "school"],
        "time_terms": ["exam date", "semester", "year"],
        "source_type": "csv",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "false",
    },
    "finance": {
        "measures": ["expense", "income", "profit", "loss", "balance"],
        "dimensions": ["category", "account", "department", "vendor", "region"],
        "time_terms": ["transaction date", "month", "quarter", "year"],
        "source_type": "parquet",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "false",
    },
    "ecommerce": {
        "measures": ["orders", "revenue", "cart value", "discount", "quantity"],
        "dimensions": ["product", "customer", "category", "channel", "country"],
        "time_terms": ["order date", "month", "week", "year"],
        "source_type": "csv",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "false",
    },
    "marketing": {
        "measures": ["clicks", "impressions", "conversion rate", "spend", "revenue"],
        "dimensions": ["campaign", "channel", "audience", "region", "ad group"],
        "time_terms": ["date", "week", "month", "quarter"],
        "source_type": "csv",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "false",
    },
    "logs": {
        "measures": ["error count", "response time", "request count", "latency"],
        "dimensions": ["service", "endpoint", "status code", "user type", "region"],
        "time_terms": ["timestamp", "hour", "day", "week"],
        "source_type": "json",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "true",
    },
    "iot": {
        "measures": ["temperature", "humidity", "pressure", "sensor value", "speed"],
        "dimensions": ["device", "location", "sensor type", "machine", "zone"],
        "time_terms": ["timestamp", "hour", "day", "month"],
        "source_type": "parquet",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "false",
    },
    "reviews": {
        "measures": ["rating", "review score", "sentiment score"],
        "dimensions": ["product", "customer", "category", "region"],
        "time_terms": ["review date", "month", "year"],
        "source_type": "json",
        "has_numeric": "true",
        "has_category": "true",
        "has_datetime": "true",
        "has_text": "true",
    },
}


HEADER = [
    "question",
    "source_type",
    "has_numeric",
    "has_category",
    "has_datetime",
    "has_text",
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


def metadata(domain):
    return [
        domain["source_type"],
        domain["has_numeric"],
        domain["has_category"],
        domain["has_datetime"],
        domain["has_text"],
    ]


def add_row(rows, question, domain, intent, answer_depth, operation, best_chart,
            chart_required, needs_numeric, needs_category, needs_datetime, needs_text):
    rows.append([
        question,
        *metadata(domain),
        intent,
        answer_depth,
        operation,
        best_chart,
        chart_required,
        needs_numeric,
        needs_category,
        needs_datetime,
        needs_text,
    ])


def generate_schema_questions(rows, domain):
    questions = [
        "How many rows are there?",
        "How many records are in this dataset?",
        "What is the row count?",
        "Count the records",
        "How large is this dataset?",
    ]

    for q in questions:
        add_row(
            rows, q, domain,
            "schema_question", "direct_answer", "count_rows", "kpi_card",
            "false", "false", "false", "false", "false"
        )

    questions = [
        "What columns are available?",
        "Show me the column names",
        "What fields are present?",
        "Show the schema",
        "List all columns",
    ]

    for q in questions:
        add_row(
            rows, q, domain,
            "schema_question", "direct_answer", "list_columns", "table",
            "false", "false", "false", "false", "false"
        )


def generate_aggregation_questions(rows, domain):
    for measure in domain["measures"]:
        questions = [
            f"What is the total {measure}?",
            f"Calculate total {measure}",
            f"What is the overall {measure}?",
            f"How much {measure} do we have?",
            f"Show total {measure}",
        ]

        for q in questions:
            add_row(
                rows, q, domain,
                "aggregation", "direct_answer", "sum", "kpi_card",
                "false", "true", "false", "false", "false"
            )

        questions = [
            f"What is the average {measure}?",
            f"Calculate average {measure}",
            f"What is the mean {measure}?",
        ]

        for q in questions:
            add_row(
                rows, q, domain,
                "aggregation", "direct_answer", "mean", "kpi_card",
                "false", "true", "false", "false", "false"
            )

        questions = [
            f"What is the maximum {measure}?",
            f"What is the highest {measure} value?",
            f"What is the minimum {measure}?",
            f"What is the lowest {measure} value?",
        ]

        for q in questions:
            operation = "max" if "maximum" in q.lower() or "highest" in q.lower() else "min"
            add_row(
                rows, q, domain,
                "aggregation", "direct_answer", operation, "kpi_card",
                "false", "true", "false", "false", "false"
            )


def generate_ranking_questions(rows, domain):
    for measure in domain["measures"]:
        for dimension in domain["dimensions"]:
            questions = [
                f"Which {dimension} has the highest {measure}?",
                f"Top 10 {dimension}s by {measure}",
                f"Rank {dimension}s by total {measure}",
                f"Which {dimension} performed best?",
                f"Show top {dimension}s by {measure}",
            ]

            for q in questions:
                add_row(
                    rows, q, domain,
                    "ranking", "small_summary", "groupby_sum_sort_desc",
                    "horizontal_bar_chart",
                    "true", "true", "true", "false", "false"
                )


def generate_comparison_questions(rows, domain):
    for measure in domain["measures"]:
        for dimension in domain["dimensions"]:
            questions = [
                f"Compare {measure} by {dimension}",
                f"Show {measure} by {dimension}",
                f"Break down {measure} by {dimension}",
                f"Group {measure} by {dimension}",
                f"Compare {measure} across {dimension}s",
            ]

            for q in questions:
                add_row(
                    rows, q, domain,
                    "comparison", "visual_answer", "groupby_sum", "bar_chart",
                    "true", "true", "true", "false", "false"
                )


def generate_trend_questions(rows, domain):
    for measure in domain["measures"]:
        questions = [
            f"Show {measure} trend over time",
            f"Show monthly {measure} trend",
            f"How has {measure} changed over time?",
            f"Plot {measure} by month",
            f"Show yearly {measure} trend",
            f"Show {measure} by date",
        ]

        for q in questions:
            add_row(
                rows, q, domain,
                "trend_analysis", "visual_answer", "time_groupby_sum",
                "line_chart",
                "true", "true", "false", "true", "false"
            )


def generate_distribution_questions(rows, domain):
    for measure in domain["measures"]:
        questions = [
            f"Show distribution of {measure}",
            f"What is the distribution of {measure}?",
            f"Show spread of {measure}",
            f"Show histogram of {measure}",
            f"Show range of {measure} values",
        ]

        for q in questions:
            add_row(
                rows, q, domain,
                "distribution", "visual_answer", "distribution", "histogram",
                "true", "true", "false", "false", "false"
            )

        questions = [
            f"Are there outliers in {measure}?",
            f"Find outliers in {measure}",
            f"Detect unusual {measure} values",
        ]

        for q in questions:
            add_row(
                rows, q, domain,
                "distribution", "small_summary", "outlier_check", "box_plot",
                "true", "true", "false", "false", "false"
            )


def generate_correlation_questions(rows, domain):
    measures = domain["measures"]

    if len(measures) < 2:
        return

    for i in range(min(len(measures) - 1, 3)):
        m1 = measures[i]
        m2 = measures[i + 1]

        questions = [
            f"Compare {m1} and {m2}",
            f"Is there a relationship between {m1} and {m2}?",
            f"Are {m1} and {m2} related?",
            f"Show correlation between {m1} and {m2}",
        ]

        for q in questions:
            add_row(
                rows, q, domain,
                "correlation", "visual_answer", "correlation", "scatter_plot",
                "true", "true", "false", "false", "false"
            )

    add_row(
        rows,
        "Show correlation between numeric columns",
        domain,
        "correlation",
        "visual_answer",
        "correlation_heatmap",
        "heatmap",
        "true",
        "true",
        "false",
        "false",
        "false",
    )


def generate_data_quality_questions(rows, domain):
    questions = [
        "Are there missing values?",
        "Show missing values in the dataset",
        "Check null values",
        "Are there blank values?",
        "Find missing data",
    ]

    for q in questions:
        add_row(
            rows, q, domain,
            "data_quality", "data_quality_answer", "null_check", "table",
            "false", "false", "false", "false", "false"
        )

    questions = [
        "Are there duplicate records?",
        "Find duplicate rows",
        "Check duplicates",
        "Does this dataset have duplicate entries?",
    ]

    for q in questions:
        add_row(
            rows, q, domain,
            "data_quality", "data_quality_answer", "duplicate_check", "table",
            "false", "false", "false", "false", "false"
        )

    questions = [
        "Check data quality",
        "Show data quality summary",
        "Are there invalid values?",
        "Check if this data is clean",
    ]

    for q in questions:
        add_row(
            rows, q, domain,
            "data_quality", "data_quality_answer", "data_quality_summary", "table",
            "false", "false", "false", "false", "false"
        )


def generate_summary_questions(rows, domain):
    questions = [
        "Analyze this dataset",
        "Give me a summary of this dataset",
        "Give business insights from this data",
        "Create a full analysis report",
        "Find key insights",
        "Do a full analysis of this dataset",
        "Give me complete business insights",
        "Summarize the dataset with charts and insights",
    ]

    for q in questions:
        add_row(
            rows, q, domain,
            "summary_analysis", "deep_analysis", "full_dataset_analysis",
            "multi_chart_dashboard",
            "true", "true", "true", "true", "false"
        )


def generate_forecasting_questions(rows, domain):
    for measure in domain["measures"][:3]:
        questions = [
            f"Forecast future {measure}",
            f"Predict next month {measure}",
            f"Show future trend for {measure}",
            f"Forecast {measure} for next quarter",
        ]

        for q in questions:
            add_row(
                rows, q, domain,
                "forecasting", "deep_analysis", "forecast", "line_chart",
                "true", "true", "false", "true", "false"
            )


def generate_diagnostic_questions(rows, domain):
    for measure in domain["measures"][:3]:
        questions = [
            f"Why did {measure} drop?",
            f"Explain {measure} decline",
            f"What caused the spike in {measure}?",
            f"Why did {measure} decrease?",
            f"Find reason for {measure} increase",
        ]

        for q in questions:
            add_row(
                rows, q, domain,
                "diagnostic_analysis", "deep_analysis", "diagnostic_analysis",
                "multi_chart_dashboard",
                "true", "true", "true", "true", "false"
            )


def generate_text_questions(rows, domain):
    if domain["has_text"] != "true":
        return

    questions = [
        "Analyze customer feedback",
        "Summarize customer reviews",
        "What are customers saying?",
        "Find sentiment in reviews",
        "Show most common words in feedback",
        "Analyze support tickets",
        "Summarize comments",
    ]

    for q in questions:
        operation = "text_summary"
        answer_depth = "small_summary"

        if "sentiment" in q.lower():
            operation = "sentiment_summary"
            answer_depth = "visual_answer"

        if "common words" in q.lower():
            operation = "word_frequency"
            answer_depth = "visual_answer"

        add_row(
            rows, q, domain,
            "text_analysis", answer_depth, operation, "bar_chart",
            "true", "false", "true", "false", "true"
        )


def generate_noisy_examples(rows):
    noisy = [
        ["what is the saleas of the company", "sales", "aggregation", "direct_answer", "sum", "kpi_card", "false", "true", "false", "false", "false"],
        ["which produt has the highest sales", "sales", "ranking", "small_summary", "groupby_sum_sort_desc", "horizontal_bar_chart", "true", "true", "true", "false", "false"],
        ["do analysisss for thiss datasetssss", "sales", "summary_analysis", "deep_analysis", "full_dataset_analysis", "multi_chart_dashboard", "true", "true", "true", "true", "false"],
        ["show monthy revenu trend", "sales", "trend_analysis", "visual_answer", "time_groupby_sum", "line_chart", "true", "true", "false", "true", "false"],
        ["are there missng values", "sales", "data_quality", "data_quality_answer", "null_check", "table", "false", "false", "false", "false", "false"],
        ["who sold the most", "sales", "ranking", "small_summary", "groupby_sum_sort_desc", "horizontal_bar_chart", "true", "true", "true", "false", "false"],
        ["Find anomalies in traffic logs", "logs", "diagnostic_analysis", "deep_analysis", "diagnostic_analysis", "multi_chart_dashboard", "true", "true", "true", "true", "false"],
        ["Detect anomalies in server logs", "logs", "diagnostic_analysis", "deep_analysis", "diagnostic_analysis", "multi_chart_dashboard", "true", "true", "true", "true", "false"],
        ["Find unusual traffic patterns", "logs", "diagnostic_analysis", "deep_analysis", "diagnostic_analysis", "multi_chart_dashboard", "true", "true", "true", "true", "false"],
        ["Why did request failures spike?", "logs", "diagnostic_analysis", "deep_analysis", "diagnostic_analysis", "multi_chart_dashboard", "true", "true", "true", "true", "false"],
        ["Explain abnormal latency changes", "logs", "diagnostic_analysis", "deep_analysis", "diagnostic_analysis", "multi_chart_dashboard", "true", "true", "true", "true", "false"],
        ["Investigate abnormal error spikes", "logs", "diagnostic_analysis", "deep_analysis", "diagnostic_analysis", "multi_chart_dashboard", "true", "true", "true", "true", "false"],
        ["Why did traffic suddenly increase?", "logs", "diagnostic_analysis", "deep_analysis", "diagnostic_analysis", "multi_chart_dashboard", "true", "true", "true", "true", "false"],
    ]

    for item in noisy:
        question, domain_name, intent, answer_depth, operation, chart, chart_required, needs_numeric, needs_category, needs_datetime, needs_text = item
        domain = DOMAINS[domain_name]
        add_row(
            rows,
            question,
            domain,
            intent,
            answer_depth,
            operation,
            chart,
            chart_required,
            needs_numeric,
            needs_category,
            needs_datetime,
            needs_text,
        )


def generate_training_data():
    rows = []

    for domain_name, domain in DOMAINS.items():
        generate_schema_questions(rows, domain)
        generate_aggregation_questions(rows, domain)
        generate_ranking_questions(rows, domain)
        generate_comparison_questions(rows, domain)
        generate_trend_questions(rows, domain)
        generate_distribution_questions(rows, domain)
        generate_correlation_questions(rows, domain)
        generate_data_quality_questions(rows, domain)
        generate_summary_questions(rows, domain)
        generate_forecasting_questions(rows, domain)
        generate_diagnostic_questions(rows, domain)
        generate_text_questions(rows, domain)

    generate_noisy_examples(rows)

    random.shuffle(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"Generated training data: {OUTPUT_FILE}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    generate_training_data()