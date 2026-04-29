from typing import Dict, Any, List, Optional


def build_chart_config(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Main chart builder.

    This converts execution results into Apache ECharts-compatible configs.

    Important:
    This does NOT replace your existing VisualizationAgent.
    Later, in your main project, this output can be passed to your existing
    VisualizationAgent as a recommendation or starter ECharts config.
    """

    chart_required = mapped_plan.get("chart_required", False)
    chart_type = mapped_plan.get("best_chart", "table")

    if not chart_required:
        return build_no_chart_response(result, mapped_plan)

    if chart_type == "kpi_card":
        return build_kpi_card(result, mapped_plan)

    if chart_type == "bar_chart":
        return build_bar_chart(result, mapped_plan)

    if chart_type == "horizontal_bar_chart":
        return build_horizontal_bar_chart(result, mapped_plan)

    if chart_type == "line_chart":
        return build_line_chart(result, mapped_plan)

    if chart_type == "area_chart":
        return build_area_chart(result, mapped_plan)

    if chart_type == "histogram":
        return build_histogram_placeholder(result, mapped_plan)

    if chart_type == "box_plot":
        return build_boxplot_placeholder(result, mapped_plan)

    if chart_type == "scatter_plot":
        return build_scatter_placeholder(result, mapped_plan)

    if chart_type in ["heatmap", "correlation_heatmap"]:
        return build_heatmap_placeholder(result, mapped_plan)

    if chart_type == "multi_chart_dashboard":
        return build_multi_chart_dashboard(result, mapped_plan)

    return build_table_chart(result, mapped_plan)


def build_no_chart_response(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Used when the planner says no chart is required.
    Example: row count, total sales, missing values.
    """

    result_type = result.get("result_type")

    if result_type == "scalar":
        return build_kpi_card(result, mapped_plan)

    if result_type in ["data_quality", "data_quality_summary"]:
        return build_table_chart(result, mapped_plan)

    return {
        "chart_required": False,
        "chart_type": "none",
        "echarts_config": None,
        "summary": result.get("summary", "No chart required for this answer."),
    }


def build_kpi_card(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    KPI card output.
    This is not a standard ECharts chart, but your frontend can render it as a card.
    """

    selected = mapped_plan.get("selected_columns", {})
    measure_column = selected.get("measure_column") or result.get("measure_column")

    return {
        "chart_required": False,
        "chart_type": "kpi_card",
        "kpi": {
            "label": measure_column or "Value",
            "value": result.get("value"),
            "summary": result.get("summary"),
        },
        "echarts_config": None,
    }


def build_bar_chart(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Vertical bar chart for category comparison.
    Example: Compare sales by country.
    """

    records = result.get("records", [])

    dimension_column = result.get("dimension_column")
    measure_column = result.get("measure_column")

    if not records or not dimension_column or not measure_column:
        return build_table_chart(result, mapped_plan)

    categories = [record.get(dimension_column) for record in records]
    values = [record.get(measure_column) for record in records]

    config = {
        "title": {
            "text": f"{measure_column} by {dimension_column}",
            "left": "center",
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "shadow",
            },
        },
        "grid": {
            "left": "8%",
            "right": "5%",
            "bottom": "12%",
            "containLabel": True,
        },
        "xAxis": {
            "type": "category",
            "data": categories,
            "axisLabel": {
                "rotate": 30,
            },
        },
        "yAxis": {
            "type": "value",
        },
        "series": [
            {
                "name": measure_column,
                "type": "bar",
                "data": values,
                "label": {
                    "show": True,
                    "position": "top",
                },
            }
        ],
    }

    return {
        "chart_required": True,
        "chart_type": "bar_chart",
        "echarts_config": config,
        "summary": result.get("summary"),
    }


def build_horizontal_bar_chart(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Horizontal bar chart for rankings.
    Example: Which product has the highest sales?
    """

    records = result.get("records", [])

    dimension_column = result.get("dimension_column")
    measure_column = result.get("measure_column")

    if not records or not dimension_column or not measure_column:
        return build_table_chart(result, mapped_plan)

    # Reverse for nicer horizontal display: highest appears at top visually.
    reversed_records = list(reversed(records))

    categories = [record.get(dimension_column) for record in reversed_records]
    values = [record.get(measure_column) for record in reversed_records]

    config = {
        "title": {
            "text": f"Top {dimension_column} by {measure_column}",
            "left": "center",
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "shadow",
            },
        },
        "grid": {
            "left": "15%",
            "right": "8%",
            "bottom": "8%",
            "containLabel": True,
        },
        "xAxis": {
            "type": "value",
        },
        "yAxis": {
            "type": "category",
            "data": categories,
        },
        "series": [
            {
                "name": measure_column,
                "type": "bar",
                "data": values,
                "label": {
                    "show": True,
                    "position": "right",
                },
            }
        ],
    }

    return {
        "chart_required": True,
        "chart_type": "horizontal_bar_chart",
        "echarts_config": config,
        "summary": result.get("summary"),
    }


def build_line_chart(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Line chart for time-series trend.
    Example: Show monthly revenue trend.
    """

    records = result.get("records", [])

    time_column = result.get("time_column")
    measure_column = result.get("measure_column")
    time_grain = result.get("time_grain", mapped_plan.get("time_grain", "month"))

    if not records or not measure_column:
        return build_table_chart(result, mapped_plan)

    x_values = [record.get("period") for record in records]
    y_values = [record.get(measure_column) for record in records]

    config = {
        "title": {
            "text": f"{measure_column} Trend by {time_grain.title()}",
            "left": "center",
        },
        "tooltip": {
            "trigger": "axis",
        },
        "grid": {
            "left": "8%",
            "right": "5%",
            "bottom": "12%",
            "containLabel": True,
        },
        "xAxis": {
            "type": "category",
            "data": x_values,
            "boundaryGap": False,
        },
        "yAxis": {
            "type": "value",
        },
        "series": [
            {
                "name": measure_column,
                "type": "line",
                "data": y_values,
                "smooth": True,
                "symbol": "circle",
                "symbolSize": 8,
                "label": {
                    "show": True,
                },
            }
        ],
    }

    return {
        "chart_required": True,
        "chart_type": "line_chart",
        "echarts_config": config,
        "summary": result.get("summary"),
        "meta": {
            "time_column": time_column,
            "measure_column": measure_column,
            "time_grain": time_grain,
        },
    }


def build_area_chart(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Area chart for time-series magnitude.
    """

    line_output = build_line_chart(result, mapped_plan)

    if not line_output.get("echarts_config"):
        return line_output

    config = line_output["echarts_config"]
    config["series"][0]["areaStyle"] = {}

    line_output["chart_type"] = "area_chart"
    return line_output


def build_table_chart(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Table-style fallback output.
    This is useful for exact values and data quality results.
    """

    records = result.get("records")

    if records is None:
        records = []

    columns = []

    if records and isinstance(records, list) and isinstance(records[0], dict):
        columns = list(records[0].keys())

    return {
        "chart_required": False,
        "chart_type": "table",
        "table": {
            "columns": columns,
            "records": records,
        },
        "echarts_config": None,
        "summary": result.get("summary"),
    }


def build_histogram_placeholder(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Placeholder for histogram.

    Your current operation_executor returns distribution statistics,
    not raw bins. Later, we can add bin generation in operation_executor.
    """

    statistics = result.get("statistics", {})

    return {
        "chart_required": True,
        "chart_type": "histogram",
        "echarts_config": None,
        "statistics": statistics,
        "summary": result.get("summary"),
        "note": "Histogram config needs binned distribution data. Add bin generation in operation_executor for full histogram rendering.",
    }


def build_boxplot_placeholder(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Placeholder for box plot / outlier output.
    """

    return {
        "chart_required": True,
        "chart_type": "box_plot",
        "echarts_config": None,
        "outlier_count": result.get("outlier_count"),
        "lower_bound": result.get("lower_bound"),
        "upper_bound": result.get("upper_bound"),
        "records": result.get("records", []),
        "summary": result.get("summary"),
        "note": "Box plot config can be added after operation_executor returns q1, median, q3, min, and max in ECharts boxplot format.",
    }


def build_scatter_placeholder(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Placeholder for scatter plot.

    Current correlation executor returns a correlation matrix.
    To build scatter plot, operation_executor needs two selected numeric columns
    and paired data points.
    """

    return {
        "chart_required": True,
        "chart_type": "scatter_plot",
        "echarts_config": None,
        "records": result.get("records", []),
        "summary": result.get("summary"),
        "note": "Scatter plot needs paired x/y numeric data. Add secondary measure mapping for better scatter support.",
    }


def build_heatmap_placeholder(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Placeholder for heatmap / correlation heatmap.
    """

    return {
        "chart_required": True,
        "chart_type": "heatmap",
        "echarts_config": None,
        "records": result.get("records", []),
        "summary": result.get("summary"),
        "note": "Heatmap config can be added after correlation matrix is converted to ECharts heatmap series format.",
    }


def build_multi_chart_dashboard(
    result: Dict[str, Any],
    mapped_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build multiple chart configs for full dataset analysis.

    This supports your main project direction:
    - KPI card
    - category breakdown chart
    - trend chart if time column exists
    - distribution/statistics block
    - data quality table
    """

    sections = result.get("sections", {})

    charts = []

    main_metric = sections.get("main_metric")
    if main_metric:
        charts.append(
            {
                "slot": "main_metric",
                "chart": build_kpi_card(main_metric, mapped_plan),
            }
        )

    category_breakdown = sections.get("category_breakdown")
    if category_breakdown:
        charts.append(
            {
                "slot": "category_breakdown",
                "chart": build_bar_chart(category_breakdown, mapped_plan),
            }
        )

    trend = sections.get("trend")
    if trend:
        charts.append(
            {
                "slot": "trend",
                "chart": build_line_chart(trend, mapped_plan),
            }
        )

    distribution = sections.get("distribution")
    if distribution:
        charts.append(
            {
                "slot": "distribution",
                "chart": build_histogram_placeholder(distribution, mapped_plan),
            }
        )

    data_quality = sections.get("data_quality")
    if data_quality:
        charts.append(
            {
                "slot": "data_quality",
                "chart": build_table_chart(data_quality, mapped_plan),
            }
        )

    return {
        "chart_required": True,
        "chart_type": "multi_chart_dashboard",
        "charts": charts,
        "echarts_config": None,
        "summary": result.get("summary"),
    }


if __name__ == "__main__":
    sample_result = {
        "result_type": "table",
        "dimension_column": "Product",
        "measure_column": "Sales",
        "records": [
            {"Product": "Laptop", "Sales": 5500},
            {"Product": "Phone", "Sales": 3350},
            {"Product": "Tablet", "Sales": 1150},
        ],
        "summary": "The top Product by Sales is Laptop with 5500.",
    }

    sample_plan = {
        "best_chart": "horizontal_bar_chart",
        "chart_required": True,
    }

    print(build_chart_config(sample_result, sample_plan))