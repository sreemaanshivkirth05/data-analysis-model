"""
test_column_mapper.py
=====================
Comprehensive tests for the Column Mapper and Analysis Plan wrapper.

Covers:
  1. The 5 canonical example questions from the spec
  2. HR Attrition dataset (Attrition/OverTime/Department)
  3. Sales dataset (Product/Country/Date/Sales/Profit)
  4. Online Retail dataset
  5. Hotel Bookings dataset
  6. Edge cases: empty questions, missing columns, ambiguous datasets

Run:
    cd src
    python test_column_mapper.py
"""

import os
import sys
import json
import traceback
from typing import Dict, Any

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from column_mapper import map_columns
from analysis_plan import create_analysis_plan

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ============================================================
# Test Infrastructure
# ============================================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.failures = []

    def check(self, label: str, actual, expected, exact: bool = True):
        if exact:
            ok = actual == expected
        else:
            # For list checks: all expected items are present
            ok = all(item in (actual or []) for item in expected)

        if ok:
            self.passed += 1
            print(f"  {GREEN}✓{RESET} {label}: {actual!r}")
        else:
            self.failed += 1
            self.failures.append(f"{label}: expected={expected!r}, got={actual!r}")
            print(f"  {RED}✗{RESET} {label}: expected={expected!r}, got={actual!r}")

    def check_not_none(self, label: str, actual):
        if actual is not None:
            self.passed += 1
            print(f"  {GREEN}✓{RESET} {label}: {actual!r}")
        else:
            self.failed += 1
            self.failures.append(f"{label}: expected not None, got None")
            print(f"  {RED}✗{RESET} {label}: expected not None, got None")

    def check_none(self, label: str, actual):
        if actual is None:
            self.passed += 1
            print(f"  {GREEN}✓{RESET} {label} is None")
        else:
            self.failed += 1
            self.failures.append(f"{label}: expected None, got {actual!r}")
            print(f"  {RED}✗{RESET} {label}: expected None, got {actual!r}")

    def check_in(self, label: str, actual, valid_values: list):
        if actual in valid_values:
            self.passed += 1
            print(f"  {GREEN}✓{RESET} {label}: {actual!r} ∈ {valid_values}")
        else:
            self.failed += 1
            self.failures.append(f"{label}: {actual!r} not in {valid_values}")
            print(f"  {RED}✗{RESET} {label}: {actual!r} not in {valid_values}")

    def check_gt(self, label: str, actual: float, threshold: float):
        if actual > threshold:
            self.passed += 1
            print(f"  {GREEN}✓{RESET} {label}: {actual:.3f} > {threshold}")
        else:
            self.failed += 1
            self.failures.append(f"{label}: {actual:.3f} not > {threshold}")
            print(f"  {RED}✗{RESET} {label}: {actual:.3f} not > {threshold}")

    def check_contains(self, label: str, actual: list, item):
        if item in (actual or []):
            self.passed += 1
            print(f"  {GREEN}✓{RESET} {label}: '{item}' in {actual}")
        else:
            self.failed += 1
            self.failures.append(f"{label}: '{item}' not in {actual}")
            print(f"  {RED}✗{RESET} {label}: '{item}' not in {actual}")

    def check_no_hallucinations(self, label: str, mapping: Dict, df: pd.DataFrame):
        existing = set(df.columns.tolist())
        bad = []
        for key in ["target_column", "metric_column", "category_column", "date_column", "text_column"]:
            col = mapping.get(key)
            if col is not None and col not in existing:
                bad.append(f"{key}={col!r}")
        for col in mapping.get("driver_columns", []):
            if col not in existing:
                bad.append(f"driver={col!r}")
        for col in mapping.get("groupby_columns", []):
            if col not in existing:
                bad.append(f"groupby={col!r}")

        if not bad:
            self.passed += 1
            print(f"  {GREEN}✓{RESET} {label}: no hallucinated columns")
        else:
            self.failed += 1
            self.failures.append(f"{label}: hallucinated columns: {bad}")
            print(f"  {RED}✗{RESET} {label}: hallucinated columns: {bad}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        status = GREEN + "PASS" + RESET if self.failed == 0 else RED + "FAIL" + RESET
        print(f"\n  [{status}] {self.passed}/{total} checks passed")
        if self.failures:
            for f in self.failures:
                print(f"    → {f}")
        return self.failed == 0


def run_test(name: str, fn) -> bool:
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}TEST: {name}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")
    r = TestResult(name)
    try:
        fn(r)
    except Exception as exc:
        print(f"  {RED}ERROR: {exc}{RESET}")
        traceback.print_exc()
        r.failed += 1
        r.failures.append(f"Exception: {exc}")
    return r.summary()


def _make_plan(intent, operation, needs_numeric=True, needs_category=False,
               needs_datetime=False, needs_text=False, best_chart="bar_chart",
               chart_required=True):
    """Build a minimal mock plan dict for testing map_columns() in isolation."""
    return {
        "intent": intent,
        "answer_depth": "visual_answer",
        "operation": operation,
        "best_chart": best_chart,
        "chart_required": chart_required,
        "required_data_roles": {
            "needs_numeric": needs_numeric,
            "needs_category": needs_category,
            "needs_datetime": needs_datetime,
            "needs_text": needs_text,
        },
        "confidence_scores": {},
        "min_confidence": 0.8,
        "low_confidence": [],
        "recommended_charts": [best_chart],
        "fallback_chart": "table",
        "planner_source": "test_mock",
        "rule_override_applied": False,
        "requires_llm_fallback": False,
        "rules_fired": [],
        "raw_ml_prediction": {},
        "rule_override_applied": False,
    }


# ============================================================
# Shared DataFrames
# ============================================================

def make_hr_df() -> pd.DataFrame:
    """HR / Employee Attrition dataset."""
    return pd.DataFrame({
        "EmployeeID":       range(1, 11),
        "Age":              [25, 32, 45, 28, 38, 51, 29, 41, 35, 47],
        "Department":       ["Sales","HR","IT","Sales","IT","HR","Sales","IT","HR","Sales"],
        "JobRole":          ["Manager","Analyst","Developer","Rep","Lead","Director","Rep","Architect","Analyst","Manager"],
        "MonthlyIncome":    [5000, 6200, 8100, 4500, 9000, 11000, 4800, 8700, 6000, 10500],
        "OverTime":         ["Yes","No","Yes","No","Yes","No","Yes","No","No","Yes"],
        "Attrition":        ["Yes","No","No","Yes","No","No","Yes","No","Yes","No"],
        "BusinessTravel":   ["Travel_Rarely","Non-Travel","Travel_Frequently","Travel_Rarely","Non-Travel",
                             "Travel_Frequently","Travel_Rarely","Non-Travel","Travel_Frequently","Travel_Rarely"],
        "YearsAtCompany":   [1, 5, 10, 2, 8, 15, 1, 9, 4, 12],
    })


def make_sales_df() -> pd.DataFrame:
    """Simple sales dataset."""
    return pd.DataFrame({
        "OrderID":    ["O001","O002","O003","O004","O005","O006","O007","O008","O009","O010"],
        "Date":       pd.to_datetime(["2024-01-01","2024-01-15","2024-02-01","2024-02-20",
                                      "2024-03-05","2024-03-20","2024-04-01","2024-04-15",
                                      "2024-05-01","2024-05-20"]),
        "Product":    ["Widget A","Widget B","Gadget X","Widget A","Gadget X","Widget B",
                       "Widget A","Gadget X","Widget B","Widget A"],
        "Country":    ["USA","UK","Germany","USA","France","UK","Germany","USA","France","UK"],
        "Sales":      [1200, 850, 2100, 950, 1800, 760, 1350, 1950, 890, 1100],
        "Profit":     [300, 180, 520, 210, 450, 140, 320, 480, 200, 270],
        "Quantity":   [10, 7, 15, 8, 12, 6, 11, 14, 7, 9],
    })


def make_subscription_df() -> pd.DataFrame:
    """SaaS / subscription churn dataset."""
    return pd.DataFrame({
        "CustomerID":  range(1, 11),
        "Segment":     ["Enterprise","SMB","Enterprise","Consumer","SMB",
                        "Consumer","Enterprise","SMB","Consumer","Enterprise"],
        "Region":      ["North","South","East","West","North","South","East","West","North","South"],
        "Plan":        ["Pro","Basic","Enterprise","Basic","Pro",
                        "Basic","Enterprise","Pro","Basic","Enterprise"],
        "MRR":         [2500, 500, 8000, 200, 1500, 300, 9000, 1800, 250, 7500],
        "Tenure":      [24, 6, 36, 3, 18, 9, 48, 15, 4, 30],
        "Churn":       ["Yes","No","No","Yes","No","Yes","No","No","Yes","No"],
        "SignupDate":  pd.to_datetime(["2022-01-01","2023-06-15","2021-03-01","2024-01-01",
                                       "2022-09-01","2023-03-15","2020-05-01","2022-07-01",
                                       "2023-11-01","2021-09-01"]),
    })


def make_ecommerce_df() -> pd.DataFrame:
    """E-commerce dataset with revenue over time."""
    return pd.DataFrame({
        "TransactionID": range(1, 11),
        "InvoiceDate":   pd.to_datetime(["2023-01-05","2023-02-12","2023-03-20","2023-04-08",
                                          "2023-05-15","2023-06-22","2023-07-03","2023-08-18",
                                          "2023-09-25","2023-10-30"]),
        "Country":       ["UK","France","Germany","UK","Italy","Spain","UK","France","Germany","Italy"],
        "Description":   ["Widget","Gadget","Widget","Gizmo","Widget","Gadget","Gizmo","Widget","Gadget","Widget"],
        "Quantity":      [3, 5, 2, 8, 4, 6, 1, 7, 3, 5],
        "UnitPrice":     [2.5, 5.0, 3.0, 1.5, 2.5, 4.0, 6.0, 2.0, 3.5, 2.5],
        "Revenue":       [7.5, 25.0, 6.0, 12.0, 10.0, 24.0, 6.0, 14.0, 10.5, 12.5],
        "CustomerID":    ["C001","C002","C003","C001","C004","C005","C002","C003","C004","C001"],
    })


def make_textfeedback_df() -> pd.DataFrame:
    """Customer feedback dataset with free-text reviews."""
    return pd.DataFrame({
        "ReviewID":     range(1, 11),
        "ProductName":  ["Widget A"]*5 + ["Gadget X"]*5,
        "Rating":       [4, 2, 5, 3, 4, 5, 1, 4, 3, 5],
        "Review":       [
            "Great product, very happy with my purchase",
            "Disappointing quality, broke after a week",
            "Excellent! Exceeded expectations completely",
            "Average product, nothing special to report",
            "Good value for money, would recommend",
            "Amazing gadget, works perfectly every time",
            "Terrible experience, stopped working immediately",
            "Pretty good, minor issues but overall fine",
            "Decent product, delivery was slow though",
            "Fantastic! Best purchase I have made",
        ],
        "Date":         pd.to_datetime(["2024-01-01","2024-01-05","2024-01-10","2024-01-15",
                                         "2024-01-20","2024-02-01","2024-02-05","2024-02-10",
                                         "2024-02-15","2024-02-20"]),
    })


# ============================================================
# Test 1 — "Does overtime affect attrition?" (HR dataset)
# ============================================================

def test_overtime_attrition(r: TestResult):
    """
    Canonical Example 1 from the spec.
    Expected: target=Attrition, category=OverTime
    """
    df = make_hr_df()
    question = "Does overtime affect attrition?"

    plan = _make_plan(
        intent="diagnostic_analysis",
        operation="groupby_target_rate",
        needs_numeric=False,
        needs_category=True,
        needs_datetime=False,
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check("target_column", mapping["target_column"], "Attrition")
    r.check("category_column", mapping["category_column"], "OverTime")
    r.check_none("metric_column should be null", mapping["metric_column"])
    r.check_contains("groupby_columns includes OverTime", mapping["groupby_columns"], "OverTime")
    r.check_gt("confidence > 0.3", mapping["confidence"], 0.3)
    r.check_no_hallucinations("no hallucinated columns", mapping, df)


# ============================================================
# Test 2 — "Which product has the highest sales?" (Sales dataset)
# ============================================================

def test_product_highest_sales(r: TestResult):
    """
    Canonical Example 2 from the spec.
    Expected: metric=Sales, category=Product
    """
    df = make_sales_df()
    question = "Which product has the highest sales?"

    plan = _make_plan(
        intent="ranking",
        operation="groupby_sum_sort_desc",
        needs_numeric=True,
        needs_category=True,
        best_chart="horizontal_bar_chart",
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check("metric_column", mapping["metric_column"], "Sales")
    r.check("category_column", mapping["category_column"], "Product")
    r.check_none("target_column should be null", mapping["target_column"])
    r.check_contains("groupby_columns includes Product", mapping["groupby_columns"], "Product")
    r.check_gt("confidence > 0.4", mapping["confidence"], 0.4)
    r.check_no_hallucinations("no hallucinated columns", mapping, df)


# ============================================================
# Test 3 — "How did revenue change over time?" (E-commerce dataset)
# ============================================================

def test_revenue_over_time(r: TestResult):
    """
    Canonical Example 3 from the spec.
    Expected: metric=Revenue, date=InvoiceDate
    """
    df = make_ecommerce_df()
    question = "How did revenue change over time?"

    plan = _make_plan(
        intent="trend_analysis",
        operation="time_groupby_sum",
        needs_numeric=True,
        needs_category=False,
        needs_datetime=True,
        best_chart="line_chart",
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check("metric_column", mapping["metric_column"], "Revenue")
    r.check("date_column", mapping["date_column"], "InvoiceDate")
    r.check_none("target_column should be null", mapping["target_column"])
    r.check_gt("confidence > 0.4", mapping["confidence"], 0.4)
    r.check_no_hallucinations("no hallucinated columns", mapping, df)


# ============================================================
# Test 4 — "Which country has the highest profit?" (Sales dataset)
# ============================================================

def test_country_highest_profit(r: TestResult):
    """
    Canonical Example 4 from the spec.
    Expected: metric=Profit, category=Country
    """
    df = make_sales_df()
    question = "Which country has the highest profit?"

    plan = _make_plan(
        intent="ranking",
        operation="groupby_sum_sort_desc",
        needs_numeric=True,
        needs_category=True,
        best_chart="horizontal_bar_chart",
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check("metric_column", mapping["metric_column"], "Profit")
    r.check("category_column", mapping["category_column"], "Country")
    r.check_none("target_column should be null", mapping["target_column"])
    r.check_contains("groupby_columns includes Country", mapping["groupby_columns"], "Country")
    r.check_gt("confidence > 0.4", mapping["confidence"], 0.4)
    r.check_no_hallucinations("no hallucinated columns", mapping, df)


# ============================================================
# Test 5 — "Why is customer churn increasing?" (Subscription dataset)
# ============================================================

def test_churn_increasing(r: TestResult):
    """
    Canonical Example 5 from the spec.
    Expected: target=Churn, date=SignupDate, driver_columns populated
    """
    df = make_subscription_df()
    question = "Why is customer churn increasing?"

    plan = _make_plan(
        intent="diagnostic_analysis",
        operation="diagnostic_analysis",
        needs_numeric=True,
        needs_category=True,
        needs_datetime=True,
        best_chart="multi_chart_dashboard",
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check("target_column", mapping["target_column"], "Churn")
    r.check_not_none("date_column should be selected", mapping["date_column"])
    r.check_no_hallucinations("no hallucinated columns", mapping, df)
    # Driver columns should include some categorical columns
    r.check_gt("at least 1 driver column", len(mapping["driver_columns"]), 0)


# ============================================================
# Test 6 — Attrition rate by department
# ============================================================

def test_attrition_by_department(r: TestResult):
    """
    'Which department has the highest attrition rate?'
    Expected: target=Attrition, category=Department
    """
    df = make_hr_df()
    question = "Which department has the highest attrition rate?"

    plan = _make_plan(
        intent="ranking",
        operation="groupby_target_rate_sort_desc",
        needs_numeric=False,
        needs_category=True,
        best_chart="horizontal_bar_chart",
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check("target_column", mapping["target_column"], "Attrition")
    r.check("category_column", mapping["category_column"], "Department")
    r.check_contains("groupby_columns includes Department", mapping["groupby_columns"], "Department")
    r.check_no_hallucinations("no hallucinated columns", mapping, df)


# ============================================================
# Test 7 — Sentiment analysis on reviews
# ============================================================

def test_sentiment_analysis(r: TestResult):
    """
    'Analyze sentiment in customer reviews'
    Expected: text_column=Review
    """
    df = make_textfeedback_df()
    question = "Analyze sentiment in customer reviews"

    plan = _make_plan(
        intent="text_analysis",
        operation="sentiment_summary",
        needs_numeric=False,
        needs_category=False,
        needs_datetime=False,
        needs_text=True,
        best_chart="bar_chart",
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check("text_column", mapping["text_column"], "Review")
    r.check_none("target_column should be null", mapping["target_column"])
    r.check_no_hallucinations("no hallucinated columns", mapping, df)


# ============================================================
# Test 8 — Total sales aggregation
# ============================================================

def test_total_sales(r: TestResult):
    """
    'What is the total sales?'
    Expected: metric=Sales, no dimension, no target
    """
    df = make_sales_df()
    question = "What is the total sales?"

    plan = _make_plan(
        intent="aggregation",
        operation="sum",
        needs_numeric=True,
        needs_category=False,
        best_chart="kpi_card",
        chart_required=False,
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check("metric_column", mapping["metric_column"], "Sales")
    r.check_none("target_column should be null", mapping["target_column"])
    r.check_none("category_column should be null for scalar op", mapping["category_column"])
    r.check_no_hallucinations("no hallucinated columns", mapping, df)


# ============================================================
# Test 9 — Hallucination safety: non-existent column names
# ============================================================

def test_hallucination_safety(r: TestResult):
    """
    No matter what the question says, map_columns() must never return
    a column that does not exist in the dataframe.
    """
    df = pd.DataFrame({
        "ProductName": ["A","B","C"],
        "Revenue":     [100, 200, 300],
    })

    question = "Show churn by region for the quarterly trend"

    # Mock plan asking for many role types
    plan = _make_plan(
        intent="diagnostic_analysis",
        operation="diagnostic_analysis",
        needs_numeric=True,
        needs_category=True,
        needs_datetime=True,
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check_no_hallucinations("no hallucinated columns", mapping, df)

    # Specific columns that DON'T exist should never appear
    nonexistent = ["Churn", "Region", "Date", "Quarter"]
    for col in nonexistent:
        for key in ["target_column", "metric_column", "category_column", "date_column"]:
            val = mapping.get(key)
            if val == col:
                r.failed += 1
                r.failures.append(f"{key}={col!r} is hallucinated (not in df)")
                print(f"  {RED}✗{RESET} Hallucinated {key}={col!r}")
                return
    r.passed += 1
    print(f"  {GREEN}✓{RESET} No non-existent columns returned")


# ============================================================
# Test 10 — Sales comparison by country
# ============================================================

def test_sales_by_country(r: TestResult):
    """
    'Compare sales across countries'
    Expected: metric=Sales, category=Country
    """
    df = make_sales_df()
    question = "Compare sales across countries"

    plan = _make_plan(
        intent="comparison",
        operation="groupby_sum",
        needs_numeric=True,
        needs_category=True,
        best_chart="bar_chart",
    )

    mapping = map_columns(question, df, plan)

    print(f"  Mapping: {json.dumps({k: v for k, v in mapping.items() if k != 'warnings'}, indent=4)}")

    r.check("metric_column", mapping["metric_column"], "Sales")
    r.check("category_column", mapping["category_column"], "Country")
    r.check_contains("groupby_columns includes Country", mapping["groupby_columns"], "Country")
    r.check_no_hallucinations("no hallucinated columns", mapping, df)


# ============================================================
# Test 11 — create_analysis_plan() integration (HR dataset)
# ============================================================

def test_create_analysis_plan_hr(r: TestResult):
    """
    Full end-to-end test of create_analysis_plan() on the HR dataset.
    Tests that planner + mapper work together correctly.
    """
    df = make_hr_df()
    question = "Does overtime affect attrition?"

    try:
        plan = create_analysis_plan(question, df)
    except Exception as exc:
        r.failed += 1
        r.failures.append(f"create_analysis_plan raised exception: {exc}")
        print(f"  {RED}✗{RESET} Exception: {exc}")
        traceback.print_exc()
        return

    print(f"  Operation:        {plan.get('operation')}")
    print(f"  Intent:           {plan.get('intent')}")
    print(f"  Best chart:       {plan.get('best_chart')}")
    print(f"  target_column:    {plan.get('target_column')}")
    print(f"  category_column:  {plan.get('category_column')}")
    print(f"  metric_column:    {plan.get('metric_column')}")
    print(f"  driver_columns:   {plan.get('driver_columns')}")
    print(f"  groupby_columns:  {plan.get('groupby_columns')}")
    print(f"  mapping_confidence: {plan.get('mapping_confidence')}")
    print(f"  is_executable:    {plan.get('is_executable')}")
    print(f"  planner_source:   {plan.get('planner_source')}")
    print(f"  errors:           {plan.get('errors')}")

    # Required keys should all be present
    for key in [
        "question", "intent", "operation", "best_chart", "chart_required",
        "target_column", "metric_column", "category_column", "date_column",
        "driver_columns", "groupby_columns", "time_grain",
        "confidence_scores", "mapping_confidence", "min_confidence",
        "is_executable", "warnings", "errors", "planner_version",
    ]:
        if key not in plan:
            r.failed += 1
            r.failures.append(f"Missing key in plan: {key!r}")
            print(f"  {RED}✗{RESET} Missing key: {key!r}")
        else:
            r.passed += 1

    r.check_no_hallucinations("no hallucinated columns", plan, df)
    r.check("no errors", plan.get("errors"), [])
    r.check("target_column=Attrition", plan.get("target_column"), "Attrition")
    r.check("category_column=OverTime", plan.get("category_column"), "OverTime")


# ============================================================
# Test 12 — create_analysis_plan() integration (Sales dataset)
# ============================================================

def test_create_analysis_plan_sales(r: TestResult):
    """
    End-to-end create_analysis_plan() on sales dataset.
    """
    df = make_sales_df()
    question = "Which product has the highest sales?"

    try:
        plan = create_analysis_plan(question, df)
    except Exception as exc:
        r.failed += 1
        r.failures.append(f"create_analysis_plan raised exception: {exc}")
        print(f"  {RED}✗{RESET} Exception: {exc}")
        return

    print(f"  Operation:        {plan.get('operation')}")
    print(f"  Intent:           {plan.get('intent')}")
    print(f"  metric_column:    {plan.get('metric_column')}")
    print(f"  category_column:  {plan.get('category_column')}")
    print(f"  is_executable:    {plan.get('is_executable')}")

    r.check("metric_column=Sales", plan.get("metric_column"), "Sales")
    r.check("category_column=Product", plan.get("category_column"), "Product")
    r.check_none("target_column=null", plan.get("target_column"))
    r.check_no_hallucinations("no hallucinated columns", plan, df)
    r.check("no errors", plan.get("errors"), [])


# ============================================================
# Test 13 — Real dataset: HR Attrition file
# ============================================================

def test_real_hr_attrition_file(r: TestResult):
    """
    Test against the real WA_Fn-UseC_-HR-Employee-Attrition.csv if it exists.
    """
    filepath = os.path.join(_ROOT_DIR, "real_datasets", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

    if not os.path.exists(filepath):
        print(f"  {YELLOW}SKIP{RESET} Real HR dataset not found at {filepath}")
        r.passed += 1  # Don't penalise for missing file
        return

    df = pd.read_csv(filepath)
    print(f"  Dataset: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)[:10]}...")

    question = "Does overtime affect attrition?"
    plan = create_analysis_plan(question, df)

    print(f"  target_column:    {plan.get('target_column')}")
    print(f"  category_column:  {plan.get('category_column')}")
    print(f"  operation:        {plan.get('operation')}")

    r.check("target_column=Attrition", plan.get("target_column"), "Attrition")
    r.check("category_column=OverTime", plan.get("category_column"), "OverTime")
    r.check_no_hallucinations("no hallucinated columns", plan, df)


# ============================================================
# Test 14 — Real dataset: Hotel Bookings
# ============================================================

def test_real_hotel_bookings(r: TestResult):
    """
    Test 'Which country has the most bookings?' against real hotel dataset.
    """
    filepath = os.path.join(_ROOT_DIR, "real_datasets", "hotel_bookings.csv")

    if not os.path.exists(filepath):
        print(f"  {YELLOW}SKIP{RESET} Hotel bookings dataset not found.")
        r.passed += 1
        return

    df = pd.read_csv(filepath)
    print(f"  Dataset: {len(df)} rows × {len(df.columns)} columns")

    question = "Which country has the most bookings?"
    plan = create_analysis_plan(question, df)

    print(f"  category_column:  {plan.get('category_column')}")
    print(f"  metric_column:    {plan.get('metric_column')}")
    print(f"  operation:        {plan.get('operation')}")

    r.check_not_none("category_column selected", plan.get("category_column"))
    r.check_no_hallucinations("no hallucinated columns", plan, df)


# ============================================================
# Test 15 — Edge case: empty DataFrame
# ============================================================

def test_empty_dataframe(r: TestResult):
    """
    create_analysis_plan() should return errors gracefully for empty df.
    """
    df = pd.DataFrame()
    question = "What is the total sales?"

    plan = create_analysis_plan(question, df)

    print(f"  errors: {plan.get('errors')}")
    print(f"  is_executable: {plan.get('is_executable')}")

    r.check("is_executable=False", plan.get("is_executable"), False)
    if plan.get("errors"):
        r.passed += 1
        print(f"  {GREEN}✓{RESET} errors list is non-empty (as expected)")
    else:
        r.failed += 1
        r.failures.append("Expected errors for empty DataFrame, got none")
        print(f"  {RED}✗{RESET} Expected errors for empty DataFrame")


# ============================================================
# Test 16 — map_columns() with pre-built metadata (no df profiling)
# ============================================================

def test_map_columns_with_metadata(r: TestResult):
    """
    Verify map_columns() works when metadata is passed directly
    (avoiding double-profiling in the main pipeline).
    """
    df = make_sales_df()
    question = "Which country has the highest profit?"

    # Pre-build metadata
    try:
        from metadata_profiler import profile_dataframe
        metadata = profile_dataframe(df)
    except ImportError:
        print(f"  {YELLOW}SKIP{RESET} metadata_profiler not importable from test context")
        r.passed += 1
        return

    plan = _make_plan(
        intent="ranking",
        operation="groupby_sum_sort_desc",
        needs_numeric=True,
        needs_category=True,
        best_chart="horizontal_bar_chart",
    )

    mapping = map_columns(question, df, plan, metadata=metadata)

    print(f"  metric_column:    {mapping['metric_column']}")
    print(f"  category_column:  {mapping['category_column']}")

    r.check("metric_column=Profit", mapping["metric_column"], "Profit")
    r.check("category_column=Country", mapping["category_column"], "Country")
    r.check_no_hallucinations("no hallucinated columns", mapping, df)


# ============================================================
# Main Runner
# ============================================================

TESTS = [
    ("Example 1: Does overtime affect attrition?",       test_overtime_attrition),
    ("Example 2: Which product has the highest sales?",   test_product_highest_sales),
    ("Example 3: How did revenue change over time?",      test_revenue_over_time),
    ("Example 4: Which country has the highest profit?",  test_country_highest_profit),
    ("Example 5: Why is customer churn increasing?",      test_churn_increasing),
    ("Attrition rate by department",                      test_attrition_by_department),
    ("Sentiment analysis on reviews",                     test_sentiment_analysis),
    ("Total sales aggregation",                           test_total_sales),
    ("Hallucination safety check",                        test_hallucination_safety),
    ("Sales comparison by country",                       test_sales_by_country),
    ("create_analysis_plan() — HR dataset",               test_create_analysis_plan_hr),
    ("create_analysis_plan() — Sales dataset",            test_create_analysis_plan_sales),
    ("Real dataset — HR Attrition CSV",                   test_real_hr_attrition_file),
    ("Real dataset — Hotel Bookings CSV",                 test_real_hotel_bookings),
    ("Edge case — Empty DataFrame",                       test_empty_dataframe),
    ("map_columns() with pre-built metadata",             test_map_columns_with_metadata),
]


def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  COLUMN MAPPER TEST SUITE{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    all_passed = 0
    all_failed = 0
    failed_tests = []

    for name, fn in TESTS:
        ok = run_test(name, fn)
        if ok:
            all_passed += 1
        else:
            all_failed += 1
            failed_tests.append(name)

    total = all_passed + all_failed

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  FINAL RESULTS{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    status = GREEN + "ALL PASS" + RESET if all_failed == 0 else RED + "SOME FAILED" + RESET
    print(f"  [{status}]  {all_passed}/{total} tests passed\n")

    if failed_tests:
        print(f"  {RED}Failed tests:{RESET}")
        for name in failed_tests:
            print(f"    • {name}")

    return all_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
