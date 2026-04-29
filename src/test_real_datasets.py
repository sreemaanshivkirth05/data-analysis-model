"""
test_real_datasets.py
=====================
Comprehensive real-dataset test for the Universal Analysis Planner.

Tests all 4 real datasets with a wide range of question types:
  - Aggregation (sum, mean, max, min)
  - Ranking (highest, lowest, top N)
  - Comparison across categories
  - Time-series / trends
  - Attrition / cancellation rate (diagnostic)
  - Correlation
  - Distribution / outlier
  - Data quality (null check)
  - Complex / ambiguous questions
  - Edge cases

Run from project root:
    python src/test_real_datasets.py
"""

import os
import sys
import json

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pandas as pd
from analysis_plan import create_analysis_plan

# ─────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _load(path, nrows=None):
    if path.endswith(".xlsx"):
        return pd.read_excel(path, nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def _check(label, actual, expected, results):
    """Assert actual == expected (or expected in actual for lists)."""
    if isinstance(expected, list):
        ok = all(e in actual for e in expected)
    else:
        ok = actual == expected
    results.append((label, ok, actual, expected))
    sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    exp_str = str(expected) if not isinstance(expected, list) else f"contains {expected}"
    print(f"  {sym} {label}: got={actual!r}  expected={exp_str}")
    return ok


def _check_not_none(label, actual, results):
    ok = actual is not None
    results.append((label, ok, actual, "not None"))
    sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {sym} {label}: {actual!r}")
    return ok


def _check_none(label, actual, results):
    ok = actual is None
    results.append((label, ok, actual, None))
    sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {sym} {label} should be None: {actual!r}")
    return ok


def _check_no_hallucinations(plan, df, results):
    existing = set(df.columns.tolist())
    used = []
    for key in ["target_column","metric_column","category_column","date_column","text_column"]:
        v = plan.get(key)
        if v:
            used.append(v)
    used += plan.get("driver_columns", [])
    used += plan.get("groupby_columns", [])
    bad = [c for c in used if c not in existing]
    ok = len(bad) == 0
    results.append(("no hallucinations", ok, bad, []))
    sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {sym} no hallucinated columns: {bad if bad else 'clean'}")
    return ok


def _check_is_executable(plan, results):
    ok = plan.get("is_executable", False)
    results.append(("is_executable", ok, ok, True))
    sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {sym} is_executable: {ok}")
    return ok


def _check_no_errors(plan, results):
    errs = plan.get("errors", [])
    ok = len(errs) == 0
    results.append(("no errors", ok, errs, []))
    sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {sym} no errors: {errs if errs else 'clean'}")
    return ok


def run_test(name, question, df, checks_fn):
    """Run a single test: plan + user-defined checks."""
    print(f"\n{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  [{name}]{RESET}")
    print(f"  Q: \"{question}\"")

    plan = create_analysis_plan(question, df)

    print(f"  intent={plan.get('intent')}  op={plan.get('operation')}  chart={plan.get('best_chart')}")
    print(f"  target={plan.get('target_column')}  metric={plan.get('metric_column')}  "
          f"category={plan.get('category_column')}  date={plan.get('date_column')}  "
          f"text={plan.get('text_column')}")
    if plan.get("driver_columns"):
        print(f"  drivers={plan.get('driver_columns')}")
    if plan.get("warnings"):
        print(f"  {YELLOW}warnings={plan.get('warnings')}{RESET}")
    if plan.get("errors"):
        print(f"  {RED}errors={plan.get('errors')}{RESET}")
    print()

    results = []
    _check_no_hallucinations(plan, df, results)
    _check_no_errors(plan, results)
    checks_fn(plan, results)

    passed = sum(1 for _, ok, _, _ in results if ok)
    total  = len(results)
    status = f"{GREEN}PASS{RESET}" if passed == total else f"{RED}FAIL{RESET}"
    print(f"\n  [{status}] {passed}/{total} checks passed")
    return passed, total


# ═══════════════════════════════════════════════════════════
# DATASET 1 — HR Employee Attrition
# ═══════════════════════════════════════════════════════════

def test_hr(df):
    suite_pass = suite_total = 0

    header = f"\n{BOLD}{'═'*60}\n  DATASET 1: HR Employee Attrition  ({len(df)}r × {len(df.columns)}c)\n{'═'*60}{RESET}"
    print(header)

    # 1a
    p, t = run_test(
        "Attrition by overtime",
        "Does overtime affect attrition?",
        df,
        lambda plan, r: [
            _check("target_column", plan.get("target_column"), "Attrition", r),
            _check("category_column", plan.get("category_column"), "OverTime", r),
            _check_not_none("groupby_columns", plan.get("groupby_columns"), r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1b
    p, t = run_test(
        "Attrition rate by department",
        "What is the attrition rate by department?",
        df,
        lambda plan, r: [
            _check("target_column", plan.get("target_column"), "Attrition", r),
            _check("category_column", plan.get("category_column"), "Department", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1c
    p, t = run_test(
        "Highest monthly income by job role",
        "Which job role has the highest monthly income?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "MonthlyIncome", r),
            _check("category_column", plan.get("category_column"), "JobRole", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1d
    p, t = run_test(
        "Average income by education",
        "What is the average monthly income by education level?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "MonthlyIncome", r),
            _check("category_column", plan.get("category_column"), "Education", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1e
    p, t = run_test(
        "Gender attrition comparison",
        "Does gender affect attrition rate?",
        df,
        lambda plan, r: [
            _check("target_column", plan.get("target_column"), "Attrition", r),
            _check("category_column", plan.get("category_column"), "Gender", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1f
    p, t = run_test(
        "Why are employees leaving (diagnostic)",
        "Why are employees leaving the company?",
        df,
        lambda plan, r: [
            _check("target_column", plan.get("target_column"), "Attrition", r),
            _check_not_none("driver_columns not empty", plan.get("driver_columns") or None, r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1g
    p, t = run_test(
        "Distribution of monthly income",
        "Show me the distribution of monthly income",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "MonthlyIncome", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1h
    p, t = run_test(
        "Total employees by department",
        "How many employees are in each department?",
        df,
        lambda plan, r: [
            _check("category_column", plan.get("category_column"), "Department", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1i
    p, t = run_test(
        "Satisfaction score across job roles",
        "What is the average job satisfaction by job role?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "JobSatisfaction", r),
            _check("category_column", plan.get("category_column"), "JobRole", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1j
    p, t = run_test(
        "Business travel attrition",
        "Which business travel frequency has the highest attrition?",
        df,
        lambda plan, r: [
            _check("target_column", plan.get("target_column"), "Attrition", r),
            _check("category_column", plan.get("category_column"), "BusinessTravel", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1k
    p, t = run_test(
        "Work-life balance analysis",
        "How does work life balance vary by department?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "WorkLifeBalance", r),
            _check("category_column", plan.get("category_column"), "Department", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 1l
    p, t = run_test(
        "Salary hike by performance",
        "What is the average salary hike by performance rating?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "PercentSalaryHike", r),
            _check("category_column", plan.get("category_column"), "PerformanceRating", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    return suite_pass, suite_total


# ═══════════════════════════════════════════════════════════
# DATASET 2 — Hotel Bookings
# ═══════════════════════════════════════════════════════════

def test_hotel(df):
    suite_pass = suite_total = 0

    header = f"\n{BOLD}{'═'*60}\n  DATASET 2: Hotel Bookings  ({len(df)}r × {len(df.columns)}c)\n{'═'*60}{RESET}"
    print(header)

    # 2a
    p, t = run_test(
        "Cancellation rate by hotel type",
        "What is the cancellation rate by hotel type?",
        df,
        lambda plan, r: [
            _check("target_column", plan.get("target_column"), "is_canceled", r),
            _check("category_column", plan.get("category_column"), "hotel", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 2b
    p, t = run_test(
        "Most bookings by country",
        "Which country has the most hotel bookings?",
        df,
        lambda plan, r: [
            _check("category_column", plan.get("category_column"), "country", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 2c
    p, t = run_test(
        "Average daily rate by market segment",
        "What is the average daily rate by market segment?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "adr", r),
            _check("category_column", plan.get("category_column"), "market_segment", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 2d
    p, t = run_test(
        "Cancellation rate by deposit type",
        "Which deposit type has the highest cancellation rate?",
        df,
        lambda plan, r: [
            _check("target_column", plan.get("target_column"), "is_canceled", r),
            _check("category_column", plan.get("category_column"), "deposit_type", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 2e
    p, t = run_test(
        "Average lead time",
        "What is the average lead time for bookings?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "lead_time", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 2f
    p, t = run_test(
        "Special requests by customer type",
        "Which customer type has the most special requests?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "total_of_special_requests", r),
            _check("category_column", plan.get("category_column"), "customer_type", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 2g
    p, t = run_test(
        "Why are bookings being canceled (diagnostic)",
        "Why are hotel bookings being canceled?",
        df,
        lambda plan, r: [
            _check("target_column", plan.get("target_column"), "is_canceled", r),
            _check_not_none("driver_columns not empty", plan.get("driver_columns") or None, r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 2h
    p, t = run_test(
        "ADR by distribution channel",
        "What is the average daily rate by distribution channel?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "adr", r),
            _check("category_column", plan.get("category_column"), "distribution_channel", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 2i
    p, t = run_test(
        "Weekend vs weeknight stays",
        "What is the average number of weekend nights stayed?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "stays_in_weekend_nights", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 2j
    p, t = run_test(
        "Repeated guests cancellation",
        "Do repeated guests cancel less than new guests?",
        df,
        lambda plan, r: [
            _check("target_column", plan.get("target_column"), "is_canceled", r),
            _check("category_column", plan.get("category_column"), "is_repeated_guest", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    return suite_pass, suite_total


# ═══════════════════════════════════════════════════════════
# DATASET 3 — Online Retail (Excel)
# ═══════════════════════════════════════════════════════════

def test_online_retail(df):
    suite_pass = suite_total = 0

    header = f"\n{BOLD}{'═'*60}\n  DATASET 3: Online Retail  ({len(df)}r × {len(df.columns)}c)\n{'═'*60}{RESET}"
    print(header)

    # 3a
    p, t = run_test(
        "Total quantity by country",
        "Which country has the highest total quantity sold?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Quantity", r),
            _check("category_column", plan.get("category_column"), "Country", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 3b
    p, t = run_test(
        "Revenue trend over time",
        "How did revenue change over time?",
        df,
        lambda plan, r: [
            _check_not_none("date_column selected", plan.get("date_column"), r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 3c
    p, t = run_test(
        "Top products by quantity",
        "What are the most popular products by quantity?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Quantity", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 3d
    p, t = run_test(
        "Average unit price by country",
        "What is the average unit price by country?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "UnitPrice", r),
            _check("category_column", plan.get("category_column"), "Country", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 3e
    p, t = run_test(
        "Quantity trends over time",
        "How did total quantity sold change month over month?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Quantity", r),
            _check_not_none("date_column selected", plan.get("date_column"), r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 3f
    p, t = run_test(
        "Compare sales across countries",
        "Compare sales across countries",
        df,
        lambda plan, r: [
            _check("category_column", plan.get("category_column"), "Country", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 3g
    p, t = run_test(
        "Total revenue by customer",
        "Which customer has spent the most?",
        df,
        lambda plan, r: [
            _check_not_none("category selected (CustomerID)", plan.get("category_column"), r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    return suite_pass, suite_total


# ═══════════════════════════════════════════════════════════
# DATASET 4 — Sales (simple CSV)
# ═══════════════════════════════════════════════════════════

def test_sales(df):
    suite_pass = suite_total = 0

    header = f"\n{BOLD}{'═'*60}\n  DATASET 4: Sales  ({len(df)}r × {len(df.columns)}c)\n{'═'*60}{RESET}"
    print(header)

    # 4a
    p, t = run_test(
        "Highest sales by product",
        "Which product has the highest sales?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Sales", r),
            _check("category_column", plan.get("category_column"), "Product", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 4b
    p, t = run_test(
        "Sales comparison by country (plural)",
        "Compare sales across countries",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Sales", r),
            _check("category_column", plan.get("category_column"), "Country", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 4c
    p, t = run_test(
        "Total profit by product",
        "What is the total profit by product?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Profit", r),
            _check("category_column", plan.get("category_column"), "Product", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 4d
    p, t = run_test(
        "Sales trend over time",
        "How did sales change over time?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Sales", r),
            _check("date_column", plan.get("date_column"), "Date", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 4e
    p, t = run_test(
        "Most profitable country",
        "Which country is most profitable?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Profit", r),
            _check("category_column", plan.get("category_column"), "Country", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 4f
    p, t = run_test(
        "Customer sales ranking",
        "Which customer has the highest sales?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Sales", r),
            _check("category_column", plan.get("category_column"), "Customer", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 4g
    p, t = run_test(
        "Total sales aggregation",
        "What is the total sales?",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Sales", r),
            _check_none("no category needed for scalar", plan.get("category_column"), r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    # 4h
    p, t = run_test(
        "Profit distribution",
        "Show me the distribution of profit",
        df,
        lambda plan, r: [
            _check("metric_column", plan.get("metric_column"), "Profit", r),
            _check_is_executable(plan, r),
        ]
    )
    suite_pass += p; suite_total += t

    return suite_pass, suite_total


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    root = _ROOT_DIR

    print(f"\n{BOLD}{'═'*60}")
    print("  UNIVERSAL ANALYSIS PLANNER — REAL DATASET TEST SUITE")
    print(f"{'═'*60}{RESET}")

    grand_pass = grand_total = 0
    dataset_results = []

    # ── Dataset 1: HR Attrition ──────────────────────────────
    try:
        df_hr = _load(f"{root}/real_datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        p, t = test_hr(df_hr)
        grand_pass += p; grand_total += t
        dataset_results.append(("HR Attrition", p, t))
    except Exception as e:
        print(f"\n{RED}HR dataset failed to load: {e}{RESET}")

    # ── Dataset 2: Hotel Bookings ────────────────────────────
    try:
        df_hotel = _load(f"{root}/real_datasets/hotel_bookings.csv")
        p, t = test_hotel(df_hotel)
        grand_pass += p; grand_total += t
        dataset_results.append(("Hotel Bookings", p, t))
    except Exception as e:
        print(f"\n{RED}Hotel dataset failed to load: {e}{RESET}")

    # ── Dataset 3: Online Retail (Excel) ─────────────────────
    try:
        df_retail = _load(f"{root}/real_datasets/Online Retail.xlsx", nrows=50000)
        p, t = test_online_retail(df_retail)
        grand_pass += p; grand_total += t
        dataset_results.append(("Online Retail", p, t))
    except Exception as e:
        print(f"\n{RED}Online Retail dataset failed to load: {e}{RESET}")

    # ── Dataset 4: Sales ─────────────────────────────────────
    try:
        df_sales = _load(f"{root}/sample_datasets/sales.csv")
        p, t = test_sales(df_sales)
        grand_pass += p; grand_total += t
        dataset_results.append(("Sales", p, t))
    except Exception as e:
        print(f"\n{RED}Sales dataset failed to load: {e}{RESET}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n\n{BOLD}{'═'*60}")
    print("  FINAL SUMMARY")
    print(f"{'═'*60}{RESET}")

    for name, p, t in dataset_results:
        pct = f"{100*p//t}%" if t else "N/A"
        status = f"{GREEN}✓{RESET}" if p == t else f"{YELLOW}△{RESET}"
        print(f"  {status} {name:25s}  {p}/{t} checks  ({pct})")

    print(f"\n  {'─'*40}")
    pct_grand = f"{100*grand_pass//grand_total}%" if grand_total else "N/A"
    if grand_pass == grand_total:
        print(f"  {GREEN}{BOLD}ALL PASS  {grand_pass}/{grand_total} checks ({pct_grand}){RESET}")
    else:
        failed = grand_total - grand_pass
        print(f"  {YELLOW}{BOLD}{failed} checks failed  {grand_pass}/{grand_total} passed ({pct_grand}){RESET}")


if __name__ == "__main__":
    main()
