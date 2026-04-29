import re
import sys
import os
from typing import Dict, Any, Optional, List, Tuple


# ============================================================
# Keyword Groups
# ============================================================

MEASURE_KEYWORDS = [
    "sales",
    "revenue",
    "amount",
    "profit",
    "price",
    "cost",
    "quantity",
    "qty",
    "total",
    "score",
    "rating",
    "value",
    "income",
    "monthly income",
    "rate",
    "monthly rate",
    "daily rate",
    "hourly rate",
    "expense",
    "margin",
    "discount",
    "orders",
    "clicks",
    "impressions",
    "conversion",
    "conversion rate",
    "salary",
    "bonus",
    "temperature",
    "humidity",
    "pressure",
    "latency",
    "response time",
    "error count",
    "request count",
    "patient count",
    "age",
    "treatment cost",
    "attendance",
    "resolution time",
    "length of stay",
    "spend",
    "budget",
    "attrition rate",
    "performance score",
    "percent salary hike",
    "years at company",
    "years in current role",
    "years since last promotion",
    "years with curr manager",
    "total working years",
    "training times last year",
    "distance from home",
    "environment satisfaction",
    "job involvement",
    "job level",
    "job satisfaction",
    "performance rating",
    "relationship satisfaction",
    "stock option level",
    "work life balance",
    "num companies worked",
]

DIMENSION_KEYWORDS = [
    "product",
    "country",
    "region",
    "category",
    "expense category",
    "segment",
    "customer",
    "customer name",
    "city",
    "state",
    "department",
    "person",
    "salesperson",
    "sales person",
    "employee",
    "employee name",
    "name",
    "job role",
    "class",
    "subject",
    "teacher",
    "school",
    "hospital",
    "diagnosis",
    "doctor",
    "campaign",
    "channel",
    "service",
    "endpoint",
    "status code",
    "device",
    "device type",
    "location",
    "sensor",
    "vendor",
    "account",
    "account name",
    "priority",
    "student",
    "student name",
    "attrition",
    "overtime",
    "over time",
    "business travel",
    "education field",
    "gender",
    "marital status",
]

TIME_KEYWORDS = [
    "date",
    "time",
    "month",
    "year",
    "day",
    "week",
    "quarter",
    "hour",
    "timestamp",
    "created",
    "created at",
    "updated",
    "order date",
    "transaction date",
    "review date",
    "admission date",
    "hire date",
    "exam date",
]

TEXT_KEYWORDS = [
    "feedback",
    "review",
    "reviews",
    "comment",
    "comments",
    "text",
    "message",
    "messages",
    "log message",
    "log messages",
    "ticket",
    "tickets",
    "description",
    "descriptions",
    "ticket description",
    "ticket descriptions",
    "notes",
    "complaint",
    "complaints",
    "open-ended",
    "response",
    "responses",
    "support",
    "topic",
    "topics",
]

ID_KEYWORDS = [
    "id",
    "uuid",
    "guid",
    "key",
    "code",
    "phone",
    "mobile",
    "zip",
    "zipcode",
    "postal",
    "ssn",
    "number",
    "no",
    "identifier",
    "transaction id",
    "order id",
    "customer id",
    "user id",
    "employee id",
    "invoice id",
    "product id",
    "ticket id",
    "review id",
    "comment id",
    "campaign id",
    "patient id",
    "device id",
    "request id",
    "student id",
    "employee number",
]

ENTITY_ALLOW_KEYWORDS = [
    "customer",
    "user",
    "employee",
    "vendor",
    "supplier",
    "person",
    "sales person",
    "salesperson",
    "account",
    "student",
    "patient",
    "device",
    "campaign",
    "endpoint",
    "service",
    "doctor",
    "job role",
]

VALID_MEASURE_BUSINESS_TYPES = [
    "currency_or_amount",
    "quantity_or_count",
    "percentage",
    "rating_or_score",
    "numeric_measure",
]

TEXT_ONLY_OPERATIONS = [
    "text_summary",
    "sentiment_summary",
    "word_frequency",
]

DIMENSION_REQUIRED_OPERATIONS = [
    "groupby_sum",
    "groupby_sum_sort_desc",
    "groupby_mean",
    "groupby_mean_sort_desc",
    "diagnostic_analysis",
    "full_dataset_analysis",
]

CORRELATION_OPERATIONS = [
    "correlation",
    "correlation_heatmap",
]

SCALAR_OPERATIONS = [
    "sum",
    "mean",
    "max",
    "min",
    "distribution",
    "outlier_check",
]

TIME_OPERATIONS = [
    "time_groupby_sum",
    "forecast",
]


# ============================================================
# Basic Helpers
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text and split CamelCase column names.

    Examples:
    MonthlyIncome -> monthly income
    JobRole -> job role
    TotalWorkingYears -> total working years
    PercentSalaryHike -> percent salary hike
    """

    text = str(text)

    # Split CamelCase: MonthlyIncome -> Monthly Income
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    text = (
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

    # Common typo cleanup
    text = text.replace("inocme", "income")
    text = text.replace("incmoe", "income")
    text = text.replace("incoem", "income")
    text = text.replace("montly", "monthly")

    # Normalize repeated spaces
    text = re.sub(r"\s+", " ", text)

    return text


# Common plural-to-singular map for dimension/measure matching
_PLURAL_ALIASES = {
    "countries":   "country",
    "products":    "product",
    "regions":     "region",
    "categories":  "category",
    "customers":   "customer",
    "employees":   "employee",
    "departments": "department",
    "channels":    "channel",
    "campaigns":   "campaign",
    "segments":    "segment",
    "cities":      "city",
    "doctors":     "doctor",
    "hospitals":   "hospital",
    "services":    "service",
    "devices":     "device",
    "vendors":     "vendor",
    "suppliers":   "supplier",
    "students":    "student",
    "patients":    "patient",
    "accounts":    "account",
    "revenues":    "revenue",
    "profits":     "profit",
    "sales":       "sales",   # already singular
    "orders":      "orders",  # already singular
}


def normalize_for_matching(text: str) -> str:
    """
    Normalize text AND convert common plural forms to their singular base
    so 'countries' matches the 'country' keyword/column.
    """
    norm = normalize_text(text)
    for plural, singular in _PLURAL_ALIASES.items():
        # Replace whole-word plurals only (bounded by spaces or string edges)
        norm = re.sub(r'\b' + re.escape(plural) + r'\b', singular, norm)
    return norm


def compact_text(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def tokens(text: str) -> set:
    return set(normalize_text(text).split())


def get_col_name(col: Dict[str, Any]) -> str:
    return str(col.get("name", ""))


def get_col_lower_name(col: Dict[str, Any]) -> str:
    return normalize_text(get_col_name(col))


def get_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def question_mentions_any(question: str, keywords: List[str]) -> bool:
    q = normalize_text(question)
    q_depl = normalize_for_matching(question)
    return any(normalize_text(keyword) in q or normalize_text(keyword) in q_depl for keyword in keywords)


def column_name_has_any(column_name: str, keywords: List[str]) -> bool:
    col = normalize_text(column_name)
    return any(normalize_text(keyword) in col for keyword in keywords)


def get_columns(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    return metadata.get("columns", [])


def score_column_for_question(column_name: str, question: str, keywords: List[str]) -> int:
    """
    Score how well a column matches the user question.

    Important:
    - Token-aware matching prevents Age from matching inside 'average'.
    - Compact matching allows MonthlyIncome to match 'monthly income'.
    - Plural-normalised question matching: 'countries' matches 'country' column.
    """

    col = normalize_text(column_name)
    q = normalize_text(question)
    q_depl = normalize_for_matching(question)  # plural-normalised question

    col_compact = compact_text(column_name)
    q_compact = compact_text(question)

    col_tokens = tokens(column_name)
    q_tokens = tokens(question)
    q_depl_tokens = set(q_depl.split())

    score = 0

    if col and (col in q or col in q_depl):
        score += 40

    if col_compact and col_compact in q_compact:
        score += 40

    for token in col_tokens:
        if len(token) >= 3 and (token in q_tokens or token in q_depl_tokens):
            score += 8

    for keyword in keywords:
        keyword_norm = normalize_text(keyword)
        keyword_compact = compact_text(keyword)
        keyword_tokens = tokens(keyword)

        if keyword_norm in col and (keyword_norm in q or keyword_norm in q_depl):
            score += 12
        elif keyword_compact and keyword_compact in col_compact and keyword_compact in q_compact:
            score += 12
        elif keyword_norm in col:
            score += 3
        elif keyword_tokens and keyword_tokens.issubset(q_depl_tokens):
            score += 1

    # ── Column synonym boost ──────────────────────────────────────────────────
    # Handles abbreviated or compound column names (e.g. 'adr' → 'daily rate',
    # 'total_of_special_requests' → 'special requests').
    for col_pattern, synonyms in COLUMN_KEYWORD_SYNONYMS.items():
        if col_pattern == col or col_pattern in col or col in col_pattern:
            for syn in synonyms:
                syn_norm = normalize_text(syn)
                if syn_norm in q or syn_norm in q_depl:
                    score += 45
                    break

    return score


# ============================================================
# Risk / Preference Helpers
# ============================================================

def is_identifier_column(col: Dict[str, Any]) -> bool:
    name = get_col_lower_name(col)

    if col.get("role") == "id":
        return True

    if col.get("business_type") == "identifier":
        return True

    if col.get("cardinality_type") in ["high_unique_numeric", "high_cardinality_category"]:
        if column_name_has_any(name, ID_KEYWORDS):
            return True

    return False


def is_zip_or_phone_like(col: Dict[str, Any]) -> bool:
    name = get_col_lower_name(col)

    risky_keywords = [
        "zip",
        "zipcode",
        "postal",
        "phone",
        "mobile",
        "telephone",
        "ssn",
    ]

    return column_name_has_any(name, risky_keywords)


def is_good_measure_business_type(col: Dict[str, Any]) -> bool:
    return col.get("business_type") in VALID_MEASURE_BUSINESS_TYPES


def is_good_dimension_business_type(col: Dict[str, Any]) -> bool:
    return col.get("business_type") in [
        "geography",
        "categorical_dimension",
        "boolean_flag",
        "rating_or_score",
    ]


def is_valid_numeric_measure(col: Dict[str, Any]) -> bool:
    return (
        col.get("semantic_type") == "numeric"
        and col.get("business_type") in VALID_MEASURE_BUSINESS_TYPES
    )


def question_allows_high_cardinality_entity(question: str) -> bool:
    return question_mentions_any(question, ENTITY_ALLOW_KEYWORDS)


def question_explicitly_requests_text_grouping(question: str) -> bool:
    q = normalize_text(question)

    text_grouping_phrases = [
        "by product",
        "by category",
        "by priority",
        "by country",
        "by region",
        "by segment",
        "by customer",
        "by channel",
        "by department",
        "by service",
        "by endpoint",
        "by status code",
        "by doctor",
        "by job role",
    ]

    return any(phrase in q for phrase in text_grouping_phrases)


def is_text_only_operation(operation: str) -> bool:
    return operation in TEXT_ONLY_OPERATIONS


def is_correlation_operation(operation: str) -> bool:
    return operation in CORRELATION_OPERATIONS


def is_scalar_operation(operation: str) -> bool:
    return operation in SCALAR_OPERATIONS


def is_time_operation(operation: str) -> bool:
    return operation in TIME_OPERATIONS


# ============================================================
# Explicit Column Overrides
# ============================================================

def get_explicit_measure_column(metadata: Dict[str, Any], question: str) -> Optional[str]:
    """
    If the user explicitly mentions a measure, choose that column directly.
    """

    q = normalize_text(question)
    q_compact = compact_text(question)

    explicit_measure_map = {
        "monthly income": ["monthly income", "monthlyincome"],
        "monthly rate": ["monthly rate", "monthlyrate"],
        "daily rate": ["daily rate", "dailyrate"],
        "hourly rate": ["hourly rate", "hourlyrate"],
        "percent salary hike": ["percent salary hike", "percentsalaryhike", "salary hike"],
        "total working years": ["total working years", "totalworkingyears", "working years"],
        "years at company": ["years at company", "yearsatcompany"],
        "years in current role": ["years in current role", "yearsincurrentrole"],
        "years since last promotion": ["years since last promotion", "yearssincelastpromotion"],
        "years with current manager": [
            "years with current manager",
            "yearswithcurrentmanager",
            "years with curr manager",
            "yearswithcurrmanager",
        ],
        "training times last year": ["training times last year", "trainingtimeslastyear"],
        "distance from home": ["distance from home", "distancefromhome"],
        "environment satisfaction": ["environment satisfaction", "environmentsatisfaction"],
        "job involvement": ["job involvement", "jobinvolvement"],
        "job level": ["job level", "joblevel"],
        "job satisfaction": ["job satisfaction", "jobsatisfaction"],
        "performance rating": ["performance rating", "performancerating"],
        "relationship satisfaction": ["relationship satisfaction", "relationshipsatisfaction"],
        "stock option level": ["stock option level", "stockoptionlevel"],
        "work life balance": ["work life balance", "worklifebalance"],
        "num companies worked": ["num companies worked", "numcompaniesworked", "number of companies worked"],
        "age": ["age"],
    }

    for _, aliases in explicit_measure_map.items():
        question_matches_measure = any(
            normalize_text(alias) in q or compact_text(alias) in q_compact
            for alias in aliases
        )

        if not question_matches_measure:
            continue

        for col in get_columns(metadata):
            col_name = get_col_name(col)
            col_norm = normalize_text(col_name)
            col_compact = compact_text(col_name)

            if col.get("role") == "id":
                continue

            if col.get("business_type") == "identifier":
                continue

            if col.get("semantic_type") != "numeric":
                continue

            for alias in aliases:
                alias_norm = normalize_text(alias)
                alias_compact = compact_text(alias)

                if alias_norm == col_norm or alias_compact == col_compact:
                    return col_name

    return None


def get_explicit_dimension_column(metadata: Dict[str, Any], question: str) -> Optional[str]:
    """
    If the user explicitly asks for a dimension in the question,
    choose the matching column directly before generic scoring.
    """

    q = normalize_text(question)
    q_depl = normalize_for_matching(question)  # plural-normalised form

    explicit_dimension_map = {
        "doctor": ["doctor"],
        "job role": ["job role"],
        "status code": ["status code"],
        "priority": ["priority"],
        "hospital": ["hospital"],
        "diagnosis": ["diagnosis"],
        "service": ["service"],
        "endpoint": ["endpoint"],
        "device type": ["device type"],
        "location": ["location"],
        "department": ["department"],
        "product": ["product"],
        "country": ["country"],
        "region": ["region"],
        "channel": ["channel"],
        "campaign": ["campaign"],
        "category": ["category"],
        "customer": ["customer name", "customer"],
        "employee": ["employee name", "employee"],
        "student": ["student name", "student"],
        "account": ["account name", "account"],
        "expense category": ["expense category"],
        "class": ["class"],
        "subject": ["subject"],
        "attrition": ["attrition"],
        "overtime": ["over time", "overtime"],
        "business travel": ["business travel"],
        "education field": ["education field"],
        "gender": ["gender"],
        "marital status": ["marital status"],
    }

    dimension_request_patterns = [
        " by ",
        " across ",
        "which ",
        "top ",
        "rank ",
        "compare ",
        "show ",
        "break down ",
        "breakdown ",
    ]

    if not any(pattern in q or pattern in q_depl for pattern in dimension_request_patterns):
        return None

    for requested_dimension, column_patterns in explicit_dimension_map.items():
        if requested_dimension in q or requested_dimension in q_depl:
            for col in get_columns(metadata):
                col_name = get_col_lower_name(col)

                if col.get("semantic_type") not in ["category", "boolean"]:
                    continue

                if col.get("role") == "id":
                    continue

                if col.get("business_type") == "identifier":
                    continue

                for pattern in column_patterns:
                    if pattern in col_name:
                        return get_col_name(col)

    return None


# ============================================================
# Scoring Helpers
# ============================================================

def question_mentions_measure_column(col: Dict[str, Any], question: str) -> bool:
    q = normalize_text(question)
    q_compact = compact_text(question)

    col_name = get_col_lower_name(col)
    col_compact = compact_text(get_col_name(col))

    if col_name and col_name in q:
        return True

    if col_compact and col_compact in q_compact:
        return True

    q_tokens = tokens(question)
    for token in tokens(get_col_name(col)):
        if len(token) >= 3 and token in q_tokens:
            return True

    return False


def get_first_mentioned_measure_column(metadata: Dict[str, Any], question: str) -> Optional[str]:
    q = normalize_text(question)
    q_compact = compact_text(question)
    candidates = []

    for col in get_columns(metadata):
        if col.get("semantic_type") != "numeric":
            continue

        if col.get("role") == "id":
            continue

        if col.get("business_type") == "identifier":
            continue

        col_name = get_col_name(col)
        col_norm = normalize_text(col_name)
        col_compact = compact_text(col_name)

        if col_norm and col_norm in q:
            candidates.append((q.index(col_norm), col_name))
        elif col_compact and col_compact in q_compact:
            candidates.append((q_compact.index(col_compact), col_name))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def get_entity_dimension_boost(col: Dict[str, Any], question: str) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    q = normalize_text(question)
    name = get_col_lower_name(col)

    entity_pairs = [
        ("employee", ["employee name", "employee"]),
        ("customer", ["customer name", "customer"]),
        ("patient", ["patient name", "patient"]),
        ("student", ["student name", "student"]),
        ("vendor", ["vendor name", "vendor"]),
        ("supplier", ["supplier name", "supplier"]),
        ("device", ["device type", "device name", "device"]),
        ("campaign", ["campaign name", "campaign"]),
        ("service", ["service"]),
        ("endpoint", ["endpoint"]),
        ("hospital", ["hospital"]),
        ("diagnosis", ["diagnosis"]),
        ("doctor", ["doctor"]),
        ("job role", ["job role"]),
        ("channel", ["channel"]),
        ("category", ["category"]),
        ("expense category", ["expense category"]),
        ("priority", ["priority"]),
        ("location", ["location"]),
        ("status code", ["status code"]),
        ("class", ["class"]),
        ("subject", ["subject"]),
        ("account", ["account name", "account"]),
        ("department", ["department"]),
        ("attrition", ["attrition"]),
        ("overtime", ["over time", "overtime"]),
    ]

    question_is_entity_level = (
        q.startswith("which ")
        or q.startswith("who ")
        or "highest" in q
        or "lowest" in q
        or "most" in q
        or "top" in q
        or "rank" in q
        or " by " in q
        or " across " in q
        or "compare" in q
        or "show" in q
        or "break down" in q
    )

    if not question_is_entity_level:
        return score, reasons

    for entity_word, column_patterns in entity_pairs:
        if entity_word in q:
            for pattern in column_patterns:
                if pattern in name:
                    boost = 70

                    if "name" in name:
                        boost += 20

                    if is_identifier_column(col):
                        boost -= 50

                    score += boost
                    reasons.append(f"entity requested: {entity_word} boost +{boost}")
                    return score, reasons

    return score, reasons


def get_domain_measure_boost(col: Dict[str, Any], question: str) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    q = normalize_text(question)
    name = get_col_lower_name(col)

    if "traffic" in q and ("log" in q or "logs" in q or "anomal" in q):
        if "request count" in name:
            score += 45
            reasons.append("traffic anomaly prefers request count +45")

        if "error count" in name and "error" not in q and "failure" not in q:
            score -= 20
            reasons.append("traffic anomaly without error wording penalizes error count -20")

    if ("error" in q or "errors" in q or "failure" in q or "failures" in q) and "error count" in name:
        score += 35
        reasons.append("error/failure wording prefers error count +35")

    direct_measure_pairs = [
        ("latency", "latency"),
        ("request count", "request count"),
        ("resolution time", "resolution time"),
        ("length of stay", "length of stay"),
        ("treatment cost", "treatment cost"),
        ("patient count", "patient count"),
        ("performance score", "performance score"),
        ("attrition rate", "attrition rate"),
        ("conversion rate", "conversion rate"),
        ("monthly income", "monthly income"),
        ("monthly rate", "monthly rate"),
        ("daily rate", "daily rate"),
        ("hourly rate", "hourly rate"),
        ("percent salary hike", "percent salary hike"),
        ("total working years", "total working years"),
        ("years at company", "years at company"),
        ("years in current role", "years in current role"),
        ("years since last promotion", "years since last promotion"),
        ("years with curr manager", "years with curr manager"),
        ("training times last year", "training times last year"),
        ("distance from home", "distance from home"),
        ("environment satisfaction", "environment satisfaction"),
        ("job involvement", "job involvement"),
        ("job level", "job level"),
        ("job satisfaction", "job satisfaction"),
        ("performance rating", "performance rating"),
        ("relationship satisfaction", "relationship satisfaction"),
        ("stock option level", "stock option level"),
        ("work life balance", "work life balance"),
        ("num companies worked", "num companies worked"),
    ]

    for phrase, col_phrase in direct_measure_pairs:
        if phrase in q and col_phrase in name:
            score += 60
            reasons.append(f"{phrase} question boost +60")

    return score, reasons


# ============================================================
# Candidate Scoring
# ============================================================

def score_measure_candidate(
    col: Dict[str, Any],
    question: str,
    operation: Optional[str] = None,
) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    name = get_col_name(col)
    semantic_type = col.get("semantic_type")
    role = col.get("role")
    business_type = col.get("business_type")
    cardinality_type = col.get("cardinality_type")
    unique_ratio = get_float(col.get("unique_ratio"), 0.0)

    if role == "measure":
        score += 30
        reasons.append("role=measure +30")

    if semantic_type == "numeric":
        score += 25
        reasons.append("semantic_type=numeric +25")

    if is_good_measure_business_type(col):
        score += 20
        reasons.append(f"business_type={business_type} +20")

    if cardinality_type == "continuous_or_measure":
        score += 10
        reasons.append("continuous_or_measure +10")

    question_score = score_column_for_question(name, question, MEASURE_KEYWORDS)
    score += question_score

    if question_score:
        reasons.append(f"question/name match +{question_score}")

    lower_name = normalize_text(name)

    if lower_name in ["sales", "revenue", "amount"]:
        score += 8
        reasons.append("common primary metric +8")
    elif lower_name in ["profit", "income", "expense", "cost", "spend", "salary", "bonus", "monthly income"]:
        score += 8
        reasons.append("common financial metric +8")
    elif lower_name in ["quantity", "qty", "orders", "count", "clicks", "request count", "error count"]:
        score += 5
        reasons.append("common count/quantity metric +5")

    domain_boost, domain_reasons = get_domain_measure_boost(col, question)
    score += domain_boost
    reasons.extend(domain_reasons)

    if is_correlation_operation(operation or "") and question_mentions_measure_column(col, question):
        score += 35
        reasons.append("correlation explicit measure mention +35")

    if is_identifier_column(col):
        score -= 60
        reasons.append("identifier/id risk -60")

    if is_zip_or_phone_like(col):
        score -= 50
        reasons.append("zip/phone-like column -50")

    if business_type == "identifier":
        score -= 40
        reasons.append("business_type=identifier -40")

    if cardinality_type == "high_unique_numeric" and not is_good_measure_business_type(col):
        if not question_mentions_measure_column(col, question):
            score -= 25
            reasons.append("high_unique_numeric without metric business type -25")

    if unique_ratio > 0.95 and business_type == "numeric_measure" and not is_valid_numeric_measure(col):
        if not question_mentions_measure_column(col, question):
            score -= 10
            reasons.append("very high unique ratio generic numeric -10")

    return score, reasons


def score_dimension_candidate(col: Dict[str, Any], question: str) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    name = get_col_name(col)
    semantic_type = col.get("semantic_type")
    role = col.get("role")
    business_type = col.get("business_type")
    cardinality_type = col.get("cardinality_type")
    unique_ratio = get_float(col.get("unique_ratio"), 0.0)

    q = normalize_text(question)
    lower_name = normalize_text(name)

    if role == "dimension":
        score += 30
        reasons.append("role=dimension +30")

    if semantic_type in ["category", "boolean"]:
        score += 20
        reasons.append(f"semantic_type={semantic_type} +20")

    if is_good_dimension_business_type(col):
        score += 15
        reasons.append(f"business_type={business_type} +15")

    if cardinality_type == "low_cardinality_category":
        score += 18
        reasons.append("low_cardinality_category +18")

    if cardinality_type == "medium_cardinality_category":
        score += 10
        reasons.append("medium_cardinality_category +10")

    if cardinality_type == "boolean":
        score += 8
        reasons.append("boolean cardinality +8")

    question_score = score_column_for_question(name, question, DIMENSION_KEYWORDS)
    score += question_score

    if question_score:
        reasons.append(f"question/name match +{question_score}")

    entity_boost, entity_reasons = get_entity_dimension_boost(col, question)
    score += entity_boost
    reasons.extend(entity_reasons)

    if "who" in q and column_name_has_any(
        lower_name,
        ["customer", "person", "sales person", "salesperson", "employee", "name", "doctor"],
    ):
        score += 20
        reasons.append("who-question entity boost +20")

    direct_dimension_boosts = [
        ("product", "product"),
        ("country", "country"),
        ("region", "region"),
        ("department", "department"),
        ("category", "category"),
        ("customer", "customer"),
        ("priority", "priority"),
        ("doctor", "doctor"),
        ("job role", "job role"),
        ("status code", "status code"),
        ("device type", "device type"),
        ("location", "location"),
        ("class", "class"),
        ("subject", "subject"),
        ("hospital", "hospital"),
        ("diagnosis", "diagnosis"),
        ("channel", "channel"),
        ("campaign", "campaign"),
        ("service", "service"),
        ("endpoint", "endpoint"),
        ("attrition", "attrition"),
        ("overtime", "over time"),
        ("over time", "over time"),
        ("business travel", "business travel"),
        ("education field", "education field"),
        ("gender", "gender"),
        ("marital status", "marital status"),
    ]

    # Also build a plural-normalised version of the question for boost matching
    q_depl = normalize_for_matching(question)

    for question_phrase, column_phrase in direct_dimension_boosts:
        if (question_phrase in q or question_phrase in q_depl) and column_phrase in lower_name:
            score += 25
            reasons.append(f"{question_phrase} dimension boost +25")

    high_cardinality = (
        cardinality_type == "high_cardinality_category"
        or unique_ratio > 0.8
    )

    if is_identifier_column(col):
        if question_allows_high_cardinality_entity(question):
            score -= 10
            reasons.append("identifier risk but entity requested -10")
        else:
            score -= 70
            reasons.append("identifier/id dimension risk -70")

    if high_cardinality and not question_allows_high_cardinality_entity(question):
        score -= 35
        reasons.append("high-cardinality dimension without entity request -35")

    if is_zip_or_phone_like(col):
        score -= 50
        reasons.append("zip/phone-like dimension risk -50")

    if business_type == "identifier":
        score -= 35
        reasons.append("business_type=identifier -35")

    return score, reasons


def score_time_candidate(col: Dict[str, Any], question: str) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    name = get_col_name(col)
    semantic_type = col.get("semantic_type")
    role = col.get("role")
    business_type = col.get("business_type")

    if role == "time":
        score += 35
        reasons.append("role=time +35")

    if semantic_type == "datetime":
        score += 30
        reasons.append("semantic_type=datetime +30")

    if business_type == "date_or_time":
        score += 25
        reasons.append("business_type=date_or_time +25")

    question_score = score_column_for_question(name, question, TIME_KEYWORDS)
    score += question_score

    if question_score:
        reasons.append(f"question/name match +{question_score}")

    lower_name = normalize_text(name)

    if "date" in lower_name:
        score += 8
        reasons.append("date name boost +8")

    if "timestamp" in lower_name:
        score += 8
        reasons.append("timestamp name boost +8")

    if "created at" in lower_name:
        score += 8
        reasons.append("created at name boost +8")

    return score, reasons


def score_text_candidate(col: Dict[str, Any], question: str) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    name = get_col_name(col)
    role = col.get("role")
    semantic_type = col.get("semantic_type")
    business_type = col.get("business_type")

    if role == "text":
        score += 35
        reasons.append("role=text +35")

    if semantic_type == "text":
        score += 30
        reasons.append("semantic_type=text +30")

    if business_type == "free_text":
        score += 25
        reasons.append("business_type=free_text +25")

    question_score = score_column_for_question(name, question, TEXT_KEYWORDS)
    score += question_score

    if question_score:
        reasons.append(f"question/name match +{question_score}")

    lower_name = normalize_text(name)

    if column_name_has_any(lower_name, TEXT_KEYWORDS):
        score += 10
        reasons.append("text keyword in column name +10")

    return score, reasons


# ============================================================
# Candidate Selection
# ============================================================

def choose_best_candidate(
    metadata: Dict[str, Any],
    question: str,
    candidate_type: str,
    operation: Optional[str] = None,
) -> Tuple[Optional[str], float, List[Dict[str, Any]], List[str]]:
    columns = metadata.get("columns", [])
    scored_candidates = []
    warnings = []

    # Direct explicit measure override.
    if candidate_type == "measure":
        explicit_measure = get_explicit_measure_column(metadata, question)

        if explicit_measure:
            candidate_debug = []

            for col in columns:
                score, reasons = score_measure_candidate(col, question, operation=operation)

                if get_col_name(col) == explicit_measure:
                    score += 300
                    reasons.append("explicit measure override +300")

                candidate_debug.append(
                    {
                        "column": get_col_name(col),
                        "score": score,
                        "role": col.get("role"),
                        "semantic_type": col.get("semantic_type"),
                        "business_type": col.get("business_type"),
                        "cardinality_type": col.get("cardinality_type"),
                        "unique_ratio": col.get("unique_ratio"),
                        "reasons": reasons,
                    }
                )

            candidate_debug.sort(key=lambda item: item["score"], reverse=True)
            return explicit_measure, 1.0, candidate_debug, warnings

    # Direct explicit dimension override.
    if candidate_type == "dimension":
        explicit_dimension = get_explicit_dimension_column(metadata, question)

        if explicit_dimension:
            candidate_debug = []

            for col in columns:
                score, reasons = score_dimension_candidate(col, question)

                if get_col_name(col) == explicit_dimension:
                    score += 150
                    reasons.append("explicit dimension override +150")

                candidate_debug.append(
                    {
                        "column": get_col_name(col),
                        "score": score,
                        "role": col.get("role"),
                        "semantic_type": col.get("semantic_type"),
                        "business_type": col.get("business_type"),
                        "cardinality_type": col.get("cardinality_type"),
                        "unique_ratio": col.get("unique_ratio"),
                        "reasons": reasons,
                    }
                )

            candidate_debug.sort(key=lambda item: item["score"], reverse=True)
            return explicit_dimension, 1.0, candidate_debug, warnings

    # Correlation special case.
    if candidate_type == "measure" and is_correlation_operation(operation or ""):
        first_mentioned = get_first_mentioned_measure_column(metadata, question)

        if first_mentioned:
            candidate_debug = []

            for col in columns:
                score, reasons = score_measure_candidate(col, question, operation=operation)

                if get_col_name(col) == first_mentioned:
                    score += 100
                    reasons.append("first-mentioned correlation measure override +100")

                candidate_debug.append(
                    {
                        "column": get_col_name(col),
                        "score": score,
                        "role": col.get("role"),
                        "semantic_type": col.get("semantic_type"),
                        "business_type": col.get("business_type"),
                        "cardinality_type": col.get("cardinality_type"),
                        "unique_ratio": col.get("unique_ratio"),
                        "reasons": reasons,
                    }
                )

            candidate_debug.sort(key=lambda item: item["score"], reverse=True)
            return first_mentioned, 1.0, candidate_debug, warnings

    for col in columns:
        if candidate_type == "measure":
            score, reasons = score_measure_candidate(col, question, operation=operation)
        elif candidate_type == "dimension":
            score, reasons = score_dimension_candidate(col, question)
        elif candidate_type == "time":
            score, reasons = score_time_candidate(col, question)
        elif candidate_type == "text":
            score, reasons = score_text_candidate(col, question)
        else:
            continue

        scored_candidates.append(
            {
                "column": get_col_name(col),
                "score": score,
                "role": col.get("role"),
                "semantic_type": col.get("semantic_type"),
                "business_type": col.get("business_type"),
                "cardinality_type": col.get("cardinality_type"),
                "unique_ratio": col.get("unique_ratio"),
                "reasons": reasons,
            }
        )

    scored_candidates.sort(key=lambda item: item["score"], reverse=True)

    if not scored_candidates:
        warnings.append(f"No candidates found for {candidate_type}.")
        return None, 0.0, [], warnings

    best = scored_candidates[0]
    best_score = best["score"]

    min_score_threshold = {
        "measure": 25,
        "dimension": 20,
        "time": 20,
        "text": 20,
    }.get(candidate_type, 20)

    if best_score < min_score_threshold:
        warnings.append(
            f"Best {candidate_type} candidate is weak: "
            f"{best['column']} scored {best_score}."
        )
        return None, 0.0, scored_candidates, warnings

    second_score = scored_candidates[1]["score"] if len(scored_candidates) > 1 else 0
    margin = best_score - second_score

    confidence = min(1.0, max(0.0, (best_score / 80.0) + (margin / 100.0)))

    if best.get("business_type") == "identifier":
        warnings.append(
            f"Selected {candidate_type} column may be identifier-like: {best['column']}."
        )

    return best["column"], round(confidence, 3), scored_candidates, warnings


# ============================================================
# Public Selection Functions
# ============================================================

def choose_best_measure_column(metadata: Dict[str, Any], question: str) -> Optional[str]:
    selected, _, _, _ = choose_best_candidate(metadata, question, "measure")
    return selected


def choose_best_dimension_column(metadata: Dict[str, Any], question: str) -> Optional[str]:
    selected, _, _, _ = choose_best_candidate(metadata, question, "dimension")
    return selected


def choose_best_time_column(metadata: Dict[str, Any], question: str) -> Optional[str]:
    selected, _, _, _ = choose_best_candidate(metadata, question, "time")
    return selected


def choose_text_column(metadata: Dict[str, Any], question: str) -> Optional[str]:
    selected, _, _, _ = choose_best_candidate(metadata, question, "text")
    return selected


def infer_time_grain(question: str) -> str:
    q = normalize_text(question)

    if "hourly" in q or "per hour" in q or "by hour" in q:
        return "hour"

    if "daily" in q or "by day" in q or "per day" in q:
        return "day"

    if "weekly" in q or "by week" in q or "per week" in q:
        return "week"

    if "yearly" in q or "annual" in q or "by year" in q:
        return "year"

    if "quarterly" in q or "quarter" in q:
        return "quarter"

    return "month"


# ============================================================
# Validation
# ============================================================

def validate_mapped_plan(mapped_plan: Dict[str, Any]) -> bool:
    operation = mapped_plan.get("operation")
    selected = mapped_plan.get("selected_columns", {})

    measure = selected.get("measure_column")
    dimension = selected.get("dimension_column")
    time_col = selected.get("time_column")
    text_col = selected.get("text_column")

    if operation in ["none"]:
        return False

    if operation in [
        "count_rows",
        "list_columns",
        "null_check",
        "duplicate_check",
        "data_quality_summary",
    ]:
        return True

    if operation in SCALAR_OPERATIONS:
        return measure is not None

    if operation in [
        "groupby_sum",
        "groupby_sum_sort_desc",
        "groupby_mean",
        "groupby_mean_sort_desc",
    ]:
        return measure is not None and dimension is not None

    if operation in TIME_OPERATIONS:
        return measure is not None and time_col is not None

    if operation in CORRELATION_OPERATIONS:
        return measure is not None

    if operation in TEXT_ONLY_OPERATIONS:
        return text_col is not None

    if operation in ["full_dataset_analysis", "diagnostic_analysis"]:
        return measure is not None

    return True


def get_validation_messages(mapped_plan: Dict[str, Any]) -> List[str]:
    messages = []

    operation = mapped_plan.get("operation")
    selected = mapped_plan.get("selected_columns", {})

    measure = selected.get("measure_column")
    dimension = selected.get("dimension_column")
    time_col = selected.get("time_column")
    text_col = selected.get("text_column")

    if operation in SCALAR_OPERATIONS and not measure:
        messages.append("No numeric measure column found for this operation.")

    if operation in [
        "groupby_sum",
        "groupby_sum_sort_desc",
        "groupby_mean",
        "groupby_mean_sort_desc",
    ]:
        if not measure:
            messages.append("No numeric measure column found for group-by operation.")
        if not dimension:
            messages.append("No categorical dimension column found for group-by operation.")

    if operation in TIME_OPERATIONS:
        if not measure:
            messages.append("No numeric measure column found for time-series operation.")
        if not time_col:
            messages.append("No datetime column found for time-series operation.")

    if operation in TEXT_ONLY_OPERATIONS and not text_col:
        messages.append("No text column found for text analysis operation.")

    if operation in ["full_dataset_analysis", "diagnostic_analysis"] and not measure:
        messages.append("No numeric measure column found for full analysis.")

    if not messages:
        messages.append("Column mapping is valid.")

    return messages


# ============================================================
# Main Public Function
# ============================================================

def map_columns_to_plan(
    question: str,
    metadata: Dict[str, Any],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    mapped_plan = dict(plan)

    operation = plan.get("operation")
    required_roles = plan.get("required_data_roles", {})

    needs_numeric = required_roles.get("needs_numeric", False)
    needs_category = required_roles.get("needs_category", False)
    needs_datetime = required_roles.get("needs_datetime", False)
    needs_text = required_roles.get("needs_text", False)

    selected_columns = {
        "measure_column": None,
        "dimension_column": None,
        "time_column": None,
        "text_column": None,
    }

    mapping_confidence = {
        "measure_column": 0.0,
        "dimension_column": 0.0,
        "time_column": 0.0,
        "text_column": 0.0,
    }

    candidate_scores = {
        "measure_column": [],
        "dimension_column": [],
        "time_column": [],
        "text_column": [],
    }

    mapping_warnings = []

    # -------------------------
    # Measure mapping
    # -------------------------

    should_map_measure = (
        operation not in TEXT_ONLY_OPERATIONS
        and (
            needs_numeric
            or operation in [
                "sum",
                "mean",
                "max",
                "min",
                "groupby_sum",
                "groupby_sum_sort_desc",
                "groupby_mean",
                "groupby_mean_sort_desc",
                "time_groupby_sum",
                "correlation",
                "correlation_heatmap",
                "distribution",
                "outlier_check",
                "forecast",
                "diagnostic_analysis",
                "full_dataset_analysis",
            ]
        )
    )

    if should_map_measure:
        selected, confidence, scores, warnings = choose_best_candidate(
            metadata,
            question,
            "measure",
            operation=operation,
        )
        selected_columns["measure_column"] = selected
        mapping_confidence["measure_column"] = confidence
        candidate_scores["measure_column"] = scores
        mapping_warnings.extend(warnings)

    # -------------------------
    # Dimension mapping
    # -------------------------

    should_map_dimension = (
        operation not in TEXT_ONLY_OPERATIONS
        and operation not in SCALAR_OPERATIONS
        and operation not in TIME_OPERATIONS
        and operation not in CORRELATION_OPERATIONS
        and (
            needs_category
            or operation in DIMENSION_REQUIRED_OPERATIONS
        )
    )

    if operation in TEXT_ONLY_OPERATIONS and question_explicitly_requests_text_grouping(question):
        should_map_dimension = True

    if should_map_dimension:
        selected, confidence, scores, warnings = choose_best_candidate(
            metadata,
            question,
            "dimension",
            operation=operation,
        )
        selected_columns["dimension_column"] = selected
        mapping_confidence["dimension_column"] = confidence
        candidate_scores["dimension_column"] = scores
        mapping_warnings.extend(warnings)

    # -------------------------
    # Time mapping
    # -------------------------

    if needs_datetime or operation in TIME_OPERATIONS or operation in [
        "diagnostic_analysis",
        "full_dataset_analysis",
    ]:
        selected, confidence, scores, warnings = choose_best_candidate(
            metadata,
            question,
            "time",
            operation=operation,
        )
        selected_columns["time_column"] = selected
        mapping_confidence["time_column"] = confidence
        candidate_scores["time_column"] = scores
        mapping_warnings.extend(warnings)

    # -------------------------
    # Text mapping
    # -------------------------

    if needs_text or operation in TEXT_ONLY_OPERATIONS:
        selected, confidence, scores, warnings = choose_best_candidate(
            metadata,
            question,
            "text",
            operation=operation,
        )
        selected_columns["text_column"] = selected
        mapping_confidence["text_column"] = confidence
        candidate_scores["text_column"] = scores
        mapping_warnings.extend(warnings)

    mapped_plan["selected_columns"] = selected_columns
    mapped_plan["time_grain"] = infer_time_grain(question)
    mapped_plan["mapping_confidence"] = mapping_confidence
    mapped_plan["mapping_warnings"] = mapping_warnings
    mapped_plan["candidate_scores"] = candidate_scores
    mapped_plan["is_executable"] = validate_mapped_plan(mapped_plan)
    mapped_plan["validation_messages"] = get_validation_messages(mapped_plan)

    return mapped_plan


# ============================================================
# Target Column Detection
# ============================================================

TARGET_OUTCOME_KEYWORDS = [
    "attrition",
    "churn",
    "churned",
    "conversion",
    "converted",
    "default",
    "defaulted",
    "fraud",
    "fraudulent",
    "dropout",
    "drop out",
    "cancellation",
    "cancelled",
    "canceled",
    "cancel",
    "lapse",
    "lapsed",
    "leaving",
    "resigned",
    "terminated",
    "purchased",
    "responded",
    "subscribed",
    "unsubscribed",
    "outcome",
    "result",
    "target",
    "label",
    "flag",
    "indicator",
    "status",
    "left",
    "exited",
    "retained",
]

TARGET_QUESTION_CAUSAL_SIGNALS = [
    "affect",
    "affects",
    "impact",
    "impacts",
    "influence",
    "influences",
    "cause",
    "causes",
    "drive",
    "drives",
    "predict",
    "predicts",
    "relate",
    "relates",
    "why",
    "reason",
    "factor",
    "factors",
    "leading to",
    "result in",
    "associated with",
    "linked to",
    "correlated with",
    "increase",
    "increases",
    "rising",
    "less than",
    "more than",
    "higher than",
    "lower than",
    "more likely",
    "less likely",
    "compared to",
    "compare",
    "differ",
    "differs",
    "difference",
    "do",
    "does",
]

TARGET_RATE_OPERATIONS = [
    "groupby_target_rate",
    "groupby_target_rate_sort_desc",
]

CAUSAL_INTENT_TYPES = [
    "diagnostic_analysis",
]

BINARY_CARDINALITY_TYPES = [
    "binary_category",
    "binary_numeric",
    "boolean",
]

# ── Compound "X rate" phrases that always trigger target selection ─────────────
# Maps the phrase in the question → list of substrings that should appear in the
# target column name (normalized, underscore-free).
COMPOUND_RATE_SIGNALS: Dict[str, List[str]] = {
    "cancellation rate": ["cancel", "is canceled", "canceled", "cancelled"],
    "cancel rate":       ["cancel", "is canceled", "canceled", "cancelled"],
    "churn rate":        ["churn", "churned"],
    "attrition rate":    ["attrition"],
    "fraud rate":        ["fraud"],
    "default rate":      ["default"],
    "dropout rate":      ["dropout", "drop out"],
    "conversion rate":   ["conversion", "converted"],
    "retention rate":    ["retain", "retained"],
    "response rate":     ["responded", "response"],
}

# ── Column-level keyword synonyms ─────────────────────────────────────────────
# Maps a column name pattern (normalized) → question phrases that should boost it.
# Used in score_column_for_question() to handle abbreviations / compound names.
COLUMN_KEYWORD_SYNONYMS: Dict[str, List[str]] = {
    "adr":                         ["daily rate", "average daily rate", "rate per night", "room rate", "nightly rate"],
    "total of special requests":   ["special requests", "special request", "extra requests", "requests"],
    "total special requests":      ["special requests", "special request", "extra requests"],
    "num special requests":        ["special requests", "special request"],
    "is canceled":                 ["cancellation", "canceled", "cancelled", "cancel rate", "cancellation rate"],
    "is cancelled":                ["cancellation", "canceled", "cancelled", "cancel rate"],
    "reservation status":          ["booking status", "reservation status"],
    "arrival date":                ["arrival", "check in", "check-in"],
    "stays in week nights":        ["weeknight", "week night stays", "weekday stays"],
    "stays in weekend nights":     ["weekend stays", "weekend nights"],
    "previous cancellations":      ["past cancellations", "prior cancellations"],
    "days in waiting list":        ["waiting list", "wait time", "wait days"],
    "required car parking spaces": ["parking", "car park", "parking spaces"],
    "percent salary hike":         ["salary hike", "pay raise", "raise", "salary increase", "hike"],
    "monthly income":              ["income", "salary", "pay", "earnings", "compensation"],
    "years at company":            ["tenure", "company tenure", "time at company", "seniority"],
    "total working years":         ["experience", "work experience", "years of experience"],
    "num companies worked":        ["job changes", "companies worked", "previous employers"],
    "invoice date":                ["order date", "purchase date", "transaction date", "sale date"],
    "unit price":                  ["price", "product price", "item price", "cost per unit"],
    "stock code":                  ["product code", "item code", "sku"],
}


def is_binary_or_boolean_column(col: Dict[str, Any]) -> bool:
    """
    Return True if a column looks like a binary outcome (Yes/No, 0/1, True/False).
    """
    semantic_type = col.get("semantic_type", "")
    cardinality_type = col.get("cardinality_type", "")
    business_type = col.get("business_type", "")

    if semantic_type == "boolean":
        return True

    if cardinality_type in BINARY_CARDINALITY_TYPES:
        return True

    if business_type == "boolean_flag":
        return True

    return False


def score_target_candidate(col: Dict[str, Any], question: str) -> Tuple[int, List[str]]:
    """
    Score a column as a potential binary target/outcome column.

    A target column is a binary outcome such as:
    - Attrition (Yes/No)
    - Churn (1/0)
    - Conversion (True/False)
    - Default (Yes/No)
    """
    score = 0
    reasons = []

    name = get_col_name(col)
    name_norm = normalize_text(name)
    name_compact = compact_text(name)

    q = normalize_text(question)
    q_compact = compact_text(question)

    # Binary/boolean type is a strong prior
    if is_binary_or_boolean_column(col):
        score += 40
        reasons.append("binary/boolean column +40")

    # Name matches a known target outcome keyword
    for keyword in TARGET_OUTCOME_KEYWORDS:
        kw_norm = normalize_text(keyword)
        kw_compact = compact_text(keyword)
        if kw_norm == name_norm or kw_compact == name_compact:
            score += 50
            reasons.append(f"name exactly matches target keyword '{keyword}' +50")
            break
        elif kw_norm in name_norm:
            score += 25
            reasons.append(f"name contains target keyword '{keyword}' +25")
            break

    # Question explicitly mentions the column name
    if name_norm and name_norm in q:
        score += 35
        reasons.append("question explicitly mentions column name +35")
    elif name_compact and name_compact in q_compact:
        score += 35
        reasons.append("question explicitly mentions column (compact match) +35")

    # Question mentions target keyword that maps to this column
    for keyword in TARGET_OUTCOME_KEYWORDS:
        kw_norm = normalize_text(keyword)
        if kw_norm in q and kw_norm in name_norm:
            score += 20
            reasons.append(f"question+column share target keyword '{keyword}' +20")
            break

    # Question has causal/diagnostic language
    has_causal_signal = any(
        normalize_text(signal) in q for signal in TARGET_QUESTION_CAUSAL_SIGNALS
    )

    if has_causal_signal:
        score += 15
        reasons.append("question has causal language +15")

    # Compound rate signal: "cancellation rate" → strong boost for 'is_canceled'
    for compound, hints in COMPOUND_RATE_SIGNALS.items():
        if normalize_text(compound) in q:
            for hint in hints:
                if normalize_text(hint) in name_norm:
                    score += 70
                    reasons.append(f"compound rate signal '{compound}' matches column +70")
                    break

    # Synonym boost: question mentions synonym for this column's outcome role
    for col_pattern, synonyms in COLUMN_KEYWORD_SYNONYMS.items():
        if col_pattern == name_norm or col_pattern in name_norm:
            for syn in synonyms:
                if normalize_text(syn) in q:
                    score += 25
                    reasons.append(f"synonym '{syn}' matches column pattern '{col_pattern}' +25")
                    break

    # Identifiers and datetime cols should never be targets
    if is_identifier_column(col):
        score -= 80
        reasons.append("identifier column -80")

    if col.get("semantic_type") == "datetime":
        score -= 80
        reasons.append("datetime column cannot be target -80")

    if col.get("role") == "time":
        score -= 80
        reasons.append("time role column cannot be target -80")

    # Pure free-text columns are not targets
    if col.get("semantic_type") == "text":
        score -= 60
        reasons.append("free-text column -60")

    return score, reasons


def choose_best_target_column(
    metadata: Dict[str, Any],
    question: str,
    plan: Dict[str, Any],
) -> Optional[str]:
    """
    Select the binary outcome/target column for a question.

    This is used for:
    - groupby_target_rate operations (e.g. 'Does overtime affect attrition?')
    - diagnostic_analysis intent (e.g. 'Why is churn increasing?')
    - Any causal/predictive question referencing a binary outcome

    Returns:
        Column name string, or None if no suitable target found.
    """
    operation = plan.get("operation", "")
    intent = plan.get("intent", "")
    q = normalize_text(question)

    # Only activate for target-rate operations, diagnostic intent, or causal language
    is_target_rate_op = operation in TARGET_RATE_OPERATIONS
    is_causal_intent = intent in CAUSAL_INTENT_TYPES
    has_causal_signal = any(
        normalize_text(signal) in q for signal in TARGET_QUESTION_CAUSAL_SIGNALS
    )
    mentions_target_keyword = any(
        normalize_text(kw) in q for kw in TARGET_OUTCOME_KEYWORDS
    )
    # "cancellation rate", "churn rate", etc. → always activate target selection
    has_rate_compound = any(
        normalize_text(compound) in q for compound in COMPOUND_RATE_SIGNALS
    )

    if not (is_target_rate_op or is_causal_intent or has_rate_compound
            or (has_causal_signal and mentions_target_keyword)):
        return None

    columns = get_columns(metadata)
    scored = []

    for col in columns:
        score, reasons = score_target_candidate(col, question)
        scored.append((score, get_col_name(col), reasons))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_col, _ = scored[0]

    # Require a minimum score — don't guess if nothing looks like a target
    MIN_TARGET_SCORE = 40
    if best_score < MIN_TARGET_SCORE:
        return None

    return best_col


# ============================================================
# Driver Column Selection (for Diagnostic / Causal Analysis)
# ============================================================

def score_driver_candidate(
    col: Dict[str, Any],
    question: str,
    target_column: Optional[str],
    measure_column: Optional[str],
) -> Tuple[int, List[str]]:
    """
    Score a column as a potential driver/explanatory variable.

    Drivers are columns that could explain variance in the target or metric.
    Used for diagnostic_analysis intent.
    """
    score = 0
    reasons = []

    name = get_col_name(col)
    name_norm = normalize_text(name)
    semantic_type = col.get("semantic_type", "")
    role = col.get("role", "")
    cardinality_type = col.get("cardinality_type", "")
    business_type = col.get("business_type", "")

    # Cannot be the target itself
    if target_column and name == target_column:
        return -999, ["is target column, excluded"]

    # Cannot be the main metric either
    if measure_column and name == measure_column:
        return -999, ["is metric column, excluded"]

    # Exclude identifiers and datetime columns
    if is_identifier_column(col):
        return -999, ["identifier column, excluded"]

    if semantic_type == "datetime" or role == "time":
        score -= 50
        reasons.append("datetime column deprioritised -50")

    if semantic_type == "text":
        score -= 40
        reasons.append("free-text column deprioritised -40")

    # Categorical dimensions with manageable cardinality are ideal drivers
    if semantic_type in ["category", "boolean"]:
        score += 30
        reasons.append(f"categorical/boolean driver +30")

    if cardinality_type == "low_cardinality_category":
        score += 25
        reasons.append("low cardinality +25")
    elif cardinality_type == "medium_cardinality_category":
        score += 15
        reasons.append("medium cardinality +15")
    elif cardinality_type in BINARY_CARDINALITY_TYPES:
        score += 20
        reasons.append("binary column +20")

    if business_type in ["geography", "categorical_dimension"]:
        score += 10
        reasons.append(f"business_type={business_type} +10")

    if business_type == "boolean_flag":
        score += 15
        reasons.append("boolean_flag +15")

    # Numeric measures can also be drivers (e.g. age, tenure, income)
    if semantic_type == "numeric" and role == "measure":
        if business_type in ["currency_or_amount", "quantity_or_count", "numeric_measure", "rating_or_score", "percentage"]:
            score += 15
            reasons.append("numeric measure as potential driver +15")

    # Column name appears in the question → likely relevant
    q = normalize_text(question)
    if name_norm and name_norm in q:
        score += 20
        reasons.append("column name appears in question +20")

    # Penalize if the question explicitly asks 'by X' and this isn't X
    # (Don't over-select unrelated columns)
    q_tokens_set = tokens(question)
    col_tokens_set = tokens(name)
    overlap = q_tokens_set & col_tokens_set
    if any(len(t) >= 4 for t in overlap):
        score += 10
        reasons.append("token overlap with question +10")

    return score, reasons


def choose_driver_columns(
    metadata: Dict[str, Any],
    question: str,
    plan: Dict[str, Any],
    target_column: Optional[str],
    measure_column: Optional[str] = None,
    max_drivers: int = 5,
) -> List[str]:
    """
    Select the top driver/explanatory columns for diagnostic analysis.

    For 'Why is churn increasing?' → returns columns like [Region, Segment, Age, Department]
    For simple comparison/ranking → returns [] (drivers not needed)

    Returns:
        List of column name strings (can be empty).
    """
    operation = plan.get("operation", "")
    intent = plan.get("intent", "")

    # Drivers are only needed for target-rate or diagnostic operations
    is_target_rate_op = operation in TARGET_RATE_OPERATIONS
    is_causal = intent in CAUSAL_INTENT_TYPES

    if not (is_target_rate_op or is_causal):
        return []

    # For simple target-rate comparisons (e.g. 'overtime vs attrition'),
    # the driver is essentially the category column — handled by dimension selection.
    # Only return drivers for true diagnostic/causal questions.
    q = normalize_text(question)
    is_complex_causal = any(
        normalize_text(signal) in q
        for signal in ["why", "reason", "cause", "factor", "factors", "explain", "increasing", "decreasing", "rising", "falling"]
    )

    if not is_complex_causal and is_target_rate_op:
        return []

    columns = get_columns(metadata)
    scored = []

    for col in columns:
        score, reasons = score_driver_candidate(col, question, target_column, measure_column)
        if score > 0:
            scored.append((score, get_col_name(col)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [col_name for _, col_name in scored[:max_drivers]]


# ============================================================
# Groupby Column Determination
# ============================================================

def determine_groupby_columns(
    category_column: Optional[str],
    driver_columns: List[str],
    plan: Dict[str, Any],
    question: str,
) -> List[str]:
    """
    Determine which column(s) to group by for the operation.

    For ranking/comparison: [category_column]
    For target-rate: [category_column]
    For diagnostic: [] (drivers are used separately by AnalysisAgent)
    For trend: [] (time column is used instead)
    """
    operation = plan.get("operation", "")

    if operation in TIME_OPERATIONS:
        # Time-series: groupby is handled by the time column, not dimensions
        return []

    if operation in CORRELATION_OPERATIONS:
        return []

    if operation in TEXT_ONLY_OPERATIONS:
        return []

    if operation in [
        "groupby_sum",
        "groupby_sum_sort_desc",
        "groupby_mean",
        "groupby_mean_sort_desc",
        "groupby_target_rate",
        "groupby_target_rate_sort_desc",
    ]:
        if category_column:
            return [category_column]

    if operation in ["full_dataset_analysis", "diagnostic_analysis"]:
        # Diagnostic analysis uses driver columns, not a single groupby
        return []

    if category_column:
        return [category_column]

    return []


# ============================================================
# Overall Mapping Confidence
# ============================================================

def compute_overall_confidence(
    confidences: Dict[str, float],
    plan: Dict[str, Any],
    mapping: Dict[str, Any],
) -> float:
    """
    Compute a single overall confidence score for the full column mapping.

    Weights essential columns more heavily based on what the operation needs.
    Penalises if required columns are missing.
    """
    operation = plan.get("operation", "")
    required_roles = plan.get("required_data_roles", {})

    weights = {}
    penalties = 0.0

    if operation in SCALAR_OPERATIONS:
        weights["measure"] = 1.0
    elif operation in TIME_OPERATIONS:
        weights["measure"] = 0.6
        weights["time"] = 0.4
    elif operation in CORRELATION_OPERATIONS:
        weights["measure"] = 1.0
    elif operation in TARGET_RATE_OPERATIONS:
        weights["dimension"] = 0.7
        # target column confidence proxied via dimension
        if not mapping.get("target_column"):
            penalties += 0.3
    elif operation in [
        "groupby_sum", "groupby_sum_sort_desc",
        "groupby_mean", "groupby_mean_sort_desc",
    ]:
        weights["measure"] = 0.5
        weights["dimension"] = 0.5
    elif operation in TEXT_ONLY_OPERATIONS:
        weights["text"] = 1.0
    else:
        # General fallback weighting
        if required_roles.get("needs_numeric"):
            weights["measure"] = 0.5
        if required_roles.get("needs_category"):
            weights["dimension"] = 0.3
        if required_roles.get("needs_datetime"):
            weights["time"] = 0.2

    if not weights:
        weights = {"measure": 1.0}

    total_weight = sum(weights.values())
    weighted_score = 0.0

    for role, weight in weights.items():
        conf = confidences.get(role, 0.0)
        weighted_score += conf * (weight / total_weight)

    # Check required columns are present
    if operation in SCALAR_OPERATIONS and not mapping.get("metric_column"):
        penalties += 0.4
    if operation in ["groupby_sum", "groupby_sum_sort_desc", "groupby_mean", "groupby_mean_sort_desc"]:
        if not mapping.get("metric_column"):
            penalties += 0.25
        if not mapping.get("category_column"):
            penalties += 0.25
    if operation in TIME_OPERATIONS and not mapping.get("date_column"):
        penalties += 0.35

    result = max(0.0, weighted_score - penalties)
    return round(result, 3)


# ============================================================
# New Clean Public API: map_columns()
# ============================================================


# ============================================================
# Internal Helpers for map_columns()
# ============================================================

def _build_minimal_metadata(df):
    """
    Build a minimal metadata dict from df dtypes.
    Used as fallback when metadata_profiler is unavailable.
    """
    import pandas as pd
    columns = []
    for col_name in df.columns:
        series = df[col_name]
        dtype_str = str(series.dtype)
        if pd.api.types.is_datetime64_any_dtype(series):
            semantic_type = "datetime"
            role = "time"
            business_type = "date_or_time"
            cardinality_type = "time"
        elif pd.api.types.is_numeric_dtype(series):
            semantic_type = "numeric"
            role = "measure"
            business_type = "numeric_measure"
            n_unique = series.nunique()
            n_rows = len(series)
            ratio = n_unique / n_rows if n_rows > 0 else 0
            cardinality_type = "high_unique_numeric" if ratio > 0.9 else "continuous_or_measure"
        else:
            semantic_type = "category"
            role = "dimension"
            business_type = "categorical_dimension"
            n_unique = series.nunique()
            n_rows = len(series)
            ratio = n_unique / n_rows if n_rows > 0 else 0
            if n_unique <= 2:
                cardinality_type = "binary_category"
            elif n_unique <= 20:
                cardinality_type = "low_cardinality_category"
            elif ratio >= 0.8:
                cardinality_type = "high_cardinality_category"
            else:
                cardinality_type = "medium_cardinality_category"
        columns.append({
            "name": str(col_name),
            "dtype": dtype_str,
            "semantic_type": semantic_type,
            "role": role,
            "business_type": business_type,
            "cardinality_type": cardinality_type,
            "null_count": int(series.isna().sum()),
            "null_percent": round(float(series.isna().mean() * 100), 2),
            "unique_count": int(series.nunique()),
            "unique_ratio": round(float(series.nunique() / len(series)), 4) if len(series) > 0 else 0.0,
            "sample_values": series.dropna().astype(str).head(5).tolist(),
        })
    return {
        "source_type": "unknown",
        "row_count": len(df),
        "column_count": len(df.columns),
        "has_numeric": any(c["semantic_type"] == "numeric" for c in columns),
        "has_category": any(c["semantic_type"] == "category" for c in columns),
        "has_datetime": any(c["semantic_type"] == "datetime" for c in columns),
        "has_text": False,
        "columns": columns,
    }


def _metadata_excluding_column(metadata, exclude_col):
    """Return a shallow copy of metadata with one column excluded."""
    import copy
    alt = copy.deepcopy(metadata)
    alt["columns"] = [c for c in alt["columns"] if c.get("name") != exclude_col]
    return alt


def _extract_by_clause_category(question: str, df) -> Optional[str]:
    """
    Parse "by X", "per X", "in each X", "for each X", "across Xs" from the
    question and find the best matching column in df.

    Used as a fallback when the ML planner predicts a scalar operation but the
    question clearly asks for a grouped breakdown.

    Returns the matched column name, or None.
    """
    q_norm  = normalize_text(question)
    q_depl  = normalize_for_matching(question)

    # Regex patterns: capture the phrase after the preposition
    _BY_PATTERNS = [
        r'\bby\s+([\w][\w ]{1,40}?)(?:\s*[?.]|$|\s+(?:and|or|with|where|when)\b)',
        r'\bper\s+([\w][\w ]{1,30}?)(?:\s*[?.]|$|\s+(?:and|or|with)\b)',
        r'\bin each\s+([\w][\w ]{1,30}?)(?:\s*[?.]|$|\s+(?:and|or|with)\b)',
        r'\bfor each\s+([\w][\w ]{1,30}?)(?:\s*[?.]|$|\s+(?:and|or|with)\b)',
        r'\bacross\s+([\w][\w ]{1,30}?)(?:\s*[?.]|$|\s+(?:and|or|with)\b)',
        r'\bbreakdown by\s+([\w][\w ]{1,30}?)(?:\s*[?.]|$)',
        r'\bsplit by\s+([\w][\w ]{1,30}?)(?:\s*[?.]|$)',
        r'\bgroup(?:ed)? by\s+([\w][\w ]{1,30}?)(?:\s*[?.]|$)',
    ]

    extracted_terms = []
    for pattern in _BY_PATTERNS:
        for match in re.finditer(pattern, q_depl):
            term = match.group(1).strip()
            # Skip very generic stop-words
            if term not in {"the", "a", "an", "its", "their", "all", "each", "every"}:
                extracted_terms.append(term)
        for match in re.finditer(pattern, q_norm):
            term = match.group(1).strip()
            if term not in {"the", "a", "an", "its", "their", "all", "each", "every"}:
                if term not in extracted_terms:
                    extracted_terms.append(term)

    if not extracted_terms:
        return None

    cols = list(df.columns)
    best_col: Optional[str] = None
    best_score = -1

    for term in extracted_terms:
        term_norm  = normalize_text(term)
        term_depl  = normalize_for_matching(term)
        term_compact = term_norm.replace(" ", "")

        for col in cols:
            col_norm    = normalize_text(col)
            col_compact = col_norm.replace(" ", "")

            sc = 0
            # Exact match (highest priority)
            if col_norm == term_norm or col_norm == term_depl:
                sc = 100
            # Column name contained in extracted term (e.g. "education" in "education level")
            elif col_norm and col_norm in term_norm:
                sc = 70
            # Extracted term contained in column name
            elif term_norm and term_norm in col_norm:
                sc = 65
            # Compact match (handles camelCase like JobRole vs "job role")
            elif col_compact == term_compact:
                sc = 80
            elif term_compact and term_compact in col_compact:
                sc = 55
            else:
                # Token overlap
                col_toks  = set(col_norm.split())
                term_toks = set(term_norm.split()) | set(term_depl.split())
                overlap   = col_toks & term_toks
                meaningful = {t for t in overlap if len(t) >= 3}
                if meaningful:
                    sc = 30 * len(meaningful)

            if sc > best_score:
                best_score = sc
                best_col   = col

    # Require a reasonable match
    if best_score >= 30 and best_col is not None:
        return best_col
    return None


# ============================================================
# New Clean Public API: map_columns()
# ============================================================


# ============================================================
# New Clean Public API: map_columns()
# ============================================================


# ============================================================
# New Clean Public API: map_columns()
# ============================================================

def map_columns(question, df, plan, metadata=None):
    """
    Map actual dataset columns to the roles required by the analysis plan.
    All returned column names are guaranteed to exist in df.columns.
    """
    # 0. Build metadata
    if metadata is None:
        try:
            try:
                from metadata_profiler import profile_dataframe
            except ImportError:
                sys.path.insert(0, os.path.dirname(__file__))
                from metadata_profiler import profile_dataframe
            metadata = profile_dataframe(df)
        except Exception:
            metadata = _build_minimal_metadata(df)

    operation      = plan.get("operation", "")
    required_roles = plan.get("required_data_roles", {})
    needs_numeric  = required_roles.get("needs_numeric", False)
    needs_category = required_roles.get("needs_category", False)
    needs_datetime = required_roles.get("needs_datetime", False)
    needs_text     = required_roles.get("needs_text", False)

    all_warnings    = []
    raw_confidences = {}

    # 1. Metric column
    metric_column     = None
    should_map_measure = (
        operation not in TEXT_ONLY_OPERATIONS
        and operation not in TARGET_RATE_OPERATIONS
        and (
            needs_numeric
            or operation in [
                "sum", "mean", "max", "min",
                "groupby_sum", "groupby_sum_sort_desc",
                "groupby_mean", "groupby_mean_sort_desc",
                "time_groupby_sum", "correlation", "correlation_heatmap",
                "distribution", "outlier_check", "forecast",
                "diagnostic_analysis", "full_dataset_analysis",
                "count_rows",
            ]
        )
    )
    if should_map_measure:
        selected, conf, _, warns = choose_best_candidate(
            metadata, question, "measure", operation=operation
        )
        metric_column = selected
        raw_confidences["measure"] = conf
        all_warnings.extend(warns)

    # 2. Category column
    category_column     = None
    should_map_dimension = (
        operation not in TEXT_ONLY_OPERATIONS
        and operation not in SCALAR_OPERATIONS
        and operation not in TIME_OPERATIONS
        and operation not in CORRELATION_OPERATIONS
        and (
            needs_category
            or operation in DIMENSION_REQUIRED_OPERATIONS
            or operation in TARGET_RATE_OPERATIONS
        )
    )
    if operation in TEXT_ONLY_OPERATIONS and question_explicitly_requests_text_grouping(question):
        should_map_dimension = True
    if should_map_dimension:
        selected, conf, _, warns = choose_best_candidate(
            metadata, question, "dimension", operation=operation
        )
        category_column = selected
        raw_confidences["dimension"] = conf
        all_warnings.extend(warns)

    # 2b. By-clause fallback: "by X" / "in each X" / "per X" / "across Xs"
    if category_column is None:
        by_col = _extract_by_clause_category(question, df)
        if by_col:
            category_column = by_col
            if "dimension" not in raw_confidences:
                raw_confidences["dimension"] = 0.75

    # 3. Date column
    date_column     = None
    should_map_time  = (
        needs_datetime
        or operation in TIME_OPERATIONS
        or operation in ["diagnostic_analysis", "full_dataset_analysis"]
    )
    if should_map_time:
        selected, conf, _, warns = choose_best_candidate(
            metadata, question, "time", operation=operation
        )
        date_column = selected
        raw_confidences["time"] = conf
        all_warnings.extend(warns)

    # 4. Text column
    text_column = None
    if needs_text or operation in TEXT_ONLY_OPERATIONS:
        selected, conf, _, warns = choose_best_candidate(
            metadata, question, "text", operation=operation
        )
        text_column = selected
        raw_confidences["text"] = conf
        all_warnings.extend(warns)

    # 5. Target column (binary outcome)
    target_column = choose_best_target_column(metadata, question, plan)

    # Avoid target == category conflict
    if target_column and category_column and target_column == category_column:
        alt_meta = _metadata_excluding_column(metadata, target_column)
        alt_col, alt_conf, _, alt_warns = choose_best_candidate(
            alt_meta, question, "dimension", operation=operation
        )
        category_column = alt_col
        raw_confidences["dimension"] = alt_conf
        all_warnings.extend(alt_warns)

    # 5b. Post-target fallback: if target selected but category still None,
    # try dimension selection unconditionally (handles "do X cancel more than Y?" patterns).
    # Also scans binary/boolean columns for question-token overlap (e.g. "repeated guests"
    # matching is_repeated_guest).
    if target_column and category_column is None:
        excl_meta = _metadata_excluding_column(metadata, target_column)
        fb_col, fb_conf, _, fb_warns = choose_best_candidate(
            excl_meta, question, "dimension", operation="groupby_target_rate"
        )
        if fb_col:
            category_column = fb_col
            raw_confidences["dimension"] = fb_conf
            all_warnings.extend(fb_warns)

        # Extra scan: binary columns mentioned in question beat generic dimension winner
        if category_column is None or True:
            q_toks = set(normalize_for_matching(question).split())
            best_bin_col, best_bin_sc = None, -1
            for col_meta in get_columns(excl_meta):
                cname = get_col_name(col_meta)
                if cname == target_column:
                    continue
                is_bin = is_binary_or_boolean_column(col_meta)
                cname_norm = normalize_text(cname)
                c_toks = set(cname_norm.split())
                overlap = {t for t in (c_toks & q_toks) if len(t) >= 4}
                if overlap and is_bin:
                    sc = len(overlap) * 30
                    if sc > best_bin_sc:
                        best_bin_sc = sc
                        best_bin_col = cname
            if best_bin_col and best_bin_sc > 20:
                # Only override if the binary column scores better than current pick
                current_score = score_column_for_question(
                    category_column or "", question, DIMENSION_KEYWORDS
                ) if category_column else 0
                if best_bin_sc >= current_score:
                    category_column = best_bin_col
                    raw_confidences["dimension"] = 0.70

    # 6. Driver columns
    driver_columns = choose_driver_columns(
        metadata, question, plan, target_column, metric_column
    )
    if not driver_columns and operation in TARGET_RATE_OPERATIONS and category_column:
        driver_columns = [category_column]

    # 7. Groupby columns
    groupby_columns = determine_groupby_columns(
        category_column, driver_columns, plan, question
    )

    # 8. Time grain
    time_grain = infer_time_grain(question)

    # 9. Overall confidence
    partial_mapping = {
        "target_column":   target_column,
        "metric_column":   metric_column,
        "category_column": category_column,
        "date_column":     date_column,
        "text_column":     text_column,
        "driver_columns":  driver_columns,
        "groupby_columns": groupby_columns,
    }
    confidence = compute_overall_confidence(raw_confidences, plan, partial_mapping)

    # 10. Safety check — validate all columns exist in df.columns
    existing = set(df.columns.tolist())

    def _safe(col):
        if col is None:
            return None
        if col in existing:
            return col
        col_lower = col.lower()
        for c in existing:
            if c.lower() == col_lower:
                all_warnings.append(
                    f"Column '{col}' matched case-insensitively to '{c}'"
                )
                return c
        all_warnings.append(f"Column '{col}' not found in dataset - removed.")
        return None

    target_column   = _safe(target_column)
    metric_column   = _safe(metric_column)
    category_column = _safe(category_column)
    date_column     = _safe(date_column)
    text_column     = _safe(text_column)
    driver_columns  = [c for c in driver_columns  if c in existing]
    groupby_columns = [c for c in groupby_columns if c in existing]

    seen_w  = set()
    deduped = []
    for w in all_warnings:
        if w not in seen_w:
            seen_w.add(w)
            deduped.append(w)

    return {
        "target_column":   target_column,
        "metric_column":   metric_column,
        "category_column": category_column,
        "date_column":     date_column,
        "text_column":     text_column,
        "driver_columns":  driver_columns,
        "groupby_columns": groupby_columns,
        "time_grain":      time_grain,
        "confidence":      confidence,
        "warnings":        deduped,
    }
