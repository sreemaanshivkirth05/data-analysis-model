import json
import os
from typing import Dict, Any, Optional

from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_router_prompt(question: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Build the prompt for the LLM router.

    The router decides whether the user question should go to:
    1. Dataset analysis pipeline
    2. Normal conversation/help response

    It also normalizes messy or misspelled analysis questions before sending them
    to the ML planner.
    """

    metadata_summary = "No dataset metadata available."

    if metadata:
        metadata_summary = json.dumps(
            {
                "source_type": metadata.get("source_type"),
                "row_count": metadata.get("row_count"),
                "column_count": metadata.get("column_count"),
                "has_numeric": metadata.get("has_numeric"),
                "has_category": metadata.get("has_category"),
                "has_datetime": metadata.get("has_datetime"),
                "has_text": metadata.get("has_text"),
                "columns": [
                    {
                        "name": col.get("name"),
                        "semantic_type": col.get("semantic_type"),
                        "role": col.get("role"),
                    }
                    for col in metadata.get("columns", [])[:20]
                ],
            },
            indent=2,
        )

    return f"""
You are a routing classifier for an AI data analysis application.

Your job:
Decide whether the user question should be routed to the dataset analysis planner or handled as normal conversation.

Route to "analysis" only if the user is asking something that requires inspecting, summarizing, calculating from, visualizing, explaining, or analyzing the uploaded dataset.

Route to "conversation" if the user is:
- greeting
- asking about the app
- asking general questions
- asking your name
- asking for help unrelated to the dataset
- giving vague text that cannot be answered from the dataset
- asking something with no clear connection to the dataset

Dataset metadata:
{metadata_summary}

User question:
{question}

Return only valid JSON with this exact schema:
{{
  "route": "analysis" or "conversation",
  "is_analysis_request": true or false,
  "confidence": number between 0 and 1,
  "reason": "short reason",
  "normalized_question": "corrected and cleaned version of the user question"
}}

Rules for normalized_question:
- Fix spelling mistakes.
- Keep the original meaning.
- If the user asks for full dataset analysis, normalize to: "Analyze this dataset and give business insights"
- If the user asks for a ranking question, normalize clearly. Example: "Which product has the highest sales?"
- If the user asks for a trend question, normalize clearly. Example: "Show monthly revenue trend"
- If the user asks for a comparison question, normalize clearly. Example: "Compare sales by country"
- If the user asks for data quality, normalize clearly. Example: "Are there missing values?"
- If the user asks small talk or conversation, keep a cleaned version of the same message.
- Do not add extra analysis details that the user did not ask for.
- Do not include markdown.
- Do not include explanation outside JSON.
"""


def safe_json_parse(content: str, original_question: str) -> Dict[str, Any]:
    """
    Safely parse the LLM response into JSON.
    If the LLM returns invalid JSON, return a safe conversation route.
    """

    try:
        result = json.loads(content)

        # Ensure required keys exist
        result.setdefault("route", "conversation")
        result.setdefault("is_analysis_request", False)
        result.setdefault("confidence", 0.0)
        result.setdefault("reason", "No reason provided.")
        result.setdefault("normalized_question", original_question)

        return result

    except json.JSONDecodeError:
        return {
            "route": "conversation",
            "is_analysis_request": False,
            "confidence": 0.0,
            "reason": "Router returned invalid JSON.",
            "normalized_question": original_question,
        }


def route_question(question: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Route the user question.

    Returns:
    {
      "route": "analysis" or "conversation",
      "is_analysis_request": true/false,
      "confidence": float,
      "reason": "...",
      "normalized_question": "..."
    }
    """

    if not question or not question.strip():
        return {
            "route": "conversation",
            "is_analysis_request": False,
            "confidence": 1.0,
            "reason": "Empty question.",
            "normalized_question": "",
        }

    prompt = build_router_prompt(question, metadata)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON router. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = response.choices[0].message.content.strip()
    result = safe_json_parse(content, question)

    return result


if __name__ == "__main__":
    test_metadata = {
        "source_type": "csv",
        "row_count": 10,
        "column_count": 6,
        "has_numeric": True,
        "has_category": True,
        "has_datetime": True,
        "has_text": False,
        "columns": [
            {"name": "Date", "semantic_type": "datetime", "role": "time"},
            {"name": "Country", "semantic_type": "category", "role": "dimension"},
            {"name": "Product", "semantic_type": "category", "role": "dimension"},
            {"name": "Sales", "semantic_type": "numeric", "role": "measure"},
            {"name": "Profit", "semantic_type": "numeric", "role": "measure"},
            {"name": "Customer", "semantic_type": "category", "role": "dimension"},
        ],
    }

    questions = [
        "hi",
        "what is your name",
        "which produt has the highest sales?",
        "can you explain this?",
        "show monthly revenue trend",
        "what should I do next?",
        "are there missing values?",
        "how does this app work?",
        "do analysisss for thiss datasetssss",
    ]

    for question in questions:
        print("\nQuestion:", question)
        print(json.dumps(route_question(question, test_metadata), indent=2))