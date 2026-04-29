import json

from metadata_profiler import profile_file
from llm_router import route_question
from predict import predict_plan


def print_metadata_summary(metadata):
    """
    Print a short metadata summary after dataset loading.
    """

    print("\nDataset loaded.")
    print(json.dumps(
        {
            "source_type": metadata["source_type"],
            "row_count": metadata["row_count"],
            "column_count": metadata["column_count"],
            "has_numeric": metadata["has_numeric"],
            "has_category": metadata["has_category"],
            "has_datetime": metadata["has_datetime"],
            "has_text": metadata["has_text"],
        },
        indent=2
    ))


def main():
    print("\n================ ROUTER + ML PLANNER TEST ================\n")

    dataset_path = input("Enter dataset path: ").strip()
    metadata = profile_file(dataset_path)

    print_metadata_summary(metadata)

    while True:
        question = input("\nAsk a question, or type 'exit': ").strip()

        if question.lower() == "exit":
            print("\nExiting test.")
            break

        route = route_question(question, metadata)

        print("\nRouter Decision:")
        print(json.dumps(route, indent=2))

        if route.get("is_analysis_request") is True:
            normalized_question = route.get("normalized_question", question)

            print("\nNormalized Question:")
            print(normalized_question)

            plan = predict_plan(normalized_question, metadata)

            print("\nML Planner Output:")
            print(json.dumps(plan, indent=2))

        else:
            print("\nConversation Response:")
            print(
                "This does not look like a dataset-analysis request. "
                "Try asking something like: 'Which product has the highest sales?'"
            )


if __name__ == "__main__":
    main()