import json

from metadata_profiler import profile_file
from predict import predict_plan


def main():
    print("\n================ UNIVERSAL ANALYSIS PLANNER TEST ================\n")

    dataset_path = input("Enter dataset path: ").strip()

    metadata = profile_file(dataset_path)

    print("\nDataset Metadata Summary:")
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

    while True:
        question = input("\nAsk a question about the dataset, or type 'exit': ").strip()

        if question.lower() == "exit":
            print("\nExiting test.")
            break

        plan = predict_plan(question, metadata)

        print("\nPredicted Analysis Plan:")
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()