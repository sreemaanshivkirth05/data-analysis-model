import os
import subprocess
import sys
import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


COMMANDS = [
    ["python", "src/generate_planner_stress_cases.py"],
    ["python", "src/evaluate_planner_stress.py"],
    ["python", "src/generate_execution_eval_cases.py"],
    ["python", "src/evaluate_execution_plan.py"],
]


def run_command(command):
    print("\n============================================================")
    print("Running:", " ".join(command))
    print("============================================================\n")

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def main():
    start = datetime.datetime.now()

    print("\n================ UNIVERSAL PLANNER STRESS TEST RUNNER ================\n")
    print("Started at:", start)

    for command in COMMANDS:
        run_command(command)

    end = datetime.datetime.now()

    print("\n================ STRESS TESTS COMPLETE ================\n")
    print("Started at:", start)
    print("Finished at:", end)
    print("Duration:", end - start)


if __name__ == "__main__":
    main()