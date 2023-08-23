import argparse 
from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from contextlib import redirect_stdout

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LLM Classification Evaluation")
    parser.add_argument("data_folder", type=str)

    return parser

def produce_report(data: pd.DataFrame, column: str) -> pd.DataFrame:
    data = data.loc[data["train_test_set"] == "test"]
    test_report = classification_report(
        data[column], data[f"pred_{column}"], output_dict=True
    )

    df = pd.DataFrame(test_report).T

    return df


def print_report(data: pd.DataFrame, column: str) -> None:
    data = data.loc[data["train_test_set"] == "test"]
    report = classification_report(data[column], data[f"pred_{column}"])

    report_width = len(report.split("\n")[0])
    print("".center(report_width, "-"))
    print(report, "\n")


def main():
    parser = create_parser()
    args = parser.parse_args()

    files = glob(f"{args.data_folder}/*_pred*.csv")
    files = [file for file in files if "only-political" not in file]

    outputs = pd.DataFrame()

    Path("output").mkdir(exist_ok=True)
    with open(f"output/{args.data_folder}_reports.txt", "w") as f:
        with redirect_stdout(f):

            for file in files:
                data = pd.read_csv(file)
                task, _, column, model = str(Path(file).stem).split("_")
                print(f"Model: {model}. Task: {task}. Outcome Variable: {column}")

                print_report(data, column)
                df = produce_report(data, column)
                test_report = df.assign(models=model, tasks=task, columns=column)

                outputs = pd.concat([outputs, test_report])

    outputs.to_csv(f"output/{args.data_folder}_outputs.csv")


if __name__ == "__main__":
    main()
