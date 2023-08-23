import argparse 
from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from contextlib import redirect_stdout

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LLM Classification Evaluation")
    parser.add_argument("data_folder", type=str, default="predictions/")
    parser.add_argument("--output_folder", type=str, default="output/")
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
    in_dir = Path(args.data_folder)
    out_dir = Path(args.output_folder)

    # Collecting prediction files from given directory
    files = glob(str(in_dir.joinpath("*_pred*.csv")))
    files = [file for file in files if "only-political" not in file]

    # Creating output directory if it does not exist.
    out_dir.mkdir(exist_ok=True)
    # Getting last part of the path to use as name of the file.
    output_name = Path(in_dir).name

    outputs = pd.DataFrame()
    with open(out_dir.joinpath(f"{output_name}_reports.txt"), "w") as buffer:
        with redirect_stdout(buffer):

            for file in files:
                data = pd.read_csv(file)
                task, _, column, model = str(Path(file).stem).split("_")
                print(f"Model: {model}. Task: {task}. Outcome Variable: {column}")

                print_report(data, column)
                df = produce_report(data, column)
                test_report = df.assign(models=model, tasks=task, columns=column)

                outputs = pd.concat([outputs, test_report])


    out_file = out_dir.joinpath(f"{output_name}_outputs.csv")
    outputs.to_csv(out_file)


if __name__ == "__main__":
    main()
