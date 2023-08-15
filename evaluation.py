from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report


def print_report(data: pd.DataFrame, column: str) -> None:
    data = data.loc[data['train_test_set'] == "test"]
    test_report = classification_report(data[column], data[f"pred_{column}"])
    report_width = len(test_report.split("\n")[0])
    
    print("".center(report_width, "-"))
    print("Test set only".center(report_width, " "))
    print("".center(report_width, "-"))
    print(test_report)
    print("\n")

def main():
    files = glob("predictions/*_pred*.csv")
    files = [file for file in files if "only-political" not in file]

    for file in files:
        data = pd.read_csv(file)
        task, _, column, model = str(Path(file).stem).split("_")
        print(f"Model: {model}. Task: {task}. Outcome Variable: {column}")

        print_report(data, column)


if __name__ == "__main__":
    main()