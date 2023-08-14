from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

files = glob("./*_pred*.csv")
files = [file for file in files "only-political" not in file]

for file in files:
    data = pd.read_csv(file)
    task, _, column, model = str(Path(file).stem).split("_")
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"Outcome Variable: {column}")
    all_report = classification_report(data[column], data[f"pred_{column}"])
    data = data[data.train_test_set == "test"]
    test_report = classification_report(data[column], data[f"pred_{column}"])
    report_width = len(all_report.split("\n")[0])
    print("".center(report_width, "-"))
    print("All entries".center(report_width, " "))
    print("".center(report_width, "-"))
    print(all_report)
    print("".center(report_width, "-"))
    print("Test set only".center(report_width, " "))
    print("".center(report_width, "-"))
    print(test_report)
    print("\n")
