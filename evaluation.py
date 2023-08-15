from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report


def produce_report(data: pd.DataFrame, column: str) -> pd.DataFrame:
    data = data.loc[data["train_test_set"] == "test"]
    test_report = classification_report(
        data[column], data[f"pred_{column}"], output_dict=True
    )

    df = pd.DataFrame(test_report).T

    return df


def main():
    files = glob("predictions/*_pred*.csv")
    files = [file for file in files if "only-political" not in file]

    outputs = pd.DataFrame()

    for file in files:
        data = pd.read_csv(file)
        task, _, column, model = str(Path(file).stem).split("_")
        print(f"Model: {model}. Task: {task}. Outcome Variable: {column}")

        test_report = produce_report(data, column)
        test_report = df.assign(models=model, tasks=task, columns=column)

        outputs = pd.concat([outputs, test_report])

    outputs.to_csv("tweet_classification_outputs.csv")


if __name__ == "__main__":
    main()
