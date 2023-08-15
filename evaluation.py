from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report


def get_report(data: pd.DataFrame, column: str):
    data = data.loc[data['train_test_set'] == "test"]
    test_report = classification_report(data[column], data[f"pred_{column}"],output_dict=True)

    return test_report


def save_report(test_report):
    df = pd.DataFrame(test_report).transpose()
    df = df.assign(models = model, tasks = task, columns = column)

    return df

def main():
    files = glob("predictions/*_pred*.csv")
    files = [file for file in files if "only-political" not in file]

    outputs = pd.DataFrame()

    for file in files:
        data = pd.read_csv(file)
        task, _, column, model = str(Path(file).stem).split("_")
        print(f"Model: {model}. Task: {task}. Outcome Variable: {column}")

        test_report = get_report(data, column)
        
        df = save_report

        outputs = pd.concat([outputs, df])

    outputs.to_csv('tweet_classification_outputs.csv')


if __name__ == "__main__":
    main()