import json
from pathlib import Path

import numpy as np
import pandas as pd
from embetter.text import GensimEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.finetune import BertFinetuneClassifier


def glove_pipeline(seed: int) -> Pipeline:
    regularization_grid = list(np.logspace(-3, 3, 7))
    cls_pipe = make_pipeline(
        GensimEncoder(
            "glove-twitter-200", agg="mean", deacc=True, lowercase=True
        ),
        SimpleImputer(),
        StandardScaler(),
        LogisticRegressionCV(Cs=regularization_grid, random_state=seed),
    )
    return cls_pipe


def main(seed: int = 0):
    print("Building pipelines")
    pipelines = {
        "glove-twitter-200": lambda: glove_pipeline(seed=seed),
        "distilbert-base-uncased": lambda: BertFinetuneClassifier(
            "distilbert-base-uncased", device="cpu"
        ),
    }

    print("Loading data.")
    data = pd.read_csv("labelled_data.csv")
    n_folds = 5
    cross_validator = StratifiedKFold(
        n_splits=n_folds, random_state=seed, shuffle=True
    )

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    records = []
    for outcome in ["political", "exemplar"]:
        for model in pipelines.keys():
            print(f"Supervised classification with {model} over {outcome}")
            classifier = pipelines[model]()
            # Turning into numpy arrays so nothing strange happens with
            # the indices
            X = np.array(data["raw_text"])
            y = np.array(data[outcome])
            # Initiating cross validation folds
            folds = cross_validator.split(X, y)
            folds = tqdm(folds, total=n_folds, desc="Cross validating...")
            for i_fold, (train_index, test_index) in enumerate(folds):
                classifier.fit(X[train_index], y[train_index])
                y_pred = classifier.predict(X[test_index])
                report = classification_report(
                    y[test_index], y_pred, output_dict=True
                )
                # The positive label is the same as the column name.
                # But THIS MIGHT CHANGE so beware that then
                # we have to relax the assumption in the next line.
                pos_label = outcome
                record = {
                    "model": model,
                    "outcome": outcome,
                    "fold": i_fold,
                    "accuracy": report["accuracy"],
                    **report[pos_label],
                }
                records.append(record)
    print("Saving evaluation results")
    pd.DataFrame.from_records(records).to_csv(
        out_dir.joinpath("cv_scores_supervised.csv")
    )
    print("DUN")


if __name__ == "__main__":
    main(seed=0)
