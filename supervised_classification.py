from pathlib import Path

import numpy as np
import pandas as pd
from embetter.text import GensimEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

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


def train_test_indices(
    data: pd.DataFrame,
    seed: int,
    train_size: float = 0.8,
):
    index = np.copy(data.index)
    np.random.default_rng(seed).shuffle(index)
    split_at = int(train_size * len(index))
    train_index, test_index = index[:split_at], index[split_at:]
    return train_index, test_index


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
    train_idx, test_idx = train_test_indices(data, seed=seed, train_size=0.8)
    data["train_test_set"] = "test"
    data["train_test_set"][train_idx] = "train"
    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)

    for outcome in ["political", "exemplar"]:
        for model in pipelines.keys():
            print(f"Supervised classification with {model} over {outcome}")
            classifier = pipelines[model]()
            X_train = data["raw_text"].loc[train_idx]
            y_train = data[outcome].loc[train_idx]
            classifier.fit(X_train, y_train)
            pred_data = data.copy()
            pred_data[f"pred_{outcome}"] = classifier.predict(data["raw_text"])
            out_path = out_dir.joinpath(
                f"supervised_pred_{outcome}_{model}.csv"
            )
            pred_data.to_csv(out_path)
    print("DUN")


if __name__ == "__main__":
    main(seed=0)
