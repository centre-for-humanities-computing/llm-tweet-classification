from pathlib import Path

import numpy as np
import pandas as pd
from embetter.text import GensimEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
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
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    scoring = ["accuracy", f1_score, recall_score, precision_score]

    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)

    for outcome in ["political", "exemplar"]:
        for model in pipelines.keys():
            print(f"Supervised classification with {model} over {outcome}")
            classifier = pipelines[model]()

            X = data["raw_text"]
            y = data[outcome]

            print("initializing scorer")

            all_scores = pd.DataFrame()
            for score in scoring:
                print(f"doing the cv for score: {score}")

                if score == "accuracy":
                    cv_scores = cross_validate(classifier, X, y, scoring=score, cv=cv)
                else:
                    custom_scorer = make_scorer(score, pos_label=outcome)
                    cv_scores = cross_validate(classifier, X, y, scoring=custom_scorer, cv=cv)

                all_scores[str(score)] = cv_scores['test_score']

            out_path = out_dir.joinpath(f"cv_scores_{outcome}_{model}.csv")

            print("saving the output")
            all_scores.to_csv(out_path)

    print("DUN")


if __name__ == "__main__":
    main(seed=0)
