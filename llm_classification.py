"""CLI for running zero and few-shot classification on a tweet dataset with
large language models and transformers."""
import argparse
import os
from pathlib import Path

import pandas as pd
from confection import Config
from sklearn.base import ClassifierMixin
from skllm import FewShotGPTClassifier, ZeroShotGPTClassifier
from skllm.config import SKLLMConfig
from stormtrooper import (
    GenerativeFewShotClassifier,
    GenerativeZeroShotClassifier,
    Text2TextFewShotClassifier,
    Text2TextZeroShotClassifier,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LLM Classifier")
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--task", default="zero-shot")
    parser.add_argument("--outcome_column", default="political")
    parser.add_argument("--in_path", default="labelled_data.csv")
    parser.add_argument("--out_dir", default="predictions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=None)
    return parser


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    # I'm resetting index so it will just be 0-N
    data = data.reset_index(drop=True)
    data.political = data.political.map({0: "apolitical", 1: "political"})
    data.exemplar = data.exemplar.map({0: "not an exemplar", 1: "exemplar"})
    return data


def prepare_model(model: str, task: str, device: str) -> ClassifierMixin:
    """Loads classifier model based on model name and task."""
    if model in {"gpt-3.5-turbo", "gpt-4"}:
        print("Initializing connection to OpenAI")
        try:
            openai_key = os.environ["OPENAI_KEY"]
            openai_org = os.environ["OPENAI_ORG"]
        except KeyError:
            raise KeyError(
                "Environment variables OPENAI_KEY and OPENAI_ORG not specified."
            )
        SKLLMConfig.set_openai_key(openai_key)
        SKLLMConfig.set_openai_org(openai_org)
        if task == "zero-shot":
            return ZeroShotGPTClassifier(model)
        else:
            return FewShotGPTClassifier(model)
    # This might not be a good requirement, so if something breaks in
    # the future, do change it.
    # Not all seq2seq models have T5 in their name just the
    # ones we're testing for this project.
    elif "t5" in model.lower():
        if task == "zero-shot":
            return Text2TextZeroShotClassifier(model, device=device)
        else:
            return Text2TextFewShotClassifier(model, device=device)
    else:
        if task == "zero-shot":
            return GenerativeZeroShotClassifier(model, device=device)
        else:
            return GenerativeFewShotClassifier(model, device=device)


def find_example_indices(
    data: pd.DataFrame, column: str, n_examples_per_class: int = 5
) -> pd.Index:
    """Finds N random examples of each label in the data set and
    returns the indices of these."""
    return data.groupby(column).sample(n_examples_per_class).index


def main():
    parser = create_parser()
    args = parser.parse_args()
    task = args.task
    model = args.model
    column = args.outcome_column
    out_dir = args.out_dir
    if args.config is not None:
        config = Config().from_disk(args.config)
        task = config["inference"]["task"]
        model = config["inference"]["model"]
        column = config["inference"]["outcome_column"]
        device = config["inference"]["device"]
    if task not in {"few-shot", "zero-shot"}:
        raise ValueError(
            f"Task should either be few-shot or zero-shot, recieved {task}"
        )
    if column not in {"political", "exemplar"}:
        raise ValueError(
            f"Column should either be political or exemplar, recieved {column}"
        )
    print(f"{task} classification over {column} with {model}.")

    print("Creating output directory.")
    Path(out_dir).mkdir(exist_ok=True)

    print("Loading data")
    data = pd.read_csv(args.in_path, index_col=0)
    data = prepare_data(data)

    print("Loading model")
    classifier = prepare_model(model, task, device=device)

    print("Fitting model")
    train_indices = find_example_indices(data, column)
    X = data.raw_text[train_indices]
    y = data[column][train_indices]
    classifier.fit(X, y)
    data["train_test_set"] = "test"
    if task == "few-shot":
        data["train_test_set"][train_indices] = "train"

    print("Inference")
    data[f"pred_{column}"] = classifier.predict(data.raw_text)

    print("Saving predictions.")
    model_file = model.replace("/", "-")
    out_path = Path(out_dir).joinpath(f"{task}_pred_{column}_{model_file}.csv")
    data.to_csv(out_path)
    print("DONE")


if __name__ == "__main__":
    main()
