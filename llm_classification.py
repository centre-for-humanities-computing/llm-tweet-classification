"""CLI for running zero and few-shot classification on a tweet dataset with
large language models and transformers."""
import argparse
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from confection import Config
from sklearn.base import ClassifierMixin
from skllm import FewShotGPTClassifier, ZeroShotGPTClassifier
from skllm.config import SKLLMConfig
from stormtrooper import (
    GenerativeFewShotClassifier,
    GenerativeZeroShotClassifier,
    SetFitFewShotClassifier,
    SetFitZeroShotClassifier,
    Text2TextFewShotClassifier,
    Text2TextZeroShotClassifier,
    ZeroShotClassifier,
)
from transformers import AutoConfig


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LLM Classifier")
    parser.add_argument("config", type=str)
    return parser


def get_model_type(
    model: str,
) -> Literal["text2text", "generative", "zeroshot", "sentence-trf"]:
    """Determines what type a Huggingface model is.
    Raises exception if the model is not stormtrooper-compatible."""
    config = AutoConfig.from_pretrained(model)
    architectures = config.architectures
    if any("ForConditionalGeneration" in arc for arc in architectures):
        return "text2text"
    elif any("ForCausalLM" in arc for arc in architectures):
        return "generative"
    elif any("ForSequenceClassification" in arc for arc in architectures):
        return "zeroshot"
    elif any("BertModel" in arc for arc in architectures):
        return "sentence-trf"
    else:
        raise ValueError(
            "Provided HuggingFace model is not compatible with stormtrooper."
        )


def prepare_model(model: str, task: str, device: str) -> ClassifierMixin:
    """Loads classifier model based on model name and task."""
    if ("gpt-3" in model) or ("gpt-4" in model):
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
            return ZeroShotGPTClassifier(openai_model=model)
        else:
            return FewShotGPTClassifier(openai_model=model)
    else:
        # We assume the model is from HuggingFace
        model_type = get_model_type(model)
        if model_type == "text2text":
            if task == "zero-shot":
                return Text2TextZeroShotClassifier(model, device=device)
            else:
                return Text2TextFewShotClassifier(model, device=device)
        elif model_type == "generative":
            if task == "zero-shot":
                return GenerativeZeroShotClassifier(model, device=device)
            else:
                return GenerativeFewShotClassifier(model, device=device)
        elif model_type == "sentence-trf":
            if task == "zero-shot":
                return SetFitZeroShotClassifier(model, device=device)
            else:
                return SetFitFewShotClassifier(model, device=device)
        else:
            if task == "zero-shot":
                return ZeroShotClassifier(model, device=device)
            else:
                raise ValueError(
                    "You cannot use a zero shot model with task 'few-shot'."
                )


def find_example_indices(
    data: pd.DataFrame,
    column: str,
    n_examples_per_class: int,
    seed: int,
) -> pd.Index:
    """Finds N random examples of each label in the data set and
    returns the indices of these."""
    return (
        data.groupby(column)
        .sample(n_examples_per_class, random_state=seed)
        .index
    )


def load_data(in_file: str) -> pd.DataFrame:
    if in_file.endswith(".tsv"):
        return pd.read_csv(in_file, sep="\t")
    elif in_file.endswith(".csv"):
        return pd.read_csv(in_file)
    else:
        raise ValueError("Input file needs to be .csv or .tsv.")


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = Config().from_disk(args.config)
    task = config["model"]["task"]
    if task not in {"few-shot", "zero-shot"}:
        raise ValueError(
            f"Task should either be few-shot or zero-shot, recieved {task}"
        )
    x_column = config["inference"]["x_column"]
    y_column = config["inference"]["y_column"]
    model_name = config["model"]["name"]
    print(f"{task} classification over {y_column} with {model_name}.")

    print("Creating output directory.")
    out_dir = config["paths"]["out_dir"]
    Path(out_dir).mkdir(exist_ok=True)

    print("Loading data")
    data = load_data(config["paths"]["in_file"])
    data = data.reset_index()

    print("Preparing training data")
    train_indices = find_example_indices(
        data,
        y_column,
        config["inference"]["n_examples"],
        seed=config["system"]["seed"],
    )
    data["train_test_set"] = "test"
    if task == "few-shot":
        data["train_test_set"][train_indices] = "train"
    X_train = data[x_column][train_indices]
    y_train = data[y_column][train_indices]

    print("Loading model")
    classifier = prepare_model(
        model_name, task, device=config["system"]["device"]
    )

    print("Fitting model")
    classifier.fit(X_train, y_train)

    print("Inference")
    data[f"pred_{y_column}"] = classifier.predict(data[x_column])

    print("Saving predictions.")
    model_file = model_name.replace("/", "-")
    out_path = Path(out_dir).joinpath(
        f"{task}_pred_{y_column}_{model_file}.csv"
    )
    data.to_csv(out_path)
    print("DONE")


if __name__ == "__main__":
    main()
