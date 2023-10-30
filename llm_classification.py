"""CLI for running zero and few-shot classification on a tweet dataset with
large language models and transformers."""
import argparse
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from confection import Config
from sklearn.base import ClassifierMixin
from stormtrooper import (GenerativeFewShotClassifier,
                          GenerativeZeroShotClassifier,
                          OpenAIFewShotClassifier, OpenAIZeroShotClassifier,
                          SetFitFewShotClassifier, SetFitZeroShotClassifier,
                          Text2TextFewShotClassifier,
                          Text2TextZeroShotClassifier, ZeroShotClassifier)
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


def prepare_model(
    model: str, task: str, device: str, custom_prompt: Optional[str]
) -> ClassifierMixin:
    """Loads classifier model based on model name and task."""
    if ("gpt-3" in model) or ("gpt-4" in model):
        model_kwargs: dict[str, Any] = dict(model_name=model)
        if "gpt-4" in model:
            model_kwargs["max_requests_per_minute"] = 200
            model_kwargs["max_tokens_per_minute"] = 20_000
        else:
            model_kwargs["max_requests_per_minute"] = 3500
            model_kwargs["max_tokens_per_minute"] = 45_000
        print("Initializing connection to OpenAI")
        if custom_prompt is not None:
            model_kwargs["prompt"] = custom_prompt
        if task == "zero-shot":
            return OpenAIZeroShotClassifier(**model_kwargs)
        else:
            return OpenAIFewShotClassifier(**model_kwargs)
    else:
        # We assume the model is from HuggingFace
        model_type = get_model_type(model)
        model_kwargs = dict(model_name=model, device=device)
        if (custom_prompt is not None) and (
            model_type in ["text2text", "generative"]
        ):
            model_kwargs["prompt"] = custom_prompt
        if model_type == "text2text":
            if task == "zero-shot":
                return Text2TextZeroShotClassifier(**model_kwargs)
            else:
                return Text2TextFewShotClassifier(**model_kwargs)
        elif model_type == "generative":
            if task == "zero-shot":
                return GenerativeZeroShotClassifier(**model_kwargs)
            else:
                return GenerativeFewShotClassifier(**model_kwargs)
        elif model_type == "sentence-trf":
            if task == "zero-shot":
                return SetFitZeroShotClassifier(**model_kwargs)
            else:
                return SetFitFewShotClassifier(**model_kwargs)
        else:
            if task == "zero-shot":
                return ZeroShotClassifier(**model_kwargs)
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


def run_config(config: Config) -> None:
    task = config["model"]["task"]
    if task not in {"few-shot", "zero-shot"}:
        raise ValueError(
            f"Task should either be few-shot or zero-shot, recieved {task}"
        )

    x_column = config["inference"]["x_column"]
    y_column = config["inference"]["y_column"]
    model_name = config["model"]["name"]
    print(f"{task} classification over {y_column} with {model_name}.")

    try:
        prompt_file = config["paths"]["prompt_file"]
        with open(prompt_file, "r") as f:
            custom_prompt = f.read()
    except KeyError:
        custom_prompt = None

    print("Creating output directory.")
    out_dir = config["paths"]["out_dir"]
    Path(out_dir).mkdir(exist_ok=True)

    print("Loading data")
    data = load_data(config["paths"]["in_file"])
    data = data.dropna(subset=[x_column, y_column])
    data = data.reset_index()

    try:
        examples_path = config["paths"]["examples"]
        examples = pd.read_csv(examples_path)
        examples = examples.dropna(subset=y_column)
        X_train = examples[x_column]
        y_train = examples[y_column]
        train_indices = []
    except KeyError:
        print("Preparing training data")
        train_indices = find_example_indices(
            data,
            y_column,
            config["inference"]["n_examples"],
            seed=config["system"]["seed"],
        )
        X_train = data[x_column][train_indices]
        y_train = data[y_column][train_indices]

    data["train_test_set"] = "test"

    if task == "few-shot":
        data["train_test_set"][train_indices] = "train"

    print("Loading model")
    classifier = prepare_model(
        model_name,
        task,
        device=config["system"]["device"],
        custom_prompt=custom_prompt,
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

    return None


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = Config().from_disk(args.config)

    run_config(config)


if __name__ == "__main__":
    main()
