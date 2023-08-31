import argparse
import pandas as pd
import seaborn as sns
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LLM Classification Plotting")
    parser.add_argument("--data_dir", type=str, default="opendata_preds/")

    return parser


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Unnamed: 0": "label"})

    model_order = [
        "sentence-transformers-all-MiniLM-L6-v2",
        "BAAI-bge-large-en",
        "google-flan-t5-xxl",
        "stabilityai-StableBeluga-13B",
    ]

    short_names = ["all-minilm-l6", "bge-large", "t5-xxl", "beluga-13b"]

    df["models"] = pd.Categorical(df["models"], ordered=True, categories=model_order)
    df["models"] = df["models"].cat.rename_categories(short_names)

    return df


def make_f1_fig(df: pd.DataFrame) -> sns.axisgrid.FacetGrid:
    options = ["accuracy", "macro avg", "weighted avg"]
    subset = df[~df["label"].isin(options)]

    f1_fig = sns.FacetGrid(
        subset, col="columns", hue="models", col_wrap=3, palette="colorblind"
    )

    f1_fig.map(sns.barplot, "f1-score", "models", errcolor="0", errwidth="1")

    return f1_fig


def make_acc_fig(df: pd.DataFrame) -> sns.axisgrid.FacetGrid:
    acc = df.loc[df["label"] == "accuracy"]

    # adding gilardi data
    gil = {
        "label": ["accuracy"] * 10,
        "f1-score": [
            0.565,
            0.57,
            0.3785,
            0.3825,
            0.7275,
            0.7025,
            0.7675,
            0.7865,
            0.609,
            0.618,
        ],
        "models": ["ChatGPT (temp 1)", "ChatGPT (temp 0.2)"] * 5,
        "tasks": ["zero shot"] * 10,
        "columns": [
            "problemsolution",
            "problemsolution",
            "frame",
            "frame",
            "relevant",
            "relevant",
            "stance",
            "stance",
            "topic",
            "topic",
        ],
    }

    gil = pd.DataFrame(gil)

    acc = pd.concat([acc, gil])

    acc_fig = sns.FacetGrid(
        acc,
        col="columns",
        hue="models",
        col_wrap=3,
        palette="colorblind",
    )

    acc_fig.map(
        sns.barplot, "f1-score", "models", order=acc["models"].unique()
    ).set_xlabels("Accuracy")

    return acc_fig


def main():
    parser = create_parser()
    args = parser.parse_args()

    data_name = Path(args.data_dir).name
    df = pd.read_csv(f"output/{data_name}_outputs.csv")

    df = clean_dataframe(df)

    f1_figure = make_f1_fig(df)
    acc_figure = make_acc_fig(df)

    Path(f"{data_name}_figures").mkdir(exist_ok=True)

    out_path = f"{data_name}_figures/"

    f1_figure.savefig(f"{out_path}f1_figure.png", dpi=300)
    acc_figure.savefig(f"{out_path}acc_figure.png", dpi=300)


if __name__ == "__main__":
    main()
