import argparse
import pandas as pd
from pathlib import Path
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_hline,
    labs,
    facet_grid,
    theme_bw,
    geom_text,
    scale_x_continuous,
    scale_y_continuous,
    scale_color_brewer,
    position_dodge,
    theme,
    element_text,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LLM Classification Plotting")
    parser.add_argument("--data_dir", type=str, default="predictions/")

    return parser


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Unnamed: 0": "label"})

    model_order = [
        "sentence-transformers-all-MiniLM-L6-v2",
        "BAAI-bge-large-en",
        "google-flan-t5-xxl",
        "stabilityai-StableBeluga-13B",
        "gpt-3.5-turbo",
        "gpt-4",
        "distilbert-base-uncased",
        "glove-twitter-200",
    ]

    short_names = [
        "all-minilm-l6",
        "bge-large",
        "t5-xxl",
        "beluga-13b",
        "gpt-3.5-turbo",
        "gpt4",
        "distilbert",
        "glove200d",
    ]

    df["models"] = pd.Categorical(df["models"], ordered=True, categories=model_order)
    df["models"] = df["models"].cat.rename_categories(short_names)

    return df


def make_f1_fig(df: pd.DataFrame):
    options = ["political", "exemplar"]

    # selecting rows based on condition

    subset = df[df["label"].isin(options)]

    f1_fig = (
        ggplot(subset, aes("models", "f1-score", color="tasks", group="tasks"))
        + geom_point(position = position_dodge(width = 0.1))
        + facet_grid("~columns")
        + theme_bw()
        + scale_color_brewer(type="qual", palette="Dark2")
        + theme(axis_text_x=element_text(rotation=10))
    )

    return f1_fig


def make_acc_fig(df: pd.DataFrame):
    options = ["accuracy"]
    # selecting rows based on condition
    subset = df[df["label"].isin(options)]
    subset = subset.loc[(subset["columns"] == "political") | (subset["columns"] == "exemplar")] 

    acc_fig = (
        ggplot(subset, aes("models", "support", color="tasks", group="tasks"))
        + geom_point(position = position_dodge(width = 0.1))
        + geom_hline(yintercept=0.5)
        + facet_grid("~columns")
        + theme_bw()
        + scale_color_brewer(type="qual", palette="Dark2")
        + theme(axis_text_x=element_text(rotation=10))
        + labs(y="Accuracy")
    )

    return acc_fig


def make_prec_rec_fig(df: pd.DataFrame):
    options = ["political", "exemplar"]
    # selecting rows based on condition
    subset = df[df["label"].isin(options)]

    prec_rec_fig = (
        ggplot(subset, aes("precision", "recall", color="models"))
        + geom_point()
        + facet_grid("label~tasks")
        + theme_bw()
        + scale_x_continuous(limits=[0, 1])
        + scale_y_continuous(limits=[0, 1])
        + scale_color_brewer(type="qual", palette=2)
    )

    return prec_rec_fig


def main():
    parser = create_parser()
    args = parser.parse_args()

    data_name = Path(args.data_dir).name
    df = pd.read_csv(f"output/{data_name}_outputs.csv")

    df = clean_dataframe(df)

    df["tasks"] = (
        df["tasks"]
        .astype("category")
        .cat.reorder_categories(["supervised", "few-shot", "zero-shot"])
    )

    f1_figure = make_f1_fig(df)
    acc_figure = make_acc_fig(df)
    prec_rec_figure = make_prec_rec_fig(df)

    Path(f"{data_name}_figures").mkdir(exist_ok=True)

    out_path = f"{data_name}_figures/"

    f1_figure.save(f"{out_path}f1_figure.png", dpi=300)
    acc_figure.save(f"{out_path}acc_figure.png", dpi=300)
    prec_rec_figure.save(f"{out_path}prec_rec_figure.png", dpi=300)


if __name__ == "__main__":
    main()
