import argparse
from pathlib import Path

import pandas as pd
import patchworklib as pw
import plotnine as p9
from plotnine import (
    ggplot,
    aes,
    geom_point,
    facet_grid,
    theme_bw,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    scale_color_brewer,
    position_dodge,
    theme,
    element_text,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LLM Classification Evaluation")
    parser.add_argument("--in_dir", type=str, default="output/")
    return parser


def reorder_models(df: pd.DataFrame) -> pd.DataFrame:
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
        "All-MiniLM-l6",
        "BGE-large",
        "T5-XXL",
        "StableBeluga-13b",
        "GPT-3.5-turbo",
        "GPT-4",
        "DistilBERT",
        "GloVe200",
    ]

    df["models"] = pd.Categorical(
        df["models"], ordered=True, categories=model_order
    )
    df["models"] = df["models"].cat.rename_categories(short_names)

    return df


def reorder_tasks(df: pd.DataFrame) -> pd.DataFrame:
    df["tasks"] = (
        df["tasks"]
        .astype("category")
        .cat.reorder_categories(["zero-shot", "few-shot", "supervised"])
    )

    return df


def create_accuracy_column(df: pd.DataFrame) -> pd.DataFrame:
    # rename columns
    df = df.rename(columns={"Unnamed: 0": "outcome", "support": "accuracy"})

    # filter for only accuracy rows
    df_acc = df.loc[df["outcome"] == "accuracy"]

    # filter for only positive labels
    df = df.loc[(df["outcome"] == "political") | (df["outcome"] == "exemplar")]

    # add the new accuracy column to the rest of the data
    df["accuracy"] = df_acc["accuracy"].values

    return df


def clean_cv_df(df: pd.DataFrame) -> pd.DataFrame:
    # renaming
    df = df.rename(columns={"model": "models"})
    # adding new columns
    df["tasks"] = "supervised"
    df["prompt"] = "generic"

    return df


def capitalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["outcome"] = df["outcome"].str.capitalize()
    df["prompt"] = df["prompt"].str.capitalize()

    return df


def make_f1_fig(df: pd.DataFrame) -> ggplot:
    options = ["Political", "Exemplar"]

    # selecting rows based on condition
    subset = df[df["outcome"].isin(options)]

    f1_fig = (
        ggplot(subset, aes("models", "f1-score", color="tasks", group="tasks"))
        + geom_point(position=position_dodge(width=0.1))
        + facet_grid("prompt~outcome")
        + theme_bw()
        + scale_color_brewer(type="qual", palette="Dark2")
        + theme(axis_text_x=element_text(rotation=90))
        + labs(x="Model", y="F1-score", color="Task")
    )

    return f1_fig


def make_acc_fig(df: pd.DataFrame) -> ggplot:
    acc_fig = (
        ggplot(df, aes("models", "accuracy", color="tasks", group="tasks"))
        + geom_point(position=position_dodge(width=0.1))
        + facet_grid("prompt~outcome")
        + theme_bw()
        + scale_color_brewer(type="qual", palette="Dark2")
        + theme(axis_text_x=element_text(rotation=90))
        + labs(x="Model", y="Accuracy", color="Task")
    )

    return acc_fig


def make_prec_rec_fig(df: pd.DataFrame) -> ggplot:
    options = ["Political", "Exemplar"]
    # selecting rows based on condition
    subset = df[df["outcome"].isin(options)]

    prec_rec_fig = (
        ggplot(df, aes("precision", "recall", color="models"))
        + geom_point()
        + facet_grid(
            "tasks ~ outcome + prompt", labeller=p9.labeller(cols=col_func)
        )
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=30, size=7))
        + scale_x_continuous(limits=[0, 1])
        + scale_y_continuous(limits=[0, 1])
        + scale_color_brewer(type="qual", palette=2)
        + labs(x="Precision", y="Recall", color="Model")
    )

    return prec_rec_fig


def combine_figs(plot1, plot2):
    g1 = pw.load_ggplot(plot1, figsize=(5, 5))
    g2 = pw.load_ggplot(plot2, figsize=(5, 5))

    g1_g2 = g1 | g2

    return g1_g2


def col_func(s: str) -> str:
    """ 
    make facet labels be Outcome + Prompt
    """
    if s == "Exemplar" or s == "Political":
        return f"{s} +"
    else:
        return s


def main():
    parser = create_parser()
    args = parser.parse_args()

    paths = ["predictions", "predictions_custom"]

    llm_df = pd.DataFrame()

    for path in paths:
        df = pd.read_csv(f"{args.in_dir}/{path}_outputs.csv")

        if path == "predictions":
            df["prompt"] = "generic"

        elif path == "predictions_custom":
            df["prompt"] = "custom"

        llm_df = pd.concat([llm_df, df])

    llm_df = create_accuracy_column(llm_df)

    cv_df = pd.read_csv("output/cv_scores_supervised.csv")
    cv_df = clean_cv_df(cv_df)

    full_df = pd.concat([cv_df, llm_df])

    full_df = reorder_models(full_df)
    full_df = reorder_tasks(full_df)
    full_df = capitalize_columns(full_df)

    f1_figure = make_f1_fig(full_df)
    acc_figure = make_acc_fig(full_df)
    prec_rec_figure = make_prec_rec_fig(full_df)

    out_path = "figures/"

    Path(out_path).mkdir(exist_ok=True)

    f1_acc_fig = combine_figs(f1_figure, acc_figure)

    f1_acc_fig.savefig(f"{out_path}f1_acc_figure.png", dpi=300)
    prec_rec_figure.save(f"{out_path}prec_rec_figure.png", dpi=300)


if __name__ == "__main__":
    main()
