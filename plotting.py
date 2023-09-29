import argparse
from glob import glob

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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Unnamed: 0": "label", "support": "accuracy"})

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
        "glove200",
    ]

    df["models"] = pd.Categorical(
        df["models"], ordered=True, categories=model_order
    )
    df["models"] = df["models"].cat.rename_categories(short_names)

    df["tasks"] = (
        df["tasks"]
        .astype("category")
        .cat.reorder_categories(["zero-shot", "few-shot", "supervised"])
    )

    return df


def make_f1_fig(df: pd.DataFrame):
    options = ["political", "exemplar"]

    # selecting rows based on condition

    subset = df[df["label"].isin(options)]

    f1_fig = (
        ggplot(subset, aes("models", "f1-score", color="tasks", group="tasks"))
        + geom_point(position=position_dodge(width=0.1))
        + facet_grid("prompt~columns")
        + theme_bw()
        + scale_color_brewer(type="qual", palette="Dark2")
        + theme(axis_text_x=element_text(rotation=90))
    )

    return f1_fig


def make_acc_fig(df: pd.DataFrame):
    options = ["accuracy"]
    # selecting rows based on condition
    subset = df[df["label"].isin(options)]
    subset = subset.loc[
        (subset["columns"] == "political") | (subset["columns"] == "exemplar")
    ]

    acc_fig = (
        ggplot(subset, aes("models", "accuracy", color="tasks", group="tasks"))
        + geom_point(position=position_dodge(width=0.1))
        + geom_hline(yintercept=0.5)
        + facet_grid("prompt~columns")
        + theme_bw()
        + scale_color_brewer(type="qual", palette="Dark2")
        + theme(axis_text_x=element_text(rotation=90))
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
        + facet_grid("tasks ~ label + prompt")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=20))
        + scale_x_continuous(limits=[0, 1])
        + scale_y_continuous(limits=[0, 1])
        + scale_color_brewer(type="qual", palette=2)
    )

    return prec_rec_fig


def main():
    paths = ["predictions", "predictions_custom"]

    full_df = pd.DataFrame()

    for path in paths:
        df = pd.read_csv(f"output/{path}_outputs.csv")

        if path == "predictions":
            df["prompt"] = "generic"

        elif path == "predictions_custom":
            df["prompt"] = "custom"

        full_df = pd.concat([full_df, df])

    full_df = clean_dataframe(full_df)

    cv_files = glob(("output/cv_scores*.csv"))

    for file in cv_files:
        _, _, column, model = file.split("_")

        cv_df = pd.read_csv(file)

        cv_df = cv_df.rename(
            columns={
                "<function f1_score at 0x7fca2d966200>": "f1-score",
                "<function recall_score at 0x7fca2d966950>": "recall",
                "<function precision_score at 0x7fca2d966830>": "precision",
                'Unnamed: 0': 'k-fold'
            }
        )
        cv_df["label"] = column
        cv_df["tasks"] = "supervised"
        cv_df["columns"] = column
        cv_df['prompt'] = "generic"

        if "glove" in model:
            cv_df['models'] = "glove200"
        elif "bert" in model:
            cv_df['models'] = "distilbert"

        full_df = pd.concat([full_df, cv_df])

    f1_figure = make_f1_fig(full_df)
    acc_figure = make_acc_fig(full_df)
    prec_rec_figure = make_prec_rec_fig(full_df)

    out_path = "figures/"

    Path(out_path).mkdir(exist_ok=True)

    f1_figure.save(f"{out_path}f1_figure.png", dpi=300)
    acc_figure.save(f"{out_path}acc_figure.png", dpi=300)
    prec_rec_figure.save(f"{out_path}prec_rec_figure.png", dpi=300)


if __name__ == "__main__":
    main()
