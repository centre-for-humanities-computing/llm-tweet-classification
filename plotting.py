import argparse
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from plotnine import (
    ggplot,
    aes,
    geom_point,
    labs,
    facet_grid,
    theme_bw,
    geom_text,
    scale_x_continuous,
    scale_y_continuous,
    scale_color_brewer,
    theme,
)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LLM Classification Plotting")
    parser.add_argument("-df", "--data_folder", type=str)
    parser.add_argument("-o", "--options", nargs = '+', default = [])

    return parser

def shorten_modelnames(models: list) -> list:
    short_names = []

    for i in models:
        if "t5" in i:
            name = re.search(r"(t5.*)", i).group(1)
            short_names.append(name)
        elif "Beluga" in i:
            name = re.search(r"(Beluga.*)", i).group(1)
            short_names.append(name.lower())
        elif "Llama" in i:
            name = re.search(r"(Llama.*b)", i).group(1)
            short_names.append(name.lower())
        elif "BAAI" in i:
            name = re.search(r"(bge.*)-en", i).group(1)
            short_names.append(name)
        elif "Mini" in i:
            name = re.search(r"(all.*)-v2", i).group(1)
            short_names.append(name.lower())
        else:
            short_names.append(i)

    return short_names


def clean_dataframe(df: pd.DataFrame()) -> pd.DataFrame:
    df = df.rename(columns={"Unnamed: 0": "label"})

    model_order = ["sentence-transformers-all-MiniLM-L6-v2",
                    "BAAI-bge-large-en",
                    "google-flan-t5-xxl",
                    "stabilityai-StableBeluga-13B",
                    "gpt-3.5-turbo",
                    "gpt-4"]

    short_names = ["all-minilm-l6", "bge-large", "t5-xxl", "beluga-13b", "gpt-3.5-turbo", "gpt4"]
    #short_names = shorten_modelnames(model_order)

    df["models"] = pd.Categorical(df["models"], ordered=True, categories=model_order)
    df["models"] = df["models"].cat.rename_categories(short_names)

    return df


def make_f1_fig(df: pd.DataFrame, options: list) -> sns.axisgrid.FacetGrid:
    # selecting rows based on condition
    subset = df[df["label"].isin(options)]

    f1_fig = sns.relplot(
        data=subset,
        x="models",
        y="f1-score",
        hue="tasks",
        col="columns",
        kind="line",
    )
    f1_fig.set_xticklabels(rotation=10)

    return f1_fig


def make_acc_fig(df: pd.DataFrame) -> sns.axisgrid.FacetGrid:
    options = ["accuracy"]
    # selecting rows based on condition
    subset = df[df["label"].isin(options)]

    acc_fig = sns.relplot(
        data=subset,
        x="models",
        y="support",
        hue="tasks",
        col="columns",
        kind="line",
    )
    acc_fig = (
        acc_fig.refline(y=0.5).set_ylabels("Accuracy").set_xticklabels(rotation=10)
    )

    return acc_fig


def make_prec_rec_fig(df: pd.DataFrame, options: list):
    # selecting rows based on condition
    subset = df[df["label"].isin(options)]

    prec_rec_fig = (
        ggplot(subset, aes("precision", "recall", color="models"))
        + geom_point()
        + geom_text(aes(label="models"), size=8, nudge_y=0.04)
        + facet_grid("tasks~columns")
        + theme_bw()
        + scale_x_continuous(limits=[0, 1])
        + scale_y_continuous(limits=[0, 1])
        + scale_color_brewer(type="qual", palette=2)
        + theme(legend_position="none")
    )

    return prec_rec_fig


def main():
    parser = create_parser()
    args = parser.parse_args()

    df = pd.read_csv(f"output/{args.data_folder}_outputs.csv")
    Path(f"{args.data_folder}_figures").mkdir(exist_ok=True)

    df = clean_dataframe(df)

    f1_figure = make_f1_fig(df, args.options)
    acc_figure = make_acc_fig(df)
    prec_rec_figure = make_prec_rec_fig(df, args.options)

    out_path = f"{args.data_folder}_figures/"

    f1_figure.savefig(f"{out_path}f1_figure.png")
    acc_figure.savefig(f"{out_path}acc_figure.png")
    prec_rec_figure.save(f"{out_path}prec_rec_figure.png")


if __name__ == "__main__":
    main()
