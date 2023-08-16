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
        else:
            short_names.append(i)

    return short_names


def clean_dataframe(df: pd.DataFrame()) -> pd.DataFrame:
    df = df.rename(columns={"Unnamed: 0": "label"})

    model_order = np.sort(df["models"].unique())
    short_names = shorten_modelnames(model_order)

    df["models"] = pd.Categorical(df["models"], ordered=True, categories=model_order)
    df["models"] = df["models"].cat.rename_categories(short_names)

    return df


def make_f1_fig(df: pd.DataFrame) -> sns.axisgrid.FacetGrid:
    options = ["political", "exemplar"]
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


def make_prec_rec_fig(df: pd.DataFrame):
    options = ["political", "exemplar"]
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
    df = pd.read_csv("output/tweet_classification_outputs.csv")
    Path("figures").mkdir(exist_ok=True)

    df = clean_dataframe(df)

    f1_figure = make_f1_fig(df)
    acc_figure = make_acc_fig(df)
    prec_rec_figure = make_prec_rec_fig(df)

    out_path = "figures/"

    f1_figure.savefig(f"{out_path}f1_figure.png")
    acc_figure.savefig(f"{out_path}acc_figure.png")
    prec_rec_figure.save(f"{out_path}prec_rec_figure.png")


if __name__ == "__main__":
    main()
