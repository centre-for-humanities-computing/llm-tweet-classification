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
    position_jitter,
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
        "distilbert",
        "glove200d",
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
    options = ["political"]

    # selecting rows based on condition

    subset = df[df["label"].isin(options)]

    f1_fig = (
        ggplot(subset, aes("models", "f1-score", color="tasks", group="tasks"))
        + geom_point()
        + theme_bw()
        + scale_color_brewer(type="qual", palette="Dark2")
        + theme(axis_text_x=element_text(rotation=10))
    )

    return f1_fig


def make_acc_fig(df: pd.DataFrame):
    options = ["accuracy"]
    # selecting rows based on condition
    subset = df[df["label"].isin(options)]
    subset = subset.loc[subset["columns"] == "political"]

    acc_fig = (
        ggplot(subset, aes("models", "support", color="tasks", group="tasks"))
        + geom_point()
        + geom_hline(yintercept=0.5)
        + theme_bw()
        + scale_color_brewer(type="qual", palette="Dark2")
        + theme(axis_text_x=element_text(rotation=10))
        + labs(y="Accuracy")
    )

    return acc_fig


def make_prec_rec_fig(df: pd.DataFrame):
    options = ["political"]
    # selecting rows based on condition
    subset = df[df["label"].isin(options)]

    prec_rec_fig = (
        ggplot(subset, aes("precision", "recall", color="models"))
        + geom_point(size=0.3, shape="+")
        # + geom_text(
        #     aes(label="models"),
        #     size=7,
        #     position=position_jitter(height=0.03),
        # )
        + facet_grid("~tasks")
        + theme_bw()
        + scale_x_continuous(limits=[0, 1])
        + scale_y_continuous(limits=[0, 1])
        + scale_color_brewer(type="qual", palette=2)
        # + theme(legend_position="none")
    )

    return prec_rec_fig


def main():
    parser = create_parser()
    args = parser.parse_args()

    data_name = Path(args.data_dir).name
    df = pd.read_csv(f"output/{data_name}_outputs.csv")

    supervised_data = pd.DataFrame(
        {
            "Unnamed: 0": ["accuracy", "political"] * 2,
            "precision": [0.86, 0.86, 0.86, 0.86],
            "recall": [0.86, 0.86, 0.8, 0.8],
            "f1-score": [0.87, 0.87, 0.83, 0.83],
            "support": [0.87, 0.87, 0.83, 0.83],
            "models": ["distilbert", "distilbert", "glove200d", "glove200d"],
            "tasks": ["supervised"] * 4,
            "columns": ["political"] * 4,
        }
    )

    df = pd.concat([df, supervised_data])

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
