from glob import glob
from pathlib import Path

import pandas as pd
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

    return df


def reorder_tasks(df: pd.DataFrame) -> pd.DataFrame:
    df["tasks"] = (
        df["tasks"]
        .astype("category")
        .cat.reorder_categories(["zero-shot", "few-shot", "supervised"])
    )

    return df


def add_acc_rows(df: pd.DataFrame) -> pd.DataFrame:

    for acc_score in df['accuracy'].unique():
        new_row = {'label': 'accuracy', 'accuracy': acc_score}
        df.loc[len(df)] = new_row

    return df


def clean_cv_df(df: pd.DataFrame, model: str, column: str) -> pd.DataFrame:
    # creating label column first as that is needed for add_acc_rows func
    df['label'] = column

    # adding accuracy rows for the accuracy plot
    df = add_acc_rows(df)

    # renaming 
    df = df.rename(
    columns={
        "f1_score": "f1-score",
        "recall_score": "recall",
        "precision_score": "precision"        }
)
    # adding new columns
    df["tasks"] = "supervised"
    df["columns"] = column
    df["prompt"] = "generic"
    df["models"] = model

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

    acc_fig = (
        ggplot(subset, aes("models", "accuracy", color="tasks", group="tasks"))
        + geom_point(position=position_dodge(width=0.1))
        + facet_grid("prompt~columns")
        + theme_bw()
        + scale_color_brewer(type="qual", palette="Dark2")
        + theme(axis_text_x=element_text(rotation=90))    
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

    full_df = full_df.rename(
        columns={"Unnamed: 0": "label", "support": "accuracy"}
    )

    cv_files = glob(("output/cv_scores*.csv"))

    for file in cv_files:
        _, _, column, model = str(Path(file).stem).split("_")
        cv_df = pd.read_csv(file)
        cv_df = add_acc_rows(cv_df)

        cv_df = clean_cv_df(cv_df, model, column)
        
        full_df = pd.concat([full_df, cv_df])

    full_df = reorder_models(full_df)
    full_df = reorder_tasks(full_df)

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
