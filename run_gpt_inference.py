from contextlib import redirect_stdout
from multiprocessing import Pool
from pathlib import Path

from confection import Config

from llm_classification import run_config


def run_log_config(config: Config, log_path: str):
    """Runs config and redirects stdout to a file."""
    with open(log_path, "w") as log_file:
        with redirect_stdout(log_file):
            run_config(config)


def main():
    default_config = Config().from_disk("configs/default_gpt_config.cfg")
    log_dir = Path("gpt_runs_logfiles/")
    log_dir.mkdir(exist_ok=True)
    print("Collecting configs and log paths.")
    configs = []
    log_paths = []
    for model in ["gpt-4", "gpt-3.5-turbo"]:
        for prompt_type in ["custom", "generic"]:
            for column in ["political", "exemplar"]:
                config = default_config.copy()
                config["paths"]["out_dir"] = f"predictions_{prompt_type}/"
                if prompt_type == "custom":
                    config["paths"][
                        "prompt_file"
                    ] = f"prompts/gpt_{column}.txt"
                config["model"]["name"] = model
                config["inference"]["y_column"] = column
                configs.append(config)
                log_paths.append(
                    log_dir.joinpath(
                        f"{model}_{prompt_type}_{column}_zero-shot.log"
                    )
                )
    print("Running all jobs in parallel with processes.")
    with Pool(processes=None) as pool:
        pool.starmap(run_log_config, zip(configs, log_paths))
    print("DONE")


if __name__ == "__main__":
    main()
