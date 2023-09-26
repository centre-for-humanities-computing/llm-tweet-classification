from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from confection import Config

from llm_classification import run_config


def run_log_config(config: Config, log_path: str):
    """Runs config and redirects stdout to a file."""
    with open(log_path, "w") as log_file:
        with redirect_stdout(log_file):
            with redirect_stderr(log_file):
                run_config(config)


def main():
    default_config = Config().from_disk("configs/default_gpt_config.cfg")
    log_dir = Path("gpt_runs_logfiles/")
    log_dir.mkdir(exist_ok=True)
    print("Collecting configs and log paths.")
    for model in ["gpt-4", "gpt-3.5-turbo"]:
        for prompt_type in ["custom", "generic"]:
            for column in ["political", "exemplar"]:
                for task in ["zero-shot", "few-shot"]:
                    config = default_config.copy()
                    config["paths"]["out_dir"] = f"predictions_{prompt_type}/"
                    if prompt_type == "custom":
                        config["paths"][
                            "prompt_file"
                        ] = f"prompts/gpt_{task}_{column}.txt"
                    config["model"]["name"] = model
                    config["model"]["task"] = task
                    config["inference"]["y_column"] = column
                    print(
                        "------------------------------\n"
                        f"Running Inference with {model}\n"
                        f" - task: {task}\n"
                        f" - prompt: {prompt_type}\n"
                        f" - column: {column}\n"
                        "------------------------------"
                    )
                    run_config(config)
    print("DONE")


if __name__ == "__main__":
    main()
