# llm-tweet-classification
Classifying tweets with large language models with zero- and few-shot learning.

## Getting Started

Install all requirements for the LLM classification script.
```bash
pip install -r requirements.txt
```

## Inference
The repo contains a CLI script `llm_classification.py`.
You can use it for running arbitrary classification tasks in `.tsv` or `.csv` files with Large Language models from either
HuggingFace or OpenAI.

If you intend to use OpenAI models, you will have to specify your API key and ORG as environment variables.

```bash
export OPENAI_API_KEY="..."
export OPENAI_ORG="..."
```

The script has one command-line argument, namely a config file of the following format:

```
[paths]
in_file="labelled_data.csv"
out_dir="predictions/"

[system]
seed=0
device="cpu"

[model]
name="google/flan-t5-base"
task="few-shot"

[inference]
x_column="raw_text"
y_column="exemplar"
n_examples=5
```

If you intend to use a custom prompt for a given model, you can save it in a txt file and add its path to the
`paths` section of the config.

 ```
[paths]
in_file="labelled_data.csv"
out_dir="predictions/"
prompt_file="custom_prompt.txt"
 ```

You can run the CLI like this:

```bash
python3 llm_classification.py "config.cfg"
```

### Config Documentation
- Paths:
    - in_file: `str` - Path to input file, either `.csv` or `.tsv`
    - out_dir: `str` - Output directory. The script creates one if not already there.
- System:
    - seed: `int` - Random seed for selecting few-shot examples. Is ignored when `task=="zero-shot"`
    - device: `str` - Device to run inference on. Change to `cuda:0` if you want to run on GPU.
- Model:
    - name: `str` - Name of the model from OpenAI or HuggingFace.
    - task: `{"few-shot", "zero-shot"}` - Indicates whether zero-shot or few-shot inference should be run.
- Inference:
    - x_column: `str` - Name of independent variable in the table.
    - y_column: `str` - Name of dependent variable in the table.
    - n_examples: `int` - Number of examples to give to few-shot models. Is ignored when `task=="zero-shot"`

## OpenAI script

For ease of use we have developed a script that generates predictions for all OpenAI models in one run. We did this, because OpenAI inference can run on low performance instances, as such it isn't a problem if it takes a long time to run.
Additionally since all instances access the same API, and there are rate limits, we could not start multiple instances and run them in parallel.

Paths in this script are hardcoded and you might need to adjust it for personal use.

```bash
python3 run_gpt_inference.py
```

## Output

This will output a table with predictions added to the `out_dir` folder in the config.

The file name format is as follows:
```python
f"predictions/{task}_pred_{column}_{model}.csv"
```

Each table will have a `pred_<y_column>` and also a `train_test_set` column that is labelled `train` for all examples included in the prompt for few-shot
learning and `test` everywhere else.

## Evaluating results
To evaluate the performance of the model(s), you can run the CLI `evaluation.py` script. It has two command line arguments: --data_folder and --output_folder. These, respectively, refer to the folder in which the predictions from the llm_classification.py script has been saved, and the folder where the classification report(s) should be saved. 
It can be run as follows:
```python
python3 evaluation.py -df "your/data/path" -of "your/out/path"
```
It expects the output file(s) from `llm_classification.py` in the specified file name format and placement. 
It will output two files to the specified out folder: 
- a txt file with the classification report for the test data for each of the files in the `predictions/` folder. 
- a csv file with the same information as the txt file, but which can be used for plotting the results. 

## Plotting results
The `plotting.py` script takes the csv-file produced by the evaluation script and makes three plots:
- acc_figure.png: The accuracy for each of the models in each task (zero-shot and few-shot) with the grey line indicating 50% (chance level) It's split into two, with the left side being the political column and the right being exemplar. 
- f1_figure.png: The f1-score for positive labels for each model in each task – again split into political and exemplar. 
- prec_rec_figure.png: Precision plotted against recall for each of the models, split into two rows and two columns. Rows indicate task (zero-shot, few-shot), columns indiciate label column (political, exemplar)

It has a single command line argument: --data_folder. This should be the same as the one specified when running the evaluation.py script. Examples: 
```python
python3 plotting.py -df "your/data/path"  
```


These are all saved in a new figures folder which adds #_figures# to the final data_folders name. 
