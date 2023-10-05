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


## Supervised Classification

For supervised models we made a separate script. This includes running and evaluating Glove-200d with logistic regression and finetuning DistilBert for classification.

This script requires different requirements, therefore you should install these from the appropriate file:

```bash
pip install -r supervised_requirements.txt
```

Paths in this script are hardcoded and you might need to adjust it for personal use.

```bash
python3 supervised_classification.py
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
To evaluate the performance of the model(s), you can run the CLI `evaluation.py` script. It has two command line arguments: --in_dir and --out_dir. These, respectively, refer to the folder in which the predictions from the llm_classification.py script has been saved (i.e., your predictions folder), and the folder where the classification report(s) should be saved. 
--in_dir defaults to 'predictions/' and --out_dir defaults to 'output/' (which is a folder that is created if it does not exist already)

It can be run as follows:

```python
python3 evaluation.py --in_dir "your/data/path" --out_dir "your/out/path"
```

It expects the output file(s) from `llm_classification.py` in the specified file name format and placement. 
It will output two files to the specified out folder: 
- a txt file with the classification report for the test data for each of the files in the --in_dir folder. 
- a csv file with the same information as the txt file, but which can be used for plotting the results. 

## Plotting results
The `plotting.py` script takes the csv-file produced by the evaluation script and makes three plots:
- acc_figure.png: The accuracy for each of the 8 models on each outcome (political, exemplar) in each task (zero-shot, few-shot) with each prompt type (generic, custom). It's split into four quadrants, with the left side being the exemplar column, the right being political, the upper line being custom prompts and the lower column being generic prompts. 
- f1_figure.png: The f1-score for positive labels for each model in each task – again split into political and exemplar + generic and custom prompt. 
- prec_rec_figure.png: Precision plotted against recall for each of the models, split into three rows and four columns. Rows indicate task (zero-shot, few-shot, supervised classification), columns indiciate label column (political, exemplar) and prompt type (generic, custom)


```python
python3 plotting.py
```


These are all saved in a figures/ folder.
