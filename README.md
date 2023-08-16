# llm-tweet-classification
Classifying tweets with large language models with zero- and few-shot learning.

## Getting Started

Install all requirements for the LLM classification script.
```bash
pip install -r requirements.txt
```

## Running Classification
The repo contains a CLI script `llm_classification.py`.
You have to pass it the name of the model (either from huggingface or OpenAI),
the task (`few-shot` or `zero-shot`) and the outcome column (`political` or `exemplar`) and
the input file (optional, if not specified it will load `labelled_data.csv`)

If you intend to use OpenAI models, you will have to specify your API key and ORG as environment variables.

```bash
export OPENAI_KEY="..."
export OPENAI_ORG="..."
```

You can run the CLI like this:

```bash
python3 llm_classification.py "google/flan-t5-base" "zero-shot" "political" --in_file "labelled_data.csv"
```

## Output

This will output a table with predictions added to the `predictions/` folder.

The file name format is as follows:
```python
f"predictions/{task}_pred_{column}_{model}.csv"
```

Each table will have a `pred_political` or `pred_exemplar` column depending on which was used
and also a `train_test_set` column that is labelled `train` for all examples included in the prompt for few-shot
learning and `test` everywhere else.

## Inference on GPU
You can specify the device on which you want to run inference. This is by default the CPU, and any Cuda device can be employed.
Keep in mind that this DOES NOT have any effect when running OpenAI's models as those do not run locally.

```bash
python3 llm_classification.py "stabilityai/StableBeluga-7B" "few-shot" "exemplar" --device "cuda:0"
```
## Evaluating results
To evaluate the performance of the model(s), you can run the `evaluation.py` script as follows.
```python
python3 evaluation.py
```
It excepts the output file(s) from `llm_classification.py` in the specified file name format and placement. 
This will output two files to the `output/` folder: 
- an out-file with the classification report for the test data for each of the files in the `predictions/` folder. 
- a csv file with the same information as the txt file, but which can be used for plotting the results. 

## Plotting results
The `plotting.py` script takes the csv-file produced by the evaluation script and makes three plots:
- acc_figure.png: The accuracy for each of the models in each task (zero-shot and few-shot) with the grey line indicating 50% (chance level) It's split into two, with the left side being the political column and the right being exemplar. 
- f1_figure.png: The f1-score for positive labels for each model in each task – again split into political and exemplar. 
- prec_rec_figure.png: Precision plotted against recall for each of the models, split into two rows and two columns. Rows indicate task (zero-shot, few-shot), columns indiciate label column (political, exemplar)

These are all saved in the `figures/` folder. 
