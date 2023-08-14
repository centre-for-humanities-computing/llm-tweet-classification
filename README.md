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
