from typing import Iterable

import numpy as np
from datasets import Dataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    pipeline,
)


class BertFinetuneClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn classifier component for finetuning a transformer
    for sequence classification with torch.

    Parameters
    ----------
    base_model: str, default 'distilbert-base-uncased'
        Pretrained transformer to start from.
    device: str, default 'cpu'
        Device to train on.
    """

    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        device: str = "cpu",
    ):
        self.base_model = base_model
        self.device = device

    def fit(self, X: Iterable[str], y):
        y = np.array(list(y))
        X = list(X)
        self.classes_ = np.unique(y)
        self.feature_mapping_ = {
            label: index for index, label in enumerate(self.classes_)
        }
        y = np.array([self.feature_mapping_[label] for label in y])
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(self.classes_),
            id2label=dict(enumerate(self.classes_)),
            label2id=self.feature_mapping_,
        )
        self.model.to(self.device)

        def tokenize(examples):
            return self.tokenizer(
                examples["text"], padding="max_length", truncation=True
            )

        ds = Dataset.from_dict(dict(text=X, label=y))
        ds = ds.map(tokenize, batched=True)
        self.trainer = Trainer(self.model, train_dataset=ds)
        self.trainer.train()
        self.pipe = pipeline(
            "text-classification",
            self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        return self

    def predict(self, X: Iterable[str]) -> np.ndarray:
        if self.pipe is None:
            raise NotFittedError("Model has not been fitted yet.")
        results = self.pipe(list(X))
        labels = [entry["label"] for entry in results]  # type: ignore
        return np.array(labels)
