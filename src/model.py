
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

NUM_LABELS = 11

MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}


def load_model(model_name):
    """
    Load a pretrained or locally saved model.

    Accepted values:
    - "bert"
    - "roberta"
    - path to a locally saved model
    """

    # --------------------------------------------------
    # LOCAL TRAINED MODEL
    # --------------------------------------------------

    if os.path.isdir(model_name):
        print(f"Loading trained model from: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        return model, tokenizer

    # --------------------------------------------------
    # PRETRAINED MODEL
    # --------------------------------------------------

    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_NAMES.keys())} "
            f"or provide a valid model directory."
        )

    pretrained_name = MODEL_NAMES[model_name]

    print(
        f"Loading pretrained model: {pretrained_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    return model, tokenizer
