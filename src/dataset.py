
from datasets import load_dataset


EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "love",
    "optimism",
    "pessimism",
    "sadness",
    "surprise",
    "trust",
]


EMOTION_MAP = {
    "0": "anger",
    "1": "anticipation",
    "2": "disgust",
    "3": "fear",
    "4": "joy",
    "5": "love",
    "6": "optimism",
    "7": "pessimism",
    "8": "sadness",
    "9": "surprise",
    "10": "trust",
}


DATASET_BASE_URL = (
    "https://huggingface.co/datasets/"
    "Hidden-States/SemEval-2018-English-Processed/"
    "resolve/main/"
)


DATA_FILES = {
    "train": DATASET_BASE_URL + "SemEval-Train.csv",
    "validation": DATASET_BASE_URL + "SemEval-Validation.csv",
    "test": DATASET_BASE_URL + "SemEval-Test.csv",
}


def load_and_prepare_dataset(tokenizer, max_length=64):
    """
    Load and preprocess the SemEval-2018 Affect in Tweets dataset.

    The dataset is:
    - loaded directly from the official CSV files hosted on Hugging Face
    - renamed from numeric emotion columns to emotion names
    - converted to multilabel format
    - tokenized using the provided tokenizer
    """

    dataset = load_dataset(
        "csv",
        data_files=DATA_FILES,
    )

    # Rename numeric emotion columns
    dataset = dataset.rename_columns(EMOTION_MAP)

    # Create multilabel target vector
    def create_labels(example):
        example["labels"] = [
            float(example[emotion])
            for emotion in EMOTIONS
        ]

        return example

    dataset = dataset.map(create_labels)

    # Tokenization
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
    )

    return dataset
