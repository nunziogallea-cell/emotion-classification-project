
import argparse
import os

import torch
from transformers import Trainer, TrainingArguments

from src.dataset import load_and_prepare_dataset
from src.model import load_model
from src.utils import compute_metrics


def main():

    # --------------------------------------------------
    # ARGUMENTS
    # --------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["bert", "roberta"],
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small CPU training test.",
    )

    args = parser.parse_args()

    print(f"Training model: {args.model}")

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    print("Loading pretrained model...")

    model, tokenizer = load_model(args.model)

    print("Model loaded.")

    # --------------------------------------------------
    # DATASET
    # --------------------------------------------------

    print("Loading dataset...")

    dataset = load_and_prepare_dataset(
        tokenizer,
        max_length=64,
    )

    print("Dataset loaded.")
    print(dataset)

    # --------------------------------------------------
    # QUICK MODE
    # --------------------------------------------------

    if args.quick:

        dataset["train"] = dataset["train"].select(
            range(min(500, len(dataset["train"])))
        )

        dataset["validation"] = dataset["validation"].select(
            range(min(100, len(dataset["validation"])))
        )

        dataset["test"] = dataset["test"].select(
            range(min(100, len(dataset["test"])))
        )

        num_train_epochs = 1

        print()
        print("QUICK TRAINING MODE")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
        print(f"Test samples: {len(dataset['test'])}")

    else:

        num_train_epochs = 3

        print()
        print("FULL TRAINING MODE")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
        print(f"Test samples: {len(dataset['test'])}")

    # --------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------

    output_dir = os.path.join(
        "models",
        args.model,
    )

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------

    training_args = TrainingArguments(

        output_dir=output_dir,

        num_train_epochs=num_train_epochs,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,

        learning_rate=2e-5,

        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="micro_f1",
        greater_is_better=True,

        logging_strategy="epoch",

        report_to="none",

        fp16=torch.cuda.is_available(),

        save_total_limit=1,
    )

    # --------------------------------------------------
    # TRAINER
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    print()
    print("Starting training...")
    print()

    trainer.train()

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------

    print()
    print("Evaluating on test set...")

    test_results = trainer.evaluate(
        dataset["test"]
    )

    print()
    print("Test results:")

    for key, value in test_results.items():
        print(f"{key}: {value}")

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    print()
    print("Saving model...")

    trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)

    print()
    print(f"Model saved to: {output_dir}")

    print()
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
