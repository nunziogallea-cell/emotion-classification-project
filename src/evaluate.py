
import argparse
import json
import os

import numpy as np

from sklearn.metrics import (
    classification_report,
    f1_score,
)

from transformers import Trainer

from src.dataset import EMOTIONS, load_and_prepare_dataset
from src.model import load_model


def sigmoid(x):
    """Convert logits into probabilities."""
    return 1 / (1 + np.exp(-x))


def find_optimal_thresholds(logits, labels):
    """
    Find the best threshold for each emotion using
    only the validation set.
    """

    probabilities = sigmoid(logits)

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    best_thresholds = {}

    for i, emotion in enumerate(EMOTIONS):

        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in thresholds:

            predictions = (
                probabilities[:, i] >= threshold
            ).astype(int)

            f1 = f1_score(
                labels[:, i],
                predictions,
                zero_division=0,
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        best_thresholds[emotion] = round(
            float(best_threshold),
            2,
        )

    return best_thresholds


def apply_thresholds(logits, thresholds):
    """
    Apply one threshold per emotion.
    """

    probabilities = sigmoid(logits)

    predictions = np.zeros_like(
        probabilities,
        dtype=int,
    )

    for i, emotion in enumerate(EMOTIONS):

        threshold = thresholds[emotion]

        predictions[:, i] = (
            probabilities[:, i] >= threshold
        ).astype(int)

    return predictions


def print_metrics(labels, predictions, title):
    """
    Print Micro-F1 and Macro-F1.
    """

    micro_f1 = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    return micro_f1, macro_f1


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a trained emotion classification model."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained model.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory where evaluation results will be saved.",
    )

    args = parser.parse_args()

    os.makedirs(
        args.results_dir,
        exist_ok=True,
    )

    print(
        f"Loading model from: {args.model_dir}"
    )

    model, tokenizer = load_model(
        args.model_dir
    )

    print("Loading dataset...")

    dataset = load_and_prepare_dataset(
        tokenizer
    )

    trainer = Trainer(
        model=model
    )

    
    #VALIDATION
    

    print("Evaluating validation set...")

    val_output = trainer.predict(
        dataset["validation"]
    )

    val_logits = val_output.predictions
    val_labels = val_output.label_ids

    val_probabilities = sigmoid(
        val_logits
    )

    # Baseline validation
    val_baseline_predictions = (
        val_probabilities >= 0.5
    ).astype(int)

    print_metrics(
        val_labels,
        val_baseline_predictions,
        "VALIDATION - BASELINE",
    )

    # Find thresholds using validation ONLY
    best_thresholds = find_optimal_thresholds(
        val_logits,
        val_labels,
    )

    print()
    print("Optimal thresholds:")
    print("-" * 40)

    for emotion in EMOTIONS:
        print(
            f"{emotion:15s}: "
            f"{best_thresholds[emotion]:.2f}"
        )

    
    #TEST
    

    print()
    print("Evaluating test set...")

    test_output = trainer.predict(
        dataset["test"]
    )

    test_logits = test_output.predictions
    test_labels = test_output.label_ids

    # Baseline: threshold = 0.5
    test_probabilities = sigmoid(
        test_logits
    )

    test_baseline_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    baseline_micro_f1, baseline_macro_f1 = print_metrics(
        test_labels,
        test_baseline_predictions,
        "TEST - BASELINE (threshold = 0.5)",
    )

    # Optimized thresholds
    test_optimized_predictions = apply_thresholds(
        test_logits,
        best_thresholds,
    )

    optimized_micro_f1, optimized_macro_f1 = print_metrics(
        test_labels,
        test_optimized_predictions,
        "TEST - OPTIMIZED THRESHOLDS",
    )

    
    #CLASSIFICATION REPORTS
    

    baseline_report = classification_report(
        test_labels,
        test_baseline_predictions,
        target_names=EMOTIONS,
        zero_division=0,
        digits=4,
    )

    optimized_report = classification_report(
        test_labels,
        test_optimized_predictions,
        target_names=EMOTIONS,
        zero_division=0,
        digits=4,
    )

    # Save baseline report
    baseline_path = os.path.join(
        args.results_dir,
        "classification_report_baseline.txt",
    )

    with open(
        baseline_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(baseline_report)

    # Save optimized report
    optimized_path = os.path.join(
        args.results_dir,
        "classification_report_optimized.txt",
    )

    with open(
        optimized_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(optimized_report)


    #SAVE THRESHOLDS
    

    thresholds_path = os.path.join(
        args.results_dir,
        "thresholds.json",
    )

    with open(
        thresholds_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            best_thresholds,
            f,
            indent=4,
        )

    
    #SAVE SUMMARY
    

    summary = {
        "baseline": {
            "micro_f1": baseline_micro_f1,
            "macro_f1": baseline_macro_f1,
        },
        "optimized": {
            "micro_f1": optimized_micro_f1,
            "macro_f1": optimized_macro_f1,
        },
    }

    summary_path = os.path.join(
        args.results_dir,
        "summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print("Results saved to:")
    print(args.results_dir)


if __name__ == "__main__":
    main()
