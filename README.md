# Emotion Detection with BERT and RoBERTa

## Overview

This project implements a multi-label emotion classification system for English tweets using Transformer-based language models.

Two pretrained Transformer models were fine-tuned and compared:

- BERT (`bert-base-uncased`)
- RoBERTa (`roberta-base`)

The task consists of predicting one or more emotions for each tweet.

The final experiments were performed using Google Colab with GPU acceleration.

---

## Dataset

The experiments use the **SemEval-2018 Task 1: Affect in Tweets** dataset.

The dataset contains English tweets annotated with eleven emotions:

- anger
- anticipation
- disgust
- fear
- joy
- love
- optimism
- pessimism
- sadness
- surprise
- trust

The dataset files are loaded from a Hugging Face-hosted repository during execution.

The task is treated as a **multi-label classification problem**, since a tweet can be associated with multiple emotions.

---

## Project Structure

```text
emotion-classification-project/
│
├── models/
│   ├── bert/
│   └── roberta/
│
├── results/
│   ├── bert/
│   ├── roberta/
│   └── comparison/
│
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   └── __init__.py
│
├── requirements.txt
├── .gitignore
└── README.md

The trained model directories are excluded from version control through .gitignore.

The results/ directory contains the evaluation results and generated plots obtained from the final experiments.

Training

The project supports both BERT and RoBERTa.

BERT
python -m src.train --model bert
RoBERTa
python -m src.train --model roberta

The full training configuration uses:

3 training epochs
training batch size: 8
evaluation batch size: 16
learning rate: 2e-5
weight decay: 0.01
maximum sequence length: 64 tokens
best checkpoint selected using validation Micro-F1
Quick Test

A reduced training mode is available for testing the training pipeline:

python -m src.train --model bert --quick

Quick mode uses a small subset of the dataset and one training epoch. It is intended only for testing the pipeline and was not used for the final reported results.

The training script automatically uses CUDA when a compatible GPU is available and falls back to CPU otherwise.

Evaluation

After training, the models can be evaluated using the following commands.

BERT
python -m src.evaluate \
    --model_dir models/bert \
    --results_dir results/bert
RoBERTa
python -m src.evaluate \
    --model_dir models/roberta \
    --results_dir results/roberta

The evaluation script:

loads the trained model;
evaluates the validation set;
estimates emotion-specific thresholds using the validation set;
evaluates the test set using the baseline threshold;
evaluates the test set using the optimized thresholds;
saves classification reports, thresholds and summary metrics.
Threshold Optimization

The evaluation procedure compares two prediction strategies.

Baseline

A fixed threshold of 0.5 is applied independently to all emotions.

Optimized Thresholds

A separate threshold is estimated for each emotion.

Thresholds are selected using only the validation set and are then applied to the test set.

Therefore, the test set is not used for threshold selection, preventing data leakage during threshold optimization.

Results

The final test-set results are:

Model	Baseline Micro-F1	Baseline Macro-F1	Optimized Micro-F1	Optimized Macro-F1
BERT	0.6945	0.5040	0.6862	0.5870
RoBERTa	0.7077	0.5282	0.7042	0.5875

RoBERTa achieves the best overall performance.

Threshold optimization substantially improves Macro-F1 for both models, while producing a small decrease in Micro-F1.

The repository contains additional per-emotion classification reports, threshold values, comparison tables and plots.

Requirements

The main dependencies are:

PyTorch
Transformers
Datasets
Accelerate
NumPy
Scikit-learn

Install the dependencies with:

pip install -r requirements.txt

The project was developed and evaluated using a GPU-enabled Google Colab environment.

Execution Environment

The final experiments were performed on Google Colab using a Tesla T4 GPU.

GPU acceleration is recommended for full training. The project can also be executed on CPU, although full training will take considerably longer.

The local repository does not include the trained model checkpoints because they are excluded through .gitignore.

The evaluation results generated from the final experiments are included in the results/ directory.

Reproducibility

The training and evaluation scripts provide the complete experimental pipeline:

Load and preprocess the dataset.
Load a pretrained BERT or RoBERTa model.
Fine-tune the model on the training set.
Select the best checkpoint using validation Micro-F1.
Evaluate the model on the test set.
Estimate emotion-specific thresholds using the validation set.
Evaluate the optimized predictions on the test set.
Save the resulting metrics and reports.

The dataset is downloaded automatically during execution from the Hugging Face-hosted dataset files defined in src/dataset.py.

Project Status

The project contains the source code, final evaluation results and documentation for the BERT and RoBERTa experiments.

The GitHub repository is private and is intended for academic evaluation.
