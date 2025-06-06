# Mental Health Text Sentiment Classification

This project implements sentiment classification for mental health texts using LLM embeddings and various machine learning models.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py
├── data_processor.py
├── model_trainer.py
├── visualizer.py
└── main.py
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your dataset as 'Combined Data.csv' in the project root directory.

## Usage

Run the main script:
```bash
python main.py
```

## Features

- Data preprocessing and embedding generation using SentenceTransformer
- Multiple classifier implementations:
  - Logistic Regression
  - Multi-layer Perceptron
  - Random Forest
- Comprehensive model evaluation metrics
- Data visualization:
  - Text length distribution
  - Class distribution
  - Confusion matrices
  - Word clouds

## Configuration

Model parameters and other settings can be adjusted in `config.py`.