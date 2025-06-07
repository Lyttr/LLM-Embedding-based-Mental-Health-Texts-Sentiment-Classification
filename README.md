# Mental Health Text Sentiment Classification

This project implements sentiment classification for mental health texts using LLM embeddings and various machine learning models.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py              # Configuration parameters
├── data_processor.py      # Data loading and preprocessing
├── model_trainer.py       # Model training and evaluation
├── visualizer.py         # Visualization utilities
├── main.py              # Main execution script
├── data/                # Data directory
│   └── Combined Data.csv # Dataset file
├── models/              # Saved models and embeddings
├── plots/               # Generated visualizations
└── results/             # Evaluation results and metrics
```

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your dataset as 'Combined Data.csv' in the data directory.

## Data Format

The input dataset should be a CSV file with the following columns:
- `statement`: Text content to be classified
- `status`: Sentiment label (positive/negative/neutral)

## Usage

Run the main script:
```bash
python main.py
```

## Features

- Data preprocessing and embedding generation using SentenceTransformer
- Multiple embedding models:
  - all-MiniLM-L6-v2
  - all-mpnet-base-v2
  - all-distilroberta-v1
  - all-MiniLM-L12-v2
- Multiple classifier implementations:
  - Logistic Regression
  - Random Forest
  - Multi-layer Perceptron
  - Support Vector Machine
- Comprehensive model evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
- Data visualization:
  - Text length distribution
  - Class distribution
  - Confusion matrices
  - ROC curves
  - Learning curves
  - Word clouds

## Configuration

Model parameters and other settings can be adjusted in `config.py`:

- `EMB_MODELS`: Configuration for embedding models
- `MODEL_PARAMS`: Parameters for each classifier
- `TEST_SIZE`: Test set proportion
- `CV_FOLDS`: Number of cross-validation folds
- `RANDOM_STATE`: Random seed for reproducibility

## Output

The pipeline generates:
- Trained models in `models/`
- Performance visualizations in `plots/`
- Evaluation metrics in `results/`
- Detailed logs in `pipeline.log`