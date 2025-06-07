# Mental Health Text Sentiment Classification

A machine learning project for sentiment classification of mental health texts using various embedding models and classifiers.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py              # Configuration parameters
├── data_processor.py      # Data loading and preprocessing
├── model_trainer.py       # Model training and evaluation
├── visualizer.py         # Visualization utilities
├── data_visualizer.py    # Data analysis visualization
├── main.py              # Main execution script
├── data/                # Data directory
│   └── Combined Data.csv # Dataset file
├── models/              # Saved models
├── plots/               # Generated visualizations
└── results/             # Evaluation results
```

## Setup

1. Create and activate virtual environment:
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

The input dataset should be a CSV file with:
- `statement`: Text content
- `status`: Sentiment label (positive/negative/neutral)

## Features

- Multiple embedding models:
  - all-MiniLM-L6-v2
  - all-mpnet-base-v2
  - all-distilroberta-v1
  - all-MiniLM-L12-v2

- Classifiers:
  - Logistic Regression
  - Random Forest
  - Multi-layer Perceptron

- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

- Visualizations:
  - Text length distribution
  - Class distribution
  - Confusion matrices
  - Feature importance
  - Model performance comparison

## Usage

Run the main script:
```bash
python main.py
```

## Output

- Trained models saved in `models/`
- Visualizations saved in `plots/`
- Evaluation metrics saved in `results/`