import logging
import os
import json
import pandas as pd
from pathlib import Path
from model_trainer import LLMClassifier
from visualizer import Visualizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_baseline.log"),
        logging.StreamHandler()
    ]
)

def setup_directories():
    """Ensure output directories exist."""
    for name in ['plots/llm', 'results']:
        Path(name).mkdir(exist_ok=True)
        logging.info(f"Directory created or exists: {name}")

def save_metrics_to_json(metrics_dict, filename):
    """Save metrics to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved metrics to {filename}")

def load_and_split_data(data_path='data/Combined Data.csv', test_size=0.2, random_state=42):
    """Load and split data for LLM evaluation."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['statement', 'status'])
    
    # Create label map
    unique_labels = df['status'].unique()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['status'])
    
    return train_df, test_df, label_map

def evaluate_llm(viz, test_df, label_map):
    """Evaluate zero/few-shot LLM classifier."""
    if 'OPENAI_API_KEY' not in os.environ:
        logging.warning("LLM evaluation skipped: OPENAI_API_KEY not found.")
        return {}

    results = {}
    try:
        llm = LLMClassifier(os.environ['OPENAI_API_KEY'])

        for prompt in ['basic', 'detailed', 'few_shot']:
            logging.info(f"LLM prompt: {prompt}")
            try:
                predictions = llm.predict(test_df['statement'], label_map, prompt)
                metrics = {
                    'accuracy': float(accuracy_score(test_df['status'], predictions)),
                    'precision': float(precision_score(test_df['status'], predictions, average='weighted')),
                    'recall': float(recall_score(test_df['status'], predictions, average='weighted')),
                    'f1': float(f1_score(test_df['status'], predictions, average='weighted'))
                }
                results[f"llm_{prompt}"] = metrics

                cm = confusion_matrix(test_df['status'], predictions)
                viz.plot_confusion_matrix(
                    cm,
                    title=f"Confusion Matrix - LLM {prompt}",
                    filename=f"plots/llm/confusion_matrix_{prompt}.png"
                )
            except Exception as e:
                logging.error(f"LLM prompt error ({prompt}): {e}")

    except Exception as e:
        logging.error(f"LLM init error: {e}")

    return results

def main():
    try:
        setup_directories()
        viz = Visualizer()
        
        # Load and split data
        train_df, test_df, label_map = load_and_split_data()
        
        # Evaluate LLM
        llm_results = evaluate_llm(viz, test_df, label_map)
        
        # Save results
        results_path = 'results/llm_metrics.json'
        save_metrics_to_json(llm_results, results_path)
        logging.info(f"Saved LLM metrics to {results_path}")
        
    except Exception as e:
        logging.error(f"LLM baseline evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 