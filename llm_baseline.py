import logging
import os
import json
import pandas as pd
from pathlib import Path
from model_trainer import LLMClassifier
from visualizer import Visualizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_baseline.log"),
        logging.StreamHandler()
    ]
)

def check_api_key():
  
    if 'OPENAI_API_KEY' not in os.environ:
        logging.error("""
OpenAI API key not found

""")
        return False
    return True

def setup_directories():

    for name in ['plots/llm', 'results']:
        Path(name).mkdir(exist_ok=True)
        logging.info(f"Directory created or exists: {name}")

def save_metrics_to_json(metrics_dict, filename):
  
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved metrics to {filename}")

def load_data(data_path='data/Combined Data.csv', sample_size=100, random_state=42):

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['statement', 'status'])
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
        logging.info(f"Sampled {sample_size} records from dataset")
    
    unique_labels = df['status'].unique()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    logging.info(f"Test set size: {len(df)}")
    return df, label_map

def evaluate_llm(viz, test_df, label_map):
  
    if not check_api_key():
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
        
        test_df, label_map = load_data()
        llm_results = evaluate_llm(viz, test_df, label_map)
        
        results_path = 'results/llm_metrics.json'
        save_metrics_to_json(llm_results, results_path)
        logging.info(f"Saved LLM metrics to {results_path}")
        
    except Exception as e:
        logging.error(f"LLM baseline evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 