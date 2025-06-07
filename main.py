import logging
import os
import time
import numpy as np
import json
from pathlib import Path
from data_processor import DataProcessor
from model_trainer import ModelTrainer, LLMClassifier
from visualizer import Visualizer
from data_visualizer import DataVisualizer
from config import EMB_MODELS
import joblib

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

# === Utility Functions ===
def setup_directories():
    """Ensure output directories exist."""
    for name in ['plots', 'models', 'results']:
        Path(name).mkdir(exist_ok=True)
        logging.info(f"Directory created or exists: {name}")

def save_metrics_to_json(metrics_dict, filename):
    """Save metrics to JSON file, converting numpy types to Python native types."""

    metrics_json = {}
    for model_name, metrics in metrics_dict.items():
        metrics_json[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1'])
        }
    
  
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved metrics to {filename}")

def evaluate_llm(processor, trainer, viz, df, X_test_idx, y_test):
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
                predictions = llm.predict(df['statement'].iloc[X_test_idx], processor.label_map, prompt)
                metrics = trainer.eval(predictions, X_test_idx, y_test, processor.label_map)
                results[f"llm_{prompt}"] = metrics

                viz.plot_cm(y_test, metrics['y_pred'], processor.label_map,
                            f"Confusion Matrix - LLM {prompt}",
                            f"confusion_matrix_llm_{prompt}.png")
            except Exception as e:
                logging.error(f"LLM prompt error ({prompt}): {e}")

    except Exception as e:
        logging.error(f"LLM init error: {e}")

    return results

def main():
    try:

        setup_directories()
        

        viz = Visualizer()
        data_viz = DataVisualizer()
        
        first_model_name = list(EMB_MODELS.keys())[0]
        
        model_results = {}
        for model_name in EMB_MODELS.keys():
            logging.info(f"Processing and evaluating model: {model_name}")

            model_plot_dir = f'plots/{model_name}'
            os.makedirs(model_plot_dir, exist_ok=True)
   
            processor = DataProcessor(model_name)

            X_train, X_test, y_train, y_test, df = processor.process_data()

            if model_name == first_model_name:
                try:
                    data_viz.visualize_data(df)
                except Exception as e:
                    logging.error(f"Data visualization error: {str(e)}")
                    logging.error("Skipping data visualization...")
            trainer = ModelTrainer(viz)
            model_results[model_name] = trainer.train_and_evaluate(
                X_train, X_test, y_train, y_test, model_name
            )
        results_path = 'results/all_models_metrics.json'
        save_metrics_to_json(model_results, results_path)
        logging.info(f"Saved metrics to {results_path}")
        viz.plot_metrics(
            model_results,
            title='All Models Performance Comparison',
            filename='plots/all_models_performance_comparison.png'
        )
        logging.info("Saved performance comparison plot")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()