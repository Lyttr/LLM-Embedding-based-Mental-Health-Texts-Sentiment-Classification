import logging
import os
import time
import numpy as np
import json
from pathlib import Path
from data_processor import DataProcessor
from model_trainer import ModelTrainer
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
    for model_name, algo_metrics in metrics_dict.items():
        metrics_json[model_name] = {}
        for algo, metrics in algo_metrics.items():
            metrics_json[model_name][algo] = {}
            for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                if metric_name in metrics:
                    metrics_json[model_name][algo][metric_name] = float(metrics[metric_name])
                else:
                    logging.warning(f"Metric {metric_name} not found for model {model_name} algorithm {algo}")
                    metrics_json[model_name][algo][metric_name] = None
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved metrics to {filename}")

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