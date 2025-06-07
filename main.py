import logging
import os
import time
import numpy as np
import json
from pathlib import Path
from data_processor import DataProcessor
from model_trainer import ModelTrainer, LLMClassifier
from visualizer import Visualizer
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
    # Convert numpy types to Python native types
    metrics_json = {}
    for model_name, metrics in metrics_dict.items():
        metrics_json[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1'])
        }
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved metrics to {filename}")

def evaluate_embedding_models(processor, trainer, viz, X_train, X_test, y_train, y_test):
    """Train and evaluate traditional ML models on different sentence embeddings."""
    results, curves = {}, {}
    
    # Create model-specific plot directories
    model_plot_dir = processor.model_name
    os.makedirs(os.path.join(viz.save_dir, model_plot_dir), exist_ok=True)
    
    # Train and evaluate models
    for algo in ['lr', 'rf', 'mlp']:
        # Create algorithm-specific plot directories
        algo_plot_dir = os.path.join(model_plot_dir, algo)
        os.makedirs(os.path.join(viz.save_dir, algo_plot_dir), exist_ok=True)
        
        model_path = f'models/{processor.model_name}_{algo}.joblib'
        
        # Check if model already exists
        if os.path.exists(model_path):
            logging.info(f"Loading existing model {algo}")
            trainer.best_models[algo] = joblib.load(model_path)
            # Evaluate the loaded model
            metrics = trainer.eval(trainer.best_models[algo], X_test, y_test, processor.label_map)
            results[algo] = metrics
        else:
            logging.info(f"Training new model: {algo}")
            try:
                sizes, train_scores, test_scores = trainer.train(algo, X_train, y_train)
                curves[algo] = (sizes, train_scores, test_scores)

                metrics = trainer.eval(trainer.best_models[algo], X_test, y_test, processor.label_map)
                results[algo] = metrics

                # Save the trained model
                joblib.dump(trainer.best_models[algo], model_path)
                logging.info(f"Saved model {algo}")
                
                # Plot learning curve for newly trained models
                viz.plot_learning_curve(sizes, train_scores, test_scores,
                                    f"Learning Curve - {algo}",
                                    os.path.join(algo_plot_dir, 'learning_curve.png'))
            except Exception as e:
                logging.error(f"Training error ({algo}): {e}")
                continue

        # Generate visualizations regardless of whether model was loaded or trained
        viz.plot_cm(y_test, metrics['y_pred'], processor.label_map,
                    f"Confusion Matrix - {algo}",
                    os.path.join(algo_plot_dir, 'confusion_matrix.png'))

    # Plot overall metrics comparison
    try:
        viz.plot_metrics(results, filename=os.path.join(model_plot_dir, 'model_performance_comparison.png'))
        # Save metrics to JSON
        save_metrics_to_json(results, os.path.join('results', f'{processor.model_name}_metrics.json'))
    except Exception as e:
        logging.error(f"Metric plot error: {e}")

    return results

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

# === Main Pipeline ===
def main():
    start = time.time()
    logging.info("=== Sentiment Analysis Pipeline Start ===")

    try:
        setup_directories()

        # Initialize components
        trainer = ModelTrainer()
        viz = Visualizer(save_dir='plots')

        # Process data with each embedding model
        all_results = {}
        for model_name in EMB_MODELS.keys():
            logging.info(f"\nProcessing with {model_name} model...")
            
            # Initialize processor with current model
            processor = DataProcessor(model_name=model_name)
            
            # Data processing
            logging.info("Processing data...")
            X_train, X_test, y_train, y_test, df = processor.process_data()

            # Data visualizations (only for the first model)
            if model_name == list(EMB_MODELS.keys())[0]:
                # Create data visualization directory
                data_viz_dir = 'data_analysis'
                os.makedirs(os.path.join(viz.save_dir, data_viz_dir), exist_ok=True)

                try:
                    viz.plot_len_dist(df, len_col='text_length', 
                                    filename=os.path.join(data_viz_dir, 'text_length_distribution.png'))
                    viz.plot_cls_dist(df, 
                                    filename=os.path.join(data_viz_dir, 'class_distribution.png'))
                    viz.plot_len_by_cls(df, len_col='text_length', 
                                      filename=os.path.join(data_viz_dir, 'length_by_class.png'))
                    viz.plot_wc(" ".join(df['statement']), 
                               filename=os.path.join(data_viz_dir, 'wordcloud.png'))
                except Exception as e:
                    logging.error(f"Visualization error: {e}")

            # Train & evaluate models
            logging.info("Training ML models...")
            model_results = evaluate_embedding_models(processor, trainer, viz, X_train, X_test, y_train, y_test)
            all_results.update({f"{model_name}_{k}": v for k, v in model_results.items()})

        # Plot overall performance comparison
        try:
            viz.plot_metrics(all_results, filename='all_models_performance_comparison.png')
            # Save all metrics to JSON
            save_metrics_to_json(all_results, 'results/all_models_metrics.json')
        except Exception as e:
            logging.error(f"Final performance plot error: {e}")

        logging.info(f"Pipeline completed in {time.time() - start:.2f} seconds.")

    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()