import logging
import os
import time
from pathlib import Path
from data_processor import DataProcessor
from model_trainer import ModelTrainer, LLMClassifier
from visualizer import Visualizer
from config import EMB_MODELS

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

def evaluate_embedding_models(processor, trainer, viz, df, y):
    """Train and evaluate traditional ML models on different sentence embeddings."""
    results, curves = {}, {}

    for name, model in EMB_MODELS.items():
        logging.info(f"Embedding with: {name}")
        try:
            from sentence_transformers import SentenceTransformer
            processor.model = SentenceTransformer(model)
            X = processor.get_embeddings(df['statement'])
            X_train, X_test, y_train, y_test = processor.split_data(X, y)

            for algo in ['lr', 'rf', 'mlp', 'svm']:
                logging.info(f"Training model: {algo} on embeddings: {name}")
                try:
                    sizes, train_scores, test_scores = trainer.train(algo, X_train, y_train)
                    curves[f"{name}_{algo}"] = (sizes, train_scores, test_scores)

                    metrics = trainer.eval(trainer.best_models[algo], X_test, y_test, processor.label_map)
                    results[f"{name}_{algo}"] = metrics

                    viz.plot_cm(y_test, metrics['y_pred'], processor.label_map,
                                f"CM - {name} {algo}", f"confusion_matrix_{name}_{algo}.png")
                    for label in processor.label_map:
                        viz.plot_roc_curve(metrics['fpr'][label], metrics['tpr'][label],
                                           metrics['roc_auc'][label], f"ROC - {name} {algo} {label}",
                                           f"roc_curve_{name}_{algo}_{label}.png")
                except Exception as e:
                    logging.error(f"Training error ({name}/{algo}): {e}")

        except Exception as e:
            logging.error(f"Embedding error ({name}): {e}")

    for key, (sizes, tr_scores, te_scores) in curves.items():
        try:
            viz.plot_learning_curve(sizes, tr_scores, te_scores,
                                    f"Learning Curve - {key}", f"learning_curve_{key}.png")
        except Exception as e:
            logging.error(f"Learning curve error ({key}): {e}")

    try:
        viz.plot_metrics(results, filename='model_performance_comparison.png')
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
        processor = DataProcessor()
        trainer = ModelTrainer()
        viz = Visualizer(save_dir='plots')

        # Data processing
        logging.info("Processing data...")
        X_train, X_test, y_train, y_test, df = processor.process_data()

        # Data visualizations
        try:
            viz.plot_len_dist(df, len_col='text_length', filename='data_length_distribution.png')
            viz.plot_cls_dist(df, filename='data_class_distribution.png')
            viz.plot_len_by_cls(df, len_col='text_length', filename='data_length_by_class.png')
            viz.plot_wc(" ".join(df['statement']), filename='data_wordcloud.png')
        except Exception as e:
            logging.error(f"Visualization error: {e}")

        # Train & evaluate models
        logging.info("Training ML models...")
        model_results = evaluate_embedding_models(processor, trainer, viz, df, y_train)

        # Evaluate LLM
        logging.info("Evaluating LLM baseline...")
        llm_results = evaluate_llm(processor, trainer, viz, df, X_test.index, y_test)

        # Combined metrics visualization
        all_results = {**model_results, **llm_results}
        try:
            viz.plot_metrics(all_results, filename='all_models_performance_comparison.png')
        except Exception as e:
            logging.error(f"Final performance plot error: {e}")

        logging.info(f"Pipeline completed in {time.time() - start:.2f} seconds.")

    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()