import logging
import os
from data_processor import DataProcessor
from model_trainer import ModelTrainer, LLMClassifier
from visualizer import Visualizer
from config import EMB_MODELS, TEST_SIZE, RANDOM_STATE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main execution function for sentiment analysis pipeline."""
    # Initialize components
    processor = DataProcessor()
    trainer = ModelTrainer()
    viz = Visualizer()
    
    # Load and process data
    df = processor.load()
    processor.prep_labels()
    
    # Generate visualizations
    viz.plot_len_dist(df)
    viz.plot_cls_dist(df)
    viz.plot_len_by_cls(df)
    viz.plot_wc(" ".join(df['text']))
    
    # Train and evaluate traditional models
    results = {}
    learning_curves = {}
    
    for emb_name, emb_model in EMB_MODELS.items():
        logging.info(f"Processing embeddings with {emb_name}")
        processor.model = emb_model
        X_train, X_test, y_train, y_test = processor.split()
        
        for model_name in ['lr', 'rf', 'mlp', 'svm']:
            logging.info(f"Training {model_name} with {emb_name} embeddings")
            
            # Train model and get learning curve
            train_sizes, train_scores, test_scores = trainer.train(
                model_name, X_train, y_train
            )
            learning_curves[f"{emb_name}_{model_name}"] = (
                train_sizes, train_scores, test_scores
            )
            
            # Evaluate model
            metrics = trainer.eval(
                trainer.best_models[model_name],
                X_test, y_test,
                processor.labels
            )
            results[f"{emb_name}_{model_name}"] = metrics
            
            # Plot confusion matrix
            viz.plot_cm(
                y_test,
                metrics['y_pred'],
                processor.labels,
                f"Confusion Matrix - {emb_name} {model_name}"
            )
            
            # Plot ROC curves
            for label in processor.labels:
                viz.plot_roc_curve(
                    metrics['fpr'][label],
                    metrics['tpr'][label],
                    metrics['roc_auc'][label],
                    f"ROC Curve - {emb_name} {model_name} {label}"
                )
    
    # Plot learning curves
    for name, (train_sizes, train_scores, test_scores) in learning_curves.items():
        viz.plot_learning_curve(
            train_sizes, train_scores, test_scores,
            f"Learning Curve - {name}"
        )
    
    # Plot performance comparison
    viz.plot_metrics(results)
    
    # Evaluate LLM baseline
    if 'OPENAI_API_KEY' in os.environ:
        logging.info("Evaluating LLM baseline")
        llm = LLMClassifier(os.environ['OPENAI_API_KEY'])
        
        # Test different prompts
        for prompt_type in ['basic', 'detailed', 'few_shot']:
            logging.info(f"Testing LLM with {prompt_type} prompt")
            predictions = llm.predict(
                X_test,
                processor.labels,
                prompt_type=prompt_type
            )
            
            # Calculate metrics
            metrics = trainer.eval(
                predictions,
                X_test,
                y_test,
                processor.labels
            )
            results[f"llm_{prompt_type}"] = metrics
            
            # Plot confusion matrix
            viz.plot_cm(
                y_test,
                metrics['y_pred'],
                processor.labels,
                f"Confusion Matrix - LLM {prompt_type}"
            )
    else:
        logging.warning("Skipping LLM baseline evaluation - No API key found")

if __name__ == "__main__":
    main() 