from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizer import Visualizer
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function for sentiment analysis pipeline."""
    try:
        # Initialize components
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        visualizer = Visualizer()
        
        # Load and process data
        logger.info("Loading data...")
        df = data_processor.load_data()
        texts = df['statement'].tolist()
        y = data_processor.prepare_labels(df)
        X = data_processor.get_embeddings(texts)
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
        
        # Add text length feature
        df = data_processor.add_text_length(df)
        
        # Data visualization
        logger.info("Generating visualizations...")
        visualizer.plot_text_length_distribution(df)
        visualizer.plot_class_distribution(df)
        visualizer.plot_text_length_by_class(df)
        visualizer.plot_wordcloud(texts)
        
        # Train and evaluate models
        models = ['logistic', 'mlp', 'random_forest']
        for model_name in models:
            logger.info(f"Training {model_name} model...")
            model = model_trainer.train_model(model_name, X_train, y_train)
            metrics, report, cm, _ = model_trainer.evaluate_model(
                model, X_test, y_test, data_processor.label_dict
            )
            
            # Output evaluation results
            logger.info(f"\n{model_name} model evaluation results:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            if 'auc' in metrics:
                logger.info(f"ROC AUC: {metrics['auc']:.4f}")
            
            logger.info("\nClassification Report:")
            logger.info(report)
            
            # Plot confusion matrix
            visualizer.plot_confusion_matrix(
                cm, 
                data_processor.label_dict,
                f'{model_name} Confusion Matrix'
            )
            
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 