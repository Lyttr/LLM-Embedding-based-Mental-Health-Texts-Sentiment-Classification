import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
import openai
import logging
from config import (
    MODEL_PARAMS,
    CV_FOLDS,
    RANDOM_STATE,
    LLM_MODEL,
    LLM_TEMP,
    LLM_MAX_TOKENS,
    LLM_PROMPTS
)
import joblib
import os

class ModelTrainer:

    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.best_models = {}

        self.models = {
            'lr': LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                C=1.0,
                n_jobs=-1
            ),
            'rf': RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1
            ),
            'mlp': MLPClassifier(
                random_state=RANDOM_STATE,
                hidden_layer_sizes=(100,),
                max_iter=1000,
                early_stopping=True,
                learning_rate_init=0.001,
                solver='adam'
            )
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, model_name):
        try:
            algo_metrics = {}

            for algo in ['lr', 'rf', 'mlp']:
                try:
                    algo_plot_dir = os.path.join('plots', model_name, algo)
                    os.makedirs(algo_plot_dir, exist_ok=True)
                    
                    model = self.models[algo]
                    model.fit(X_train, y_train)
                    self.best_models[algo] = model
                    model_path = f'models/{model_name}_{algo}.joblib'
                    joblib.dump(model, model_path)
                    logging.info(f"Saved model {algo}")

                    metrics = self._evaluate_model(X_test, y_test, model_name, algo)
                    algo_metrics[algo] = metrics
                    

                    self.visualizer.plot_learning_curve(
                        model,
                        X_train,
                        y_train,
                        title=f'{algo} Learning Curve',
                        filename=os.path.join(algo_plot_dir, 'learning_curve.png')
                    )
 
                    if algo == 'rf':
                        self.visualizer.plot_feature_importance(
                            range(X_train.shape[1]),
                            model.feature_importances_,
                            title=f'{algo} Feature Importance',
                            filename=os.path.join(algo_plot_dir, 'feature_importance.png')
                        )
                    
                except Exception as e:
                    logging.error(f"Error in {algo} for {model_name}: {str(e)}")
                    continue

            self.visualizer.plot_metrics(
                {model_name: algo_metrics},
                title=f'{model_name} Model Performance Comparison',
                filename=os.path.join('plots', model_name, 'model_performance_comparison.png')
            )
            
            return algo_metrics
            
        except Exception as e:
            logging.error(f"Model evaluation error: {str(e)}")
            raise

    def _evaluate_model(self, X_test, y_test, model_name, algo):
        """Evaluate model and return metrics dictionary."""
        try:
            model = self.best_models[algo]
            y_pred = model.predict(X_test)
            
      
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
           
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
 
            algo_plot_dir = os.path.join('plots', model_name, algo)
            os.makedirs(algo_plot_dir, exist_ok=True)
            
            self.visualizer.plot_cm(
                y_test,
                y_pred,
                labels=['negative', 'positive'],
                title=f'{algo} Confusion Matrix',
                filename=os.path.join(algo_plot_dir, 'confusion_matrix.png')
            )
            
            return metrics
            
        except Exception as e:
            logging.error(f"Model evaluation error: {str(e)}")
            raise

class LLMClassifier:
    def __init__(self, api_key):
        """Initialize OpenAI API."""
        openai.api_key = api_key
        self.model = LLM_MODEL
        self.temperature = LLM_TEMP
        self.max_tokens = LLM_MAX_TOKENS
        self.prompts = LLM_PROMPTS

    def predict(self, texts, label_map, prompt_type='basic'):
        """Get predictions from GPT-3.5."""
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        predictions = []
        for text in texts:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a sentiment analysis expert."},
                        {"role": "user", "content": self.prompts[prompt_type].format(text=text)}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                pred = response.choices[0].message.content.strip().lower()
                if pred in label_map:
                    predictions.append(pred)
                else:
                    predictions.append('neutral')
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                predictions.append('neutral')

        return predictions 