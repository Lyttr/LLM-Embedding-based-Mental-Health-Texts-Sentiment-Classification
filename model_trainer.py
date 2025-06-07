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

                    algo_plot_dir = f'plots/{model_name}/{algo}'
                    os.makedirs(algo_plot_dir, exist_ok=True)

                    model = self.models[algo]
                    model.fit(X_train, y_train)
                    self.best_models[algo] = model
                    model_path = f'models/{model_name}_{algo}.joblib'
                    joblib.dump(model, model_path)
                    logging.info(f"Saved model {algo}")
                    metrics = self._evaluate_model(
                        X_test, y_test, model_name, algo
                    )
                    algo_metrics[algo] = metrics
                    
                except Exception as e:
                    logging.error(f"Error in {algo} for {model_name}: {str(e)}")
                    continue
            
            return algo_metrics
                    
        except Exception as e:
            logging.error(f"Model training and evaluation error: {str(e)}")
            raise

    def _evaluate_model(self, X_test, y_test, model_name, algo):

        try:

            if algo not in self.best_models:
                raise ValueError(f"No trained model found for {algo}")
                
            best_model = self.best_models[algo]
            
            y_pred = best_model.predict(X_test)
            y_test = np.array(y_test)
            y_pred = np.array(y_pred)
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1': float(f1_score(y_test, y_pred, average='weighted'))
            }
            cm = confusion_matrix(y_test, y_pred)
            self.visualizer.plot_confusion_matrix(
                cm,
                title=f'{algo} Confusion Matrix',
                filename=f'plots/{model_name}/{algo}/confusion_matrix.png'
            )
            self.visualizer.plot_learning_curve(
                best_model,
                X_test,
                y_test,
                title=f'{algo} Learning Curve',
                filename=f'plots/{model_name}/{algo}/learning_curve.png'
            )
            
            self.visualizer.plot_metrics(
                metrics,
                title=f'{model_name} - {algo} Performance',
                filename=f'plots/{model_name}/{algo}/model_performance_comparison.png'
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