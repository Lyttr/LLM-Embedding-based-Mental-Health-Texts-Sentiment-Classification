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
from openai import OpenAI

class ModelTrainer:

    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.best_models = {}
        self.models = {
            'lr': LogisticRegression(
                max_iter=200,
                n_jobs=1,
                solver='liblinear',
                C=1.0
            ),
            'rf': RandomForestClassifier(
                n_estimators=30,
                max_depth=8,
                n_jobs=1,
                min_samples_split=10,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42,
                early_stopping=True,
                verbose=True
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
        self.client = OpenAI(api_key=api_key)
        self.valid_labels = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
        
    def predict(self, texts, label_map, prompt_type='basic'):
        predictions = []
        
        for text in texts:
            try:
            
                prompt_template = LLM_PROMPTS.get(prompt_type, LLM_PROMPTS['basic'])
                prompt = prompt_template.format(text=text)
                
                messages = [
                    {"role": "system", "content": "You are a mental health text classifier. Your task is to classify the given text into one of the following categories: Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, or Suicidal. Respond with ONLY the category name, nothing else."},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=LLM_TEMP,
                    max_tokens=LLM_MAX_TOKENS
                )
                
                pred = response.choices[0].message.content.strip()
                
                if pred in self.valid_labels:
                    predictions.append(pred)
                else:
                    logging.warning(f"Invalid prediction: {pred}, using default label")
                    predictions.append('Normal')  
                    
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                predictions.append('Normal')  

        return predictions 