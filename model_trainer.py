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

class ModelTrainer:
    def __init__(self):
        self.best_models = {}
        self.models = {
            'lr': LogisticRegression(random_state=RANDOM_STATE),
            'rf': RandomForestClassifier(random_state=RANDOM_STATE),
            'mlp': MLPClassifier(random_state=RANDOM_STATE)
        }

    def train(self, algo, X_train, y_train):
        """Train model with GridSearchCV."""
        if algo not in self.models:
            raise ValueError(f"Unknown algorithm: {algo}")

        # Get parameters for the algorithm
        param_grid = MODEL_PARAMS[algo]
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            self.models[algo],
            param_grid,
            cv=CV_FOLDS,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Store the best model
        self.best_models[algo] = grid_search.best_estimator_
        
        # Get learning curve data
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            self.best_models[algo],
            X_train,
            y_train,
            train_sizes=train_sizes,
            cv=CV_FOLDS,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        return train_sizes_abs, train_scores, test_scores

    def eval(self, X_test, y_test, model_name, algo):
        """评估模型性能"""
        try:
            # 获取最佳模型
            best_model = self.models[algo].best_estimator_
            
            # 预测
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            
            # 保存评估结果
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
            
            # 可视化评估结果
            self.visualizer.plot_confusion_matrix(
                cm,
                title=f'{algo} Confusion Matrix',
                filename=f'plots/{model_name}/{algo}/confusion_matrix.png'
            )
            
            # 绘制学习曲线
            self.visualizer.plot_learning_curve(
                best_model,
                X_test,
                y_test,
                title=f'{algo} Learning Curve',
                filename=f'plots/{model_name}/{algo}/learning_curve.png'
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
                # Map prediction to label
                if pred in label_map:
                    predictions.append(pred)
                else:
                    # Default to neutral if prediction is unclear
                    predictions.append('neutral')
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                predictions.append('neutral')

        return predictions 