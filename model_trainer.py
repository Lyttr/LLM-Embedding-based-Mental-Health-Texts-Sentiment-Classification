import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, roc_curve, auc
)
from config import (
    LR_PARAMS, RF_PARAMS, MLP_PARAMS, SVM_PARAMS,
    CV_FOLDS, CV_SCORING, LLM_MODEL, LLM_TEMP,
    LLM_MAX_TOKENS, LLM_PROMPTS
)
import openai
import time

class ModelTrainer:
    def __init__(self):
        self.clfs = {
            'lr': LogisticRegression(),
            'rf': RandomForestClassifier(),
            'mlp': MLPClassifier(),
            'svm': SVC(probability=True)
        }
        self.best_models = {}
        self.metrics = {}
    
    def train(self, name, X, y):
        """Train model with hyperparameter tuning."""
        if name == 'lr':
            params = LR_PARAMS
        elif name == 'rf':
            params = RF_PARAMS
        elif name == 'mlp':
            params = MLP_PARAMS
        elif name == 'svm':
            params = SVM_PARAMS
        else:
            raise ValueError(f"Unknown model: {name}")
        
        grid = GridSearchCV(
            self.clfs[name],
            params,
            cv=CV_FOLDS,
            scoring=CV_SCORING,
            refit='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X, y)
        self.best_models[name] = grid.best_estimator_
        
        # Get learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            grid.best_estimator_, X, y,
            cv=CV_FOLDS,
            scoring='f1_weighted',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        return train_sizes, train_scores, test_scores
    
    def eval(self, clf, X, y, labels):
        """Evaluate model performance."""
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted'
        )
        
        # Calculate ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y == i).astype(int),
                y_prob[:, i]
            )
            roc_auc[label] = auc(fpr[label], tpr[label])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

class LLMClassifier:
    def __init__(self, api_key):
        """Initialize LLM classifier."""
        openai.api_key = api_key
        self.model = LLM_MODEL
        self.temp = LLM_TEMP
        self.max_tokens = LLM_MAX_TOKENS
        self.prompts = LLM_PROMPTS
    
    def predict(self, texts, categories, prompt_type='basic'):
        """Make predictions using LLM."""
        predictions = []
        prompt = self.prompts[prompt_type]
        
        for text in texts:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{
                        'role': 'user',
                        'content': prompt.format(
                            categories=', '.join(categories),
                            text=text
                        )
                    }],
                    temperature=self.temp,
                    max_tokens=self.max_tokens
                )
                
                pred = response.choices[0].message.content.strip().lower()
                predictions.append(pred)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                predictions.append(None)
        
        return predictions 