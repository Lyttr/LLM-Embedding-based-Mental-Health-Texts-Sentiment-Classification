from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from config import LOGISTIC_REGRESSION_PARAMS, MLP_PARAMS, RANDOM_FOREST_PARAMS

class ModelTrainer:
    def __init__(self):
        self.clfs = {
            'logistic': LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
            'mlp': MLPClassifier(**MLP_PARAMS),
            'rf': RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        }
    
    def train(self, name, X, y):
        """Train a specified model on the training data."""
        if name not in self.clfs:
            raise ValueError(f"Unsupported model type: {name}")
            
        try:
            clf = self.clfs[name]
            clf.fit(X, y)
            return clf
        except Exception as e:
            raise Exception(f"Model training failed: {str(e)}")
    
    def eval(self, clf, X, y, labels):
        """Evaluate model performance and return metrics."""
        try:
            y_pred = clf.predict(X)
            
            metrics = {
                'acc': accuracy_score(y, y_pred),
                'prec': precision_score(y, y_pred, average='weighted', zero_division=0),
                'rec': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
            
            if len(set(y)) == 2:
                metrics['auc'] = roc_auc_score(y, clf.predict_proba(X)[:, 1])
            
            report = classification_report(y, y_pred, target_names=labels.keys())
            cm = confusion_matrix(y, y_pred)
            
            return metrics, report, cm, y_pred
            
        except Exception as e:
            raise Exception(f"Model evaluation failed: {str(e)}") 