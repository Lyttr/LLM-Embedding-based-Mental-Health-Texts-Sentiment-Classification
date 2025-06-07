import logging
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from visualizer import Visualizer

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tfidf_baseline.log"),
        logging.StreamHandler()
    ]
)

def setup_directories():
    for name in ['plots/tfidf', 'results']:
        Path(name).mkdir(exist_ok=True)
        logging.info(f"Directory created or exists: {name}")

def save_metrics_to_json(metrics_dict, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved metrics to {filename}")

def load_data(data_path='data/Combined Data.csv', sample_size=100, random_state=42):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['statement', 'status'])
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
        logging.info(f"Sampled {sample_size} records from dataset")
    
    unique_labels = df['status'].unique()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    logging.info(f"Dataset size: {len(df)}")
    return df, label_map

def train_and_evaluate_tfidf(viz, df, label_map, test_size=0.2, random_state=42):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['statement'], 
        df['status'],
        test_size=test_size,
        random_state=random_state
    )
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Transform text data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    results = {}
    classifiers = {
        'svm': SVC(kernel='linear', probability=True),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
    }
    
    for name, clf in classifiers.items():
        logging.info(f"Training {name} classifier...")
        clf.fit(X_train_tfidf, y_train)
        predictions = clf.predict(X_test_tfidf)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, predictions)),
            'precision': float(precision_score(y_test, predictions, average='weighted')),
            'recall': float(recall_score(y_test, predictions, average='weighted')),
            'f1': float(f1_score(y_test, predictions, average='weighted'))
        }
        results[f"tfidf_{name}"] = metrics
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, predictions)
        viz.plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix - TF-IDF {name}",
            filename=f"plots/tfidf/confusion_matrix_{name}.png"
        )
    
    return results

def main():
    try:
        setup_directories()
        viz = Visualizer()
        
        df, label_map = load_data()
        tfidf_results = train_and_evaluate_tfidf(viz, df, label_map)
        
        results_path = 'results/tfidf_metrics.json'
        save_metrics_to_json(tfidf_results, results_path)
        logging.info(f"Saved TF-IDF metrics to {results_path}")
        
    except Exception as e:
        logging.error(f"TF-IDF baseline evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 