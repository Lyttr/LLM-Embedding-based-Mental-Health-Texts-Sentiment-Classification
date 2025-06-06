import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from config import DATA_PATH, EMB_MODELS, TEST_SIZE, RANDOM_STATE

class DataProcessor:
    def __init__(self):
        """Initialize data processor."""
        self.model = SentenceTransformer(EMB_MODELS['minilm'])
        self.labels = {}
        self.idx2label = {}
    
    def load(self):
        """Load and clean data from CSV file."""
        try:
            df = pd.read_csv(DATA_PATH)
            df = df.dropna(subset=['statement', 'status'])
            df['statement'] = df['statement'].astype(str)
            return df
        except FileNotFoundError:
            raise Exception(f"Data file not found: {DATA_PATH}")
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")
    
    def prep_labels(self, df):
        """Prepare labels for classification."""
        try:
            unique_labels = df['status'].unique()
            self.labels = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx2label = {idx: label for label, idx in self.labels.items()}
            return df['status'].map(self.labels)
        except Exception as e:
            raise Exception(f"Failed to prepare labels: {str(e)}")
    
    def get_emb(self, texts):
        """Get embeddings for texts."""
        try:
            return self.model.encode(texts)
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def split(self, X, y):
        """Split data into train and test sets."""
        try:
            return train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )
        except Exception as e:
            raise Exception(f"Failed to split data: {str(e)}")
    
    def add_len(self, df):
        """Add text length feature."""
        try:
            df['len'] = df['statement'].str.len()
            return df
        except Exception as e:
            raise Exception(f"Failed to add length feature: {str(e)}") 