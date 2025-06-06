import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from config import DATA_PATH, EMBEDDING_MODEL, TEST_SIZE, RANDOM_STATE

class DataProcessor:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.labels = None
        self.idx2label = None
        
    def load(self):
        """Load and clean the dataset."""
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
        """Convert text labels to numeric indices."""
        uniq_labels = df['status'].unique()
        self.labels = {label: i for i, label in enumerate(uniq_labels)}
        self.idx2label = {i: label for label, i in self.labels.items()}
        return df['status'].map(self.labels).values
    
    def get_emb(self, texts):
        """Generate embeddings for input texts."""
        try:
            return self.model.encode(texts)
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def split(self, X, y):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    def add_len(self, df):
        """Add text length as a feature."""
        df['len'] = df['statement'].str.split().str.len()
        return df 