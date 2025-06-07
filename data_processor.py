import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
from config import EMB_MODELS, TEST_SIZE, RANDOM_STATE, DATA_PATH, EMBEDDINGS_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DataProcessor:
    def __init__(self, model_name):
        if model_name not in EMB_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        self.data_path = DATA_PATH
        self.embeddings_path = Path(EMBEDDINGS_DIR) / f'embeddings_{model_name}.npy'
        self.model_config = EMB_MODELS[model_name]
        self.model = SentenceTransformer(self.model_config['model_name'])
        self.label_map = None  # Will be created from actual data
        self.max_length = self.model_config.get('max_length', 128)
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load and prepare the dataset."""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
                
            df = pd.read_csv(self.data_path)
            df = df.dropna(subset=['statement', 'status'])
            
            if 'statement' not in df.columns or 'status' not in df.columns:
                raise ValueError("Data file must contain 'statement' and 'status' columns")
                
            logging.info(f"Loaded {len(df)} samples from {self.data_path}")
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def prepare_labels(self, df):
        """Convert mental health status labels to numeric values."""
        try:
            if 'status' not in df.columns:
                raise ValueError("DataFrame must contain 'status' column")
                
            # Get unique mental health statuses
            statuses = sorted(df['status'].unique())
            
            # Log status distribution
            status_counts = df['status'].value_counts()
            logging.info("Mental health status distribution in dataset:")
            for status, count in status_counts.items():
                logging.info(f"{status}: {count} samples")
                
            # Create label map for mental health statuses
            self.label_map = {status: i for i, status in enumerate(statuses)}
            logging.info(f"Label mapping: {self.label_map}")
            
            return df['status'].map(self.label_map)
            
        except Exception as e:
            logging.error(f"Label preparation error: {str(e)}")
            raise

    def generate_embeddings(self, texts):
        """Generate embeddings for texts using the specified model."""
        try:
            if not texts:
                raise ValueError("No texts provided for embedding generation")
            if self.embeddings_path.exists():
                logging.info(f"Loading existing embeddings from {self.embeddings_path}")
                return np.load(self.embeddings_path)
            
            logging.info(f"Generating embeddings using {self.model_name}")
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                max_length=self.max_length,
                convert_to_numpy=True
            )
            self.embeddings_path.parent.mkdir(exist_ok=True)
            np.save(self.embeddings_path, embeddings)
            logging.info(f"Saved embeddings to {self.embeddings_path}")
            
            return embeddings
            
        except Exception as e:
            logging.error(f"Embedding generation error: {str(e)}")
            raise

    def process_data(self):
        try:
            df = self.load_data()
            y = self.prepare_labels(df)
            X = self.generate_embeddings(df['statement'].tolist())
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            return X_train, X_test, y_train, y_test, df
            
        except Exception as e:
            logging.error(f"Data processing error: {str(e)}")
            raise 