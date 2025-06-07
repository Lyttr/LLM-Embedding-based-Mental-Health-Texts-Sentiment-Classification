import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
from config import EMB_MODELS, TEST_SIZE, RANDOM_STATE, DATA_PATH

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DataProcessor:
    def __init__(self, model_name):
        """Initialize data processor with specified embedding model."""
        if model_name not in EMB_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        self.model_config = EMB_MODELS[model_name]
        self.model = SentenceTransformer(self.model_config['model_name'])
        self.label_map = {
            'positive': 0,
            'negative': 1,
            'neutral': 2
        }
        self.max_length = self.model_config.get('max_length', 128)
        self.scaler = StandardScaler()
    
    def load_data(self, filepath=None):
        """Load and prepare the dataset."""
        if filepath is None:
            filepath = DATA_PATH
            
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
                
            df = pd.read_csv(filepath)
            df = df.dropna(subset=['statement', 'status'])
            if 'statement' not in df.columns or 'status' not in df.columns:
                raise ValueError("Data file must contain 'statement' and 'status' columns")
                
            logging.info(f"Loaded {len(df)} samples from {filepath}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def prepare_labels(self, df):
        """Convert mental health status labels to numeric values."""
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

    def generate_embeddings(self, texts):
        """Generate embeddings for texts using the specified model."""
        if not texts:
            raise ValueError("No texts provided for embedding generation")
            
        embeddings_path = Path('models') / f'embeddings_{self.model_name}.npy'
        
        # Check if embeddings already exist
        if embeddings_path.exists():
            logging.info(f"Loading existing embeddings from {embeddings_path}")
            return np.load(embeddings_path)
        
        logging.info(f"Generating embeddings using {self.model_name}")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            max_length=self.max_length,
            convert_to_numpy=True
        )
        
        # Save embeddings
        embeddings_path.parent.mkdir(exist_ok=True)
        np.save(embeddings_path, embeddings)
        logging.info(f"Saved embeddings to {embeddings_path}")
        
        return embeddings

    def process_data(self):
        """Process data and generate embeddings."""
        # Load data
        df = self.load_data()
        
        # Add text length column
        df['text_length'] = df['statement'].str.len()
        
        # Prepare labels
        y = self.prepare_labels(df)
        
        # Generate embeddings
        X = self.generate_embeddings(df['statement'].tolist())
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Split data with stratification to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=y  # Ensure balanced class distribution in splits
        )
        
        return X_train, X_test, y_train, y_test, df 