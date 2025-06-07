import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from config import DATA_PATH, EMB_MODELS, TEST_SIZE, RANDOM_STATE
import logging

class DataProcessor:
    def __init__(self, model_name='minilm'):
        """Initialize data processor with specified model.
        
        Args:
            model_name (str): Name of the embedding model to use. Defaults to 'minilm'.
        """
        self.model = SentenceTransformer(EMB_MODELS[model_name])
        self.label_map = {}
        self.reverse_label_map = {}
    
    def load_data(self):
        """Load and preprocess the dataset.
        
        Returns:
            pd.DataFrame: Cleaned and preprocessed dataframe
        """
        try:
            # Load data
            df = pd.read_csv(DATA_PATH)
            logging.info(f"原始数据行数: {len(df)}")
            
            # Clean data
            df = df.dropna(subset=['statement', 'status'])
            logging.info(f"删除空值后的行数: {len(df)}")
            
            df['statement'] = df['statement'].astype(str)
            
            
            
            return df
        except FileNotFoundError:
            raise Exception(f"Data file not found: {DATA_PATH}")
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")
    
    def prepare_labels(self, df):
        """Prepare labels for classification.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'status' column
            
        Returns:
            np.ndarray: Numeric labels
        """
        try:
            # Create label mappings
            unique_labels = sorted(df['status'].unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
            
            # Convert labels to numeric values
            return df['status'].map(self.label_map).values
        except Exception as e:
            raise Exception(f"Failed to prepare labels: {str(e)}")
    
    def get_embeddings(self, texts):
        """Generate embeddings for input texts.
        
        Args:
            texts (Union[pd.Series, list]): Input texts to encode
            
        Returns:
            np.ndarray: Text embeddings
        """
        try:
            # Convert to list and ensure string type
            if isinstance(texts, pd.Series):
                texts = texts.tolist()
            texts = [str(text) for text in texts]
            
            # Generate embeddings
            return self.model.encode(texts)
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def split_data(self, X, y):
        """Split data into train and test sets.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            return train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )
        except Exception as e:
            raise Exception(f"Failed to split data: {str(e)}")
    
    def process_data(self):
        """Complete data processing pipeline.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, df)
        """
        # Load and preprocess data
        df = self.load_data()
        
        # Prepare labels
        y = self.prepare_labels(df)
        
        # Generate embeddings
        X = self.get_embeddings(df['statement'])
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        return X_train, X_test, y_train, y_test, df 