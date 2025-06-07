from pathlib import Path

# Data paths
DATA_PATH = Path('data/Combined Data.csv')
EMBEDDINGS_DIR = 'embeddings'

# Model params
EMB_MODELS = {
    'all-MiniLM-L6-v2': {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'max_length': 128
    },
    'all-mpnet-base-v2': {
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'max_length': 128
    },
    'all-distilroberta-v1': {
        'model_name': 'sentence-transformers/all-distilroberta-v1',
        'max_length': 128
    },
    'all-MiniLM-L12-v2': {
        'model_name': 'sentence-transformers/all-MiniLM-L12-v2',
        'max_length': 128
    }
}

# === Model Configurations ===
MODEL_PARAMS = {
    'lr': {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [1000],
        'n_jobs': [-1]
    },
    'rf': {
        'n_estimators': [50],
        'max_depth': [10],
        'min_samples_split': [5],
        'n_jobs': [-1]
    },
    'mlp': {
        'hidden_layer_sizes': [(64,)],
        'max_iter': [1000],
        'early_stopping': [True],
        'solver': ['adam'],
        'learning_rate_init': [0.001]
    }
}

# === Training Configurations ===
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3

# === LLM Configurations ===
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMP = 0.3
LLM_MAX_TOKENS = 10


LLM_PROMPTS = {
 
    'basic': """Analyze this text and classify it into one of these mental health categories:
- Anxiety
- Bipolar
- Depression
- Normal
- Personality disorder
- Stress
- Suicidal

Text: {text}
Category:""",
    
  
    'detailed': """Analyze this text in detail for mental health classification. Consider:
- Emotional state and intensity
- Behavioral patterns
- Thought processes
- Risk factors
- Coping mechanisms

Classify into one of these categories:
- Anxiety
- Bipolar
- Depression
- Normal
- Personality disorder
- Stress
- Suicidal

Text: {text}
Category:""",
    
    
    'few_shot': """Here are some examples of mental health text classification:

Text: "I can't stop worrying about everything. My heart races and I feel like I can't breathe."
Category: Anxiety

Text: "I feel so low and empty. Nothing brings me joy anymore, and I can't get out of bed."
Category: Depression

Text: "I'm having a great day! Everything is wonderful and I feel like I can accomplish anything!"
Category: Normal

Text: "I'm having extreme mood swings. One moment I'm full of energy and ideas, the next I'm completely exhausted."
Category: Bipolar

Text: "I can't take it anymore. Life is too painful and I don't want to continue."
Category: Suicidal

Now classify this text: {text}
Category:"""
}

FIG_SIZE = (6, 5)
WC_SIZE = (800, 400)
PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = 'husl' 