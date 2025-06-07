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

# LLM prompt templates for different classification strategies
LLM_PROMPTS = {
    # Basic prompt for simple classification
    'basic': "Classify the sentiment of this text as positive, negative, or neutral: {text}",
    
    # Detailed prompt for more nuanced analysis
    'detailed': """Analyze the sentiment of this text in detail. 
Consider the emotional tone, context, and implications. 
Classify as positive, negative, or neutral: {text}""",
    
    # Few-shot prompt with examples
    'few_shot': """Here are some examples of sentiment classification:

Text: "I feel happy and content today."
Sentiment: positive

Text: "This is terrible and I hate it."
Sentiment: negative

Text: "The weather is cloudy."
Sentiment: neutral

Now classify this text: {text}"""
}

FIG_SIZE = (6, 5)
WC_SIZE = (800, 400)
PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = 'husl' 