from pathlib import Path

# Data paths
DATA_PATH = Path('Combined Data.csv')

# Model params
EMB_MODELS = {
    'minilm': 'all-MiniLM-L6-v2',
    'mpnet': 'all-mpnet-base-v2',
    'distilbert': 'all-distilroberta-v1',
    'use': 'all-MiniLM-L12-v2'
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

# LLM params
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMP = 0.0
LLM_MAX_TOKENS = 10
LLM_PROMPTS = {
    'basic': """Classify the following text into one of these categories: {categories}
Text: {text}
Category:""",
    
    'detailed': """Analyze the sentiment of the following text and classify it into one of these categories: {categories}
Consider the emotional tone, context, and overall sentiment.
Text: {text}
Category:""",
    
    'few_shot': """Here are some examples of text classification:
Text: "I feel so happy and excited about my future!"
Category: positive

Text: "I'm really struggling with my mental health lately."
Category: negative

Now classify this text into one of these categories: {categories}
Text: {text}
Category:"""
}

# Classifier params
LR_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear']
}

MLP_PARAMS = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': 200,
    'random_state': RANDOM_STATE,
    'early_stopping': True,
    'verbose': True,
    'n_jobs': -1
}

RF_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': 1
}

SVM_PARAMS = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'random_state': RANDOM_STATE
}

# Cross-validation
CV_FOLDS = 5
CV_SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

# Viz params
FIG_SIZE = (6, 5)
WC_SIZE = (800, 400)
PLOT_STYLE = 'seaborn'
COLOR_PALETTE = 'husl' 