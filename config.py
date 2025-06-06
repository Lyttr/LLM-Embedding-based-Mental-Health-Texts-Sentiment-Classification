from pathlib import Path

# Data paths
DATA_PATH = Path('Combined Data.csv')

# Model params
EMB_MODEL = 'all-MiniLM-L6-v2'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Classifier params
LR_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

MLP_PARAMS = {
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 200,
    'random_state': RANDOM_STATE,
    'early_stopping': True,
    'verbose': True,
    'n_jobs': -1
}

RF_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': 1
}

# Viz params
FIG_SIZE = (6, 5)
WC_SIZE = (800, 400) 