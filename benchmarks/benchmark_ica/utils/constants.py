import numpy as np

EVAL_FREQ = 2 ** 2
PATIENCE = 100000
BATCH_SIZES = [100]

# STEP_SIZES = [0.01, 0.1, 1]
# RETRACTIONS = ['landing', 'exp', 'cayley']
STEP_SIZES = [0.05, 0.1, 0.5, 1]
RETRACTIONS = ['landing', 'exp']
# Get a random seed
MAX_SEED = 2 ** 32 - 1
# RANDOM_STATE = np.random.randint(MAX_SEED)
RANDOM_STATE = None
print(f"SEED = {RANDOM_STATE}")
