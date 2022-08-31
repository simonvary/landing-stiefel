import numpy as np

EVAL_FREQ = 2 ** 8
PATIENCE = 1000000
BATCH_SIZES = [100]

# STEP_SIZES = [0.01, 0.1, 1]
# RETRACTIONS = ['landing', 'exp', 'cayley']
STEP_SIZES = [.05, 0.1, .5]
RETRACTIONS = ['landing', 'exp']
# Get a random seed
MAX_SEED = 2 ** 32 - 1
# RANDOM_STATE = np.random.randint(MAX_SEED)
RANDOM_STATE = None
print(f"SEED = {RANDOM_STATE}")
