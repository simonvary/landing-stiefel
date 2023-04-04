import numpy as np

EVAL_FREQ = 2 ** 4
PATIENCE = 1000000000
BATCH_SIZES = [100, "full"]
# BATCH_SIZES = [100]

# STEP_SIZES = [0.01, 0.1, 1]
# RETRACTIONS = ['landing', 'exp', 'cayley']
STEP_SIZES = [0.05, 0.1, .5, 1.0]
RETRACTIONS = ['landing', 'exp', 'proj']
# RETRACTIONS = ['landing']
# Get a random seed
MAX_SEED = 2 ** 32 - 1
RANDOM_STATE = np.random.randint(MAX_SEED)
# RANDOM_STATE = None
print(f"SEED = {RANDOM_STATE}")
