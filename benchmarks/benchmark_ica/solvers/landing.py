from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


# import pdb

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.linalg import expm
    # import sys
    # sys.path.append("../../")
    from landing_stiefel.variance_reduction import optimizer
    IcaOracle = import_ctx.import_from(
        'helper_ica', 'IcaOracle'
    )
    constants = import_ctx.import_from("constants")
    MinibatchSampler = import_ctx.import_from("minibatch_sampler", "MinibatchSampler")


def init_memory(
    x, memory, oracle, sampler
):
    n = sampler.n_batches
    _, n_features, _ = memory.shape
    avg_grad = np.zeros((n_features, n_features))
    for _ in range(n):
        this_slice, id = sampler.get_batch()
        idx, weight = id
        grad = oracle.grad(x, this_slice)
        memory[idx] = grad
        avg_grad += grad * weight
    memory[-1] = avg_grad
    return memory


def rule_algo(batch_size, use_vr, retraction):
    if use_vr:
        if batch_size != "full":
            return retraction == "landing"
        else:
            return False
    return True


class Solver(BaseSolver):
    name = 'Riemannian SGD' 

    stopping_criterion = SufficientProgressCriterion(patience=constants.PATIENCE,
                                                     strategy="callback")

    parameters = {
        "step_size": constants.STEP_SIZES,
        "batch_size": constants.BATCH_SIZES,
        "use_vr": [True, False],
        "retraction": constants.RETRACTIONS
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, X, mixing):
        self.X = X
        self.lbda = 1.
        self.oracle = IcaOracle(X)


    def skip(self, *args, **kwargs):
        return not rule_algo(self.batch_size, self.use_vr, self.retraction), None


    def run(self, callback):
        eval_freq = constants.EVAL_FREQ
        if self.batch_size == "full":
            eval_freq = 1
        rng = np.random.RandomState(constants.RANDOM_STATE)

        X = self.X
        
        n_samples, n_features = X.shape
        x = np.eye(n_features)
        if self.batch_size == "full":
            batch_size = n_samples
        else:
            batch_size = self.batch_size
        sampler = MinibatchSampler(
            n_samples, batch_size=batch_size
        )
        if self.use_vr:
            memory = np.zeros((sampler.n_batches + 1, n_features, n_features))
            memory = init_memory(x, memory, self.oracle, sampler)
            # for i in range(sampler.n_batches):
            #     memory[i] = self.oracle.grad()
        else:
            memory = np.empty((1, 1))
        while callback(x):
            x = optimizer(
                self.oracle,
                x,
                eval_freq,
                sampler,
                self.step_size,
                memory,
                self.lbda,
                saga=self.use_vr,
                retraction=self.retraction,
                seed=rng.randint(constants.MAX_SEED)
            )
            if np.isnan(x).any():
                raise ValueError()
        self.unmixing = x

    def get_result(self):
        return self.unmixing
