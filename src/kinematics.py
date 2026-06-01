import typing as T
import numpy as np

from src.dh_table import DHTable

def err_simple(target_loc):
    def inner(r: DHTable):
        pos_err = np.sum((target_loc - r.fwd()) ** 2)
        reg = 1e-3 * np.sum(r.q ** 2)
        val = pos_err + reg
        if not np.isfinite(val):
            raise RuntimeError("non finite")
        return val
    
    return inner

def err_constraints(target_loc, q_min, q_max):
    def inner(r: DHTable):
        pos_err = np.sum((target_loc - r.fwd()) ** 2)
        reg = 1e-3 * np.sum(r.q ** 2)
        
        upper_violation = np.maximum(0, r.q - q_max)
        lower_violation = np.maximum(0, q_min - r.q)
        
        limit_err = 1e4 * np.sum(upper_violation**2 + lower_violation**2)
        
        val = pos_err + reg + limit_err
        if not np.isfinite(val):
            raise RuntimeError("non finite")
        return val
    
    return inner

class OptimIK:
    @staticmethod
    def run(
        dh: DHTable,
        q_min: np.ndarray,
        q_max: np.ndarray,
        eps: float = 0.01,
        alpha: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-3,
        patience: int = 10,
        cb: T.Callable[[float, np.ndarray], T.Any] = lambda x, y: None,
        err: T.Callable[[DHTable], float] = err_simple(np.array([0, 0, 0]))
    ):
        best_err = float("inf")
        iters_since_best = 0

        # for _ in tqdm(range(max_iter)):
        for _ in range(max_iter):
            base_err = err(dh)

            if base_err < tol:
                break

            if base_err < best_err:
                best_err = base_err
                iters_since_best = 0
            else:
                iters_since_best += 1
                if iters_since_best >= patience:
                    break

            grad = np.zeros(dh.q.shape[0])
            for i in range(dh.q.shape[0]):
                r_plus = dh.clone()
                r_plus.q[i] += eps
                grad[i] = (err(r_plus) - base_err) / eps

            # norm = np.linalg.norm(grad)
            # if norm < 1e-12:
            #     break
            # grad /= norm

            dh.q -= alpha * grad
            dh.q = np.clip(dh.q, q_min, q_max)
            cb(base_err, dh.q)
