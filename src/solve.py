import numpy as np
from src.util import gen_trajectory
from src.dh_table import DHTable
from src.kinematics import OptimIK, err_constraints

def solve(dh: DHTable, targets: np.ndarray, q_min: np.ndarray, q_max: np.ndarray, qd_max: np.ndarray, qdd_max: np.ndarray, n_points: int = 100):
    t_total = []
    q_total = []
    qd_total = []
    time_offset = 0.0

    for i, target in enumerate(targets):
        q0 = dh.q.copy()
        OptimIK.run(dh, q_min, q_max, alpha=0.01, max_iter=10000, err=err_constraints(target, q_min, q_max))
        q1 = dh.q.copy()

        t_seg, q_seg, qd_seg = gen_trajectory(q0, q1, qd_max, qdd_max, q_min, q_max, n_points)

        t_total.append(t_seg + time_offset)
        q_total.append(q_seg)
        qd_total.append(qd_seg)

        time_offset += t_seg[-1]

    t_composite = np.concatenate(t_total)
    q_composite = np.concatenate(q_total, axis=0)
    qd_composite = np.concatenate(qd_total, axis=0)

    return t_composite, q_composite, qd_composite