import numpy as np


def rot_x(theta):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )

def base_translation(base_a: np.ndarray, base_b: np.ndarray):
    # A -> B
    return base_b.T @ base_a
