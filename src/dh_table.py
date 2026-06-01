import numpy as np
from src.util import generate_dh_table, solve_fwd

class DHTable:
    def __init__(self, dh: list) -> None:
        self.dh = dh
        self.q = np.zeros(len(dh))

    @staticmethod
    def from_kin(joint_locs, joint_axes):
        dh = generate_dh_table(joint_locs, joint_axes)

        return DHTable(dh)
    
    @property
    def dh_rot(self):
        return [[x[0] + self.q[idx]] + list(x[1:]) for idx, x in enumerate(self.dh)]

    def fwd(self):
        _, position = solve_fwd(self.dh_rot)
        position = np.where(np.abs(position) < 1e-15, 0.0, position)
        return position
    
    def clone(self):
        x = DHTable(self.dh.copy())
        x.q = self.q.copy()
        return x

    def __repr__(self) -> str:
        s = ""
        s += "--- DH Table ---\n"
        s += f"{'Link':<6} | {'Theta (deg)':<12} | {'d':<6} | {'a':<8} | {'Alpha (deg)'}\n"
        s += "-" * 55 + "\n"
        for idx, (theta, d, a, alpha) in enumerate(self.dh_rot, 1):
            s += f"{idx:<6} | {np.rad2deg(theta):<12.1f} | {d:<6.1f} | {a:<8.3f} | {np.rad2deg(alpha):.1f}\n"
        s += "--- Forward Kinematics Result ---\n"
        
        s += f"Head: {self.fwd()}\n"

        return s

