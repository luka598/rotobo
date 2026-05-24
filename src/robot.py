import typing as T
import numpy as np
import matplotlib.pyplot as plt

from src.util import base_translation, rot_x

class Robot:
    def __init__(self, joint_locs: np.ndarray, joint_bases: np.ndarray) -> None:
        self.n_joints = len(joint_locs)
        self.n_links = self.n_joints - 1

        self.joint_locs = joint_locs.astype(np.float64)
        self.joint_bases = joint_bases.astype(np.float64)

        self.path_locs = []
        self.path_bases = []

        self.path_locs.append(self.joint_locs.copy())

        self._check_joints()
    
    def _check_joints(self):
        if not np.isfinite(self.joint_locs).all():
            raise RuntimeError("Joint locations are not initialised")

        if not np.isfinite(self.joint_bases).all():
            raise RuntimeError("Joint bases are not initialised")

        
    def plot(self, target: T.Optional[np.ndarray] = None, show: bool = False):
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.view_init(elev=20, azim=135)  # type: ignore
        bounds = (-5.0, 5.0)
        ax.set_xlim(*bounds)
        ax.set_ylim(*bounds)
        ax.set_zlim(*bounds)  # type: ignore

        # Path trace
        if len(self.path_locs) > 0:
            path = np.asarray(self.path_locs, dtype=np.float64)

            # max_pts = 100
            # steps_total = len(path)

            # if steps_total > max_pts:
            #     idx = np.linspace(0, steps_total - 1, max_pts).astype(int)
            #     path = path[idx]

            _, joints, _ = path.shape

            for j in range(joints):
                pts = path[:, j, :]
                ax.plot(
                    pts[:, 0], pts[:, 2], pts[:, 1],
                    alpha=0.5,
                )

        # Current skeleton
        ax.plot(self.joint_locs[:, 0], self.joint_locs[:, 2], self.joint_locs[:, 1], c="black")

        for idx in range(len(self.joint_locs)):
            if idx == 0:
                c_dot = "purple"
            elif idx == len(self.joint_locs) - 1:
                c_dot = "yellow"
            else:
                c_dot = "black"

            ax.scatter(self.joint_locs[idx, 0], self.joint_locs[idx, 2], self.joint_locs[idx, 1], c=c_dot)

        scale = 0.25
        colors = ["red", "green", "blue"]

        for i in range(len(self.joint_locs)):
            start = self.joint_locs[i]

            for axis_idx, color in enumerate(colors):
                direction = self.joint_bases[i, axis_idx]

                if np.isnan(direction).any():
                    continue

                direction = direction * scale

                ax.quiver(
                    start[0], start[2], start[1],
                    direction[0], direction[2], direction[1],
                    color=color,
                    linewidth=1
                )

        if target is not None:
            ax.scatter(target[0], target[2], target[1], c="red", marker="*", s=50)

        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()

        if show:
            plt.show()

    def print_info(self):
        print(f"Joints: {len(self.joint_locs)} / Links: {len(self.joint_locs) - 1}")

    def clone(self):
        r = Robot(self.joint_locs.copy(), self.joint_bases.copy())
        r.path_locs = self.path_locs.copy()
        return r

    # FWD Kinematics

    def _base_translation(self, joint_a: int, joint_b: int):
        return base_translation(self.joint_bases[joint_a], self.joint_bases[joint_b])

    def _translate_operation(self, op: np.ndarray, joint_a: int, joint_b: int):
        T = self._base_translation(joint_a, joint_b)
        return T @ op @ T.T

    def rotate_joint(self, joint: int, phi: float):
        T = base_translation(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), self.joint_bases[joint])
        op = T @ rot_x(phi) @ T.T
        self.joint_bases[joint] @= op
        for j in range(joint + 1, self.n_joints):
            self.joint_locs[j] = self.joint_locs[joint] + ((self.joint_locs[j] - self.joint_locs[joint]) @ op)
            self.joint_bases[j] @= op

        for j in range(joint, self.n_joints):
            self.joint_bases[j] = np.linalg.qr(self.joint_bases[j].T)[0].T

        self.path_locs.append(self.joint_locs.copy())
