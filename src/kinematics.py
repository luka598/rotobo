import numpy as np
from src.robot import Robot

class OptimIK:
    @staticmethod
    def run(robot: Robot, target_loc, target_base, eps=0.01, alpha=0.001, max_iter=100):
        err_path = []

        def err(r: Robot):
            return np.sum((target_loc - r.joint_locs[-1]) ** 2)

        for _ in range(max_iter):
            base_err = err(robot)
            err_path.append(base_err)
            if base_err < 1e-6:
                break

            grad = np.zeros(robot.n_joints - 1)
            for i in range(robot.n_joints - 1):
                r_plus = robot.clone()
                r_plus.rotate_joint(i, eps)
                grad[i] = (err(r_plus) - base_err) / eps

            rotations = -alpha * grad

            for i in range(robot.n_joints - 1):
                robot.rotate_joint(i, rotations[i])
            
        return err_path