import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.robot import Robot
from src.kinematics import OptimIK

joints = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [2, 1, 0],
    [3, 1, 0],
    [4, 1, 0],
    [5, 1, 0],
])

bases = np.full((joints.shape[0], 3, 3), np.nan)
bases[:, 2] = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
])

bases[0, 0] = np.array([1, 0, 0])
bases[0, 1] = np.array([0, 1, 0])

for idx in range(1, len(joints)):
    ja, jb = joints[idx - 1], joints[idx]
    ba, bb = bases[idx - 1], bases[idx]

    # print(ja, jb, "|", ba, bb)

    z_prev = bases[idx - 1, 2]
    z_new = bases[idx, 2]

    prev_x = bases[idx - 1, 0]

    x = np.cross(z_prev, z_new)

    if np.linalg.norm(x) < 1e-8:
        x = prev_x
    else:
        x = x / np.linalg.norm(x)

    y = np.cross(z_new, x)
    y = y / np.linalg.norm(y)

    x = np.cross(y, z_new)
    x = x / np.linalg.norm(x)

    bases[idx, 0] = x
    bases[idx, 1] = y

dh_bases = np.zeros_like(bases)
dh_bases[0] = bases[0]

for i in range(1, len(joints)):
    z_prev = dh_bases[i - 1, 2]
    z_curr = bases[i, 2]

    z_prev = z_prev / np.linalg.norm(z_prev)
    z_curr = z_curr / np.linalg.norm(z_curr)

    x = np.cross(z_prev, z_curr)
    nx = np.linalg.norm(x)

    if nx < 1e-8:
        x = dh_bases[i - 1, 0]
    else:
        x = x / nx

    y = np.cross(z_curr, x)
    y = y / np.linalg.norm(y)

    x = np.cross(y, z_curr)
    x = x / np.linalg.norm(x)

    dh_bases[i, 0] = x
    dh_bases[i, 1] = y
    dh_bases[i, 2] = z_curr
# print("---")
# print(bases)
# print("---")
# print(dh_bases)
# print("---")
# plot_joints(joints, bases)

robot = Robot(joints, bases)
robot.print_info()
target = np.array([1, 1, 2])

prev_err = 10000.0

for i in tqdm(range(1000)):
    robot.plot(target=target, show=False)
    plt.savefig(f"ignore_img/{i:03d}.png")
    plt.close()

    err = OptimIK.run(robot, target, robot.joint_bases[-1], alpha=0.01, eps=0.01, max_iter=1)
    if np.abs(prev_err - err[0]) < 0.0001:
        break

    prev_err = err[0]
