import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.robot import Robot
from src.dh_table import DHTable
from src.solve import solve
from src.util import gen_gif

np.set_printoptions(suppress=True, formatter={'float_kind': '{:.10f}'.format})

# ==================================
# Joint locations
# ==================================

joint_locs = np.array([
    [0.000, 0.000, 0.630],
    [0.600, 0.000, 0.630],
    [0.600, 0.000, 1.910],
    [0.600, 0.000, 2.110],
    [2.192, 0.000, 2.110],
    [2.392, 0.000, 2.110],
], dtype=float)

joint_locs -= joint_locs[0] # Move base to (0, 0, 0)

# ==================================
# Joint bases
# ==================================

joint_bases = np.full((6, 3, 3), np.nan, dtype=float)

joint_bases[:, 2] = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=float)

joint_bases[0, 0] = np.array([1.0, 0.0, 0.0])
joint_bases[0, 1] = np.array([0.0, 1.0, 0.0])

for idx in range(1, len(joint_locs)):
    z_prev = joint_bases[idx - 1, 2]
    z_new = joint_bases[idx, 2]
    prev_x = joint_bases[idx - 1, 0]

    x = np.cross(z_prev, z_new)
    if np.linalg.norm(x) < 1e-8:
        x = prev_x
    else:
        x = x / np.linalg.norm(x)

    y = np.cross(z_new, x)
    y = y / np.linalg.norm(y)

    x = np.cross(y, z_new)
    x = x / np.linalg.norm(x)

    joint_bases[idx, 0] = x
    joint_bases[idx, 1] = y

# ==================================
# Angle constraints
# ==================================

q_min = np.deg2rad([
    -180,
    -40,
    -180,
    -300,
    -120,
    -360,
]).astype(float)

q_max = np.deg2rad([
    180,
    160,
    70,
    300,
    120,
    360,
]).astype(float)

# ==================================
# Speed constraints
# ==================================

qd_max = np.deg2rad([
    110,
    90,
    90,
    150,
    120,
    235,
]).astype(float)

qdd_max = np.deg2rad([
    250.0,
    250.0,
    250.0,
    500.0,
    500.0,
    1000.0,
])

# ==================================
# DH
# ==================================

dh = DHTable.from_kin(joint_locs, joint_bases[:, 2])
print(dh)

# ==================================
# Solve
# ==================================

t_arr, q_t, qd_t = solve(dh, np.array([
    [1, 2, 1],
    [2, 2, 1],
    [1, 2, 1],
]), q_min, q_max, qd_max, qdd_max)

# ==================================
# Plot position and accel
# ==================================

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

for i in range(q_t.shape[1]):
    axs[0].plot(t_arr, np.degrees(q_t[:, i]), label=f"q{i}")

axs[0].set_ylabel("Rotacija (°)")
axs[0].legend()

for i in range(qd_t.shape[1]):
    axs[1].plot(t_arr, np.degrees(qd_t[:, i]), label=f"q{i}")

axs[1].set_ylabel("Brzina (°)")
axs[1].set_xlabel("Vrijeme")
axs[1].legend()

fig.suptitle("Stanja zglobova kroz vrijeme")
fig.tight_layout()
fig.savefig("pos_accel.png", dpi=300)

# ==================================
# Plot pose
# ==================================

r = Robot(joint_locs, joint_bases)

last_q = np.zeros_like(dh.q) 

frames = []

for i in tqdm(range(len(t_arr))):
    deltas = q_t[i] - last_q
    
    for idx, delta in enumerate(deltas):
        r.rotate_joint(idx, delta)
    last_q = q_t[i].copy()

    img = r.plot(target=None, show=False)
    frames.append(img)

print(r.joint_locs[-1])
print(dh)
print(f"Ukupno vrijeme: {t_arr[-1]:.2f} s")

gen_gif(frames, "robot.gif")