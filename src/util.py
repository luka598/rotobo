import numpy as np
from PIL import Image
import io

def generate_dh_table(locs, axes):
    num_joints = len(locs)
    if num_joints < 2:
        return []

    # --- dodatni frame za alat
    
    pos_list = list(locs)
    ax_list = list(axes)
    
    z_last_joint = ax_list[-2]
    z_tool = ax_list[-1]
    cos_alpha_last = np.dot(z_last_joint, z_tool)
    
    # okomiti
    if np.abs(cos_alpha_last) < 0.999:
        D_last = pos_list[-1] - pos_list[-2]
        s = (np.dot(D_last, z_last_joint) * cos_alpha_last - np.dot(D_last, z_tool)) / (1.0 - cos_alpha_last**2)
        intersection_point = pos_list[-1] + s * z_tool
        
        if np.linalg.norm(pos_list[-1] - intersection_point) > 1e-5:
            pos_list.insert(-1, intersection_point)
            ax_list.insert(-1, z_tool)
            
    pos = np.array(pos_list, dtype=float)
    ax = np.array(ax_list, dtype=float)
    num_frames = len(pos)

    # ---

    z_prev = ax[0] / np.linalg.norm(ax[0])
    
    if np.abs(z_prev[2]) < 0.9:
        x_prev = np.array([0.0, 0.0, 1.0]) - z_prev[2] * z_prev
    else:
        x_prev = np.array([1.0, 0.0, 0.0]) - z_prev[0] * z_prev
    x_prev /= np.linalg.norm(x_prev)
    
    O_prev = pos[0]
    
    dh_table = []

    # ---

    for i in range(num_frames - 1):
        z_curr = ax[i] / np.linalg.norm(ax[i])
        z_next = ax[i+1] / np.linalg.norm(ax[i+1])
        
        # --- zaj. normala
        cos_alpha = np.clip(np.dot(z_curr, z_next), -1.0, 1.0)
        
        if np.abs(cos_alpha) > 0.999: # z su paralelni
            v = pos[i+1] - pos[i]
            x_curr = v - np.dot(v, z_curr) * z_curr
            
            # oba su na istoj z liniji
            if np.linalg.norm(x_curr) < 1e-6:
                if np.abs(np.dot(z_curr, [1, 0, 0])) < 0.9:
                    x_curr = np.cross(z_curr, [1, 0, 0])
                else:
                    x_curr = np.cross(z_curr, [0, 1, 0])
        else:
            x_curr = np.cross(z_curr, z_next)
            
        x_curr /= np.linalg.norm(x_curr)
        
        # --- tocke na zajednickoj normali

        D = pos[i+1] - pos[i]
        if np.abs(cos_alpha) > 0.999:  # paralel
            s = 0.0
            t = np.dot(D, z_curr)
        else:
            s = (np.dot(D, z_curr) * cos_alpha - np.dot(D, z_next)) / (1.0 - cos_alpha**2)
            t = np.dot(D, z_curr) + s * cos_alpha
            
        Q_curr = pos[i] + t * z_curr
        Q_next = pos[i+1] + s * z_next

        # --- 
        
        # 1. theta
        cos_theta = np.clip(np.dot(x_prev, x_curr), -1.0, 1.0)
        sin_theta = np.dot(np.cross(x_prev, x_curr), z_curr)
        theta = np.arctan2(sin_theta, cos_theta)
        
        # 2. d 
        d = np.dot(Q_curr - O_prev, z_curr)
        
        # 3. a
        a = np.dot(Q_next - Q_curr, x_curr)
        
        # 4. alpha
        sin_alpha = np.dot(np.cross(z_curr, z_next), x_curr)
        alpha = np.arctan2(sin_alpha, cos_alpha)
        
        dh_table.append((theta, d, a, alpha))
        
        # ---
        x_prev = x_curr
        z_prev = z_next
        O_prev = Q_next
        
    return dh_table

def solve_fwd(dh_table):
    T_total = np.identity(4)
    for theta, d, a, alpha in dh_table:
        T_i = np.array([
            [np.cos(theta), -np.cos(alpha) * np.sin(theta),  np.sin(alpha) * np.sin(theta), a * np.cos(theta)],
            [np.sin(theta),  np.cos(alpha) * np.cos(theta), -np.sin(alpha) * np.cos(theta), a * np.sin(theta)],
            [0.0,            np.sin(alpha),                 np.cos(alpha),                d],
            [0.0,            0.0,                           0.0,                          1.0]
        ])
        T_total = np.dot(T_total, T_i)

    return T_total, T_total[0:3, 3]

def gen_trajectory(
    q0: np.ndarray,
    q1: np.ndarray,
    qd_max: np.ndarray,
    qdd_max: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    n_points: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if np.any(q0 < q_min) or np.any(q0 > q_max):
        raise RuntimeError("q0 oob")
    if np.any(q1 < q_min) or np.any(q1 > q_max):
        raise RuntimeError("q1 oob")

    dq = q1 - q0
    sgn_dq = np.sign(dq)

    moving = np.abs(dq) > 1e-6
    
    T1 = np.zeros_like(q0, dtype=float)
    TU = np.zeros_like(q0, dtype=float)
    
    # efektinvna max brzina (moze biti manja ako je kratka distanca)
    qd_max_eff = qd_max.copy()
    
    for j in range(len(q0)):
        if moving[j]:
            dq_abs = np.abs(dq[j])
            v_max = qd_max[j]
            a_max = qdd_max[j]
            
            # v_max -> trapez
            if dq_abs >= (v_max**2) / a_max:
                T1[j] = v_max / a_max
                TU[j] = (dq_abs / v_max) + T1[j]
            # ispod v_max -> trokut
            else:
                v_peak = np.sqrt(dq_abs * a_max)
                qd_max_eff[j] = v_peak
                T1[j] = v_peak / a_max
                TU[j] = 2 * T1[j]

    T_ref = np.max(TU)
    
    # nema pokreta
    if T_ref == 0:
        return np.zeros(n_points), np.tile(q0, (n_points, 1)), np.zeros((n_points, len(q0)))

    k = np.zeros_like(q0, dtype=float)
    k[moving] = TU[moving] / T_ref
    
    v_sync = qd_max_eff * k
    a_sync = qdd_max * (k ** 2)
    
    T1_sync = np.zeros_like(q0, dtype=float)
    T1_sync[moving] = v_sync[moving] / a_sync[moving]

    t_arr = np.linspace(0, T_ref, n_points)
    
    num_joints = len(q0)
    q_t = np.zeros((n_points, num_joints))
    qd_t = np.zeros((n_points, num_joints))

    for j in range(num_joints):
        if not moving[j]:
            q_t[:, j] = q0[j]
            continue
            
        for i, t in enumerate(t_arr):
            # ubrzavanje
            if t <= T1_sync[j]:
                q_t[i, j] = q0[j] + 0.5 * a_sync[j] * (t**2) * sgn_dq[j]
                qd_t[i, j] = a_sync[j] * t * sgn_dq[j]
                
            # jednako gibanje
            elif t <= (T_ref - T1_sync[j] + 1e-9):
                q_t[i, j] = q0[j] + (0.5 * a_sync[j] * (T1_sync[j]**2) + v_sync[j] * (t - T1_sync[j])) * sgn_dq[j]
                qd_t[i, j] = v_sync[j] * sgn_dq[j]
                
            # usporavanje
            else:
                q_t[i, j] = q1[j] - 0.5 * a_sync[j] * ((T_ref - t)**2) * sgn_dq[j]
                qd_t[i, j] = a_sync[j] * (T_ref - t) * sgn_dq[j]

    return t_arr, q_t, qd_t

def gen_gif(frames, output):
    frames = [np.array(Image.open(io.BytesIO(b)).convert("RGB")) for b in frames]
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output,
        save_all=True,
        append_images=pil_frames[1:],
        duration=33,
        loop=0,
        optimize=True
    )