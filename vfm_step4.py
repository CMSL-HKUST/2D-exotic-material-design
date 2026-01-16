"""
Version: final.1
Date: 2025-11-19
Author: mou guangjin
Email: mouguangjin@ust.hk

Step 4 : This script is used to solve the interior VFM equations (x+y).


Output file: identified stiffness matrix

To do: set WEIGHT_HALF_DOMAIN and N_HALF_FRAMES
       set Cij_th values
"""

import numpy as np
import os

# ============================================================
# =========== User Settings（你可以修改这里）====================
# ============================================================
OUTPUT_DIR   = "outputs"

HALF_NPZ     = os.path.join(OUTPUT_DIR, "half_domain_multiline_system.npz")
INTERIOR_ALL_NPZ = os.path.join(OUTPUT_DIR, "interior_system_all.npz")  # 直接读取所有内域方程

WEIGHT_HALF_DOMAIN = 10  # 半域权重
N_HALF_FRAMES      = 10      # 半域使用多少帧（最后 N 帧）


# ---------- 理论值（可选） ----------
HAS_THEORY = True
C11_th = 56.73
C22_th = 545#566.72
C66_th = 118.85
C12_th = 73.95



# ================= Load half-domain ================
half = np.load(HALF_NPZ)
A_half_raw = half["A"]      # shape = (n_equations, 2)  -> [C22, C12]
b_half_all = half["b"]      # shape = (n_equations,)

n_half_all = A_half_raw.shape[0]
print(f"[INFO] Loaded half-domain: A_half_raw = {A_half_raw.shape}")

# each frame has 3 half domain equations (1/4,1/2,3/4)
HALF_PER_FRAME = 3
total_frames_available = n_half_all // HALF_PER_FRAME
print(f"[INFO] Half-domain contains {total_frames_available} frames.")

if N_HALF_FRAMES > total_frames_available:
    raise ValueError(
        f"N_HALF_FRAMES={N_HALF_FRAMES} exceeds total available frames {total_frames_available}"
    )

# the equation indices of the last N_HALF_FRAMES frames
start_idx = (total_frames_available - N_HALF_FRAMES) * HALF_PER_FRAME
end_idx   = total_frames_available * HALF_PER_FRAME

A_half_raw = A_half_raw[start_idx:end_idx]
b_half     = b_half_all[start_idx:end_idx]

print(f"[INFO] Using last {N_HALF_FRAMES} half-domain frames → {A_half_raw.shape[0]} equations")

# expand to 4 columns: C11,C22,C66,C12
n_half = A_half_raw.shape[0]
A_half = np.zeros((n_half, 4))
A_half[:, 1] = A_half_raw[:, 0]   # C22
A_half[:, 3] = A_half_raw[:, 1]   # C12

print(f"[INFO] Expanded A_half to {A_half.shape} (columns: [C11, C22, C66, C12])")
print(f"[INFO] Half-domain weight = {WEIGHT_HALF_DOMAIN}\n")



# ================= Load interior (x+y) ===============
interior_all = np.load(INTERIOR_ALL_NPZ)
A_int_xy = interior_all["A"]  # shape = (n_eq, 4) -> [C11, C22, C66, C12]
b_int_xy = interior_all["b"]  # zeros

#make sure A_int_xy is a 2D array
if A_int_xy.ndim == 1:
    if len(A_int_xy) == 0:
        A_int_xy = A_int_xy.reshape(0, 4)
    else:
        n_eq = len(A_int_xy) // 4
        if len(A_int_xy) == n_eq * 4:
            A_int_xy = A_int_xy.reshape(n_eq, 4)
        else:
            raise ValueError(f"Cannot reshape A_int_xy from shape {A_int_xy.shape} to (n_eq, 4)")

print(f"[INFO] Loaded interior-ALL system: A_int_xy = {A_int_xy.shape}, b_int_xy = {b_int_xy.shape}")
print(f"[INFO] Using ALL interior equations (both x & y directions).")



# ============= Stack with weighting ===============
A_half_w = A_half * WEIGHT_HALF_DOMAIN
b_half_w = b_half * WEIGHT_HALF_DOMAIN

A_stack = np.vstack([A_half_w, A_int_xy])
b_stack = np.concatenate([b_half_w, b_int_xy])

print(f"[INFO] Global stacked system: A_stack = {A_stack.shape}, b_stack = {b_stack.shape}")
print(f"[INFO]   Half-domain equations weighted by {WEIGHT_HALF_DOMAIN}")
print(f"[INFO]   Interior XY equations weighted by 1.0")


# ================= SVD Analysis of Combined System =================
print("\n" + "="*70)
print("SVD SENSITIVITY ANALYSIS OF COMBINED SYSTEM")
print("="*70)

# Perform SVD
u, s, vt = np.linalg.svd(A_stack, full_matrices=False)
rank = np.linalg.matrix_rank(A_stack)

print(f"\n[SVD INFO] Matrix A_stack shape: {A_stack.shape}")
print(f"[SVD INFO] Rank: {rank} / {min(A_stack.shape)}")

# Singular values
print("\n[SVD RESULT] Singular values (information content):")
param_names = ["C11", "C22", "C66", "C12"]
for i, sv in enumerate(s):
    rel_info = (sv / s[0] * 100) if s[0] > 1e-16 else 0
    print(f"  s[{i}] = {sv:.6e}  ({rel_info:.2f}% of s[0])")

# Condition number
if s[-1] > 1e-16:
    cond = s[0] / s[-1]
else:
    cond = np.inf
print(f"\n[SVD RESULT] Condition number: {cond:.3e}")

if cond < 100:
    print("             ✓ Excellent conditioning (cond < 100)")
elif cond < 1000:
    print("             ⚠ Good conditioning (100 < cond < 1000)")
elif cond < 10000:
    print("             ⚠ Moderate conditioning (1000 < cond < 10000)")
else:
    print("             ✗ Poor conditioning (cond > 10000)")

# Right singular vectors (parameter space directions)
print("\n[SVD RESULT] Right singular vectors V^T (parameter directions):")
print(f"             Parameters: {param_names}")
for i in range(vt.shape[0]):
    v = vt[i, :]
    v_str = ", ".join(f"{val:+.4f}" for val in v)
    info_pct = (s[i] / s[0] * 100) if s[0] > 1e-16 else 0
    print(f"  v[{i}] = [{v_str}]  (info: {info_pct:.1f}%)")

# Most identifiable parameter combination
v_max = vt[0, :]
print("\n[SVD INTERPRET] Most identifiable parameter combination:")
combo_terms_max = [f"{name}*{coef:+.4f}" for name, coef in zip(param_names, v_max)]
combo_str_max = " + ".join(combo_terms_max).replace("+ -", "- ")
print(f"  Direction: {combo_str_max}")

abs_coeffs_max = np.abs(v_max)
max_idx_max = np.argmax(abs_coeffs_max)
if abs_coeffs_max[max_idx_max] > 0.5:
    print(f"  → Dominated by: {param_names[max_idx_max]} (|coef|={abs_coeffs_max[max_idx_max]:.4f})")
else:
    print(f"  → Mixed combination")

# Least identifiable parameter combination
v_min = vt[-1, :]
print("\n[SVD INTERPRET] Least identifiable parameter combination:")
combo_terms_min = [f"{name}*{coef:+.4f}" for name, coef in zip(param_names, v_min)]
combo_str_min = " + ".join(combo_terms_min).replace("+ -", "- ")
print(f"  Direction: {combo_str_min}")

abs_coeffs_min = np.abs(v_min)
max_idx_min = np.argmax(abs_coeffs_min)
if abs_coeffs_min[max_idx_min] > 0.5:
    print(f"  → Dominated by: {param_names[max_idx_min]} (|coef|={abs_coeffs_min[max_idx_min]:.4f})")
else:
    print(f"  → Mixed combination")

# Parameter sensitivity analysis
print("\n[SVD INTERPRET] Parameter sensitivity summary:")
for i, pname in enumerate(param_names):
    # Calculate how much each singular vector contributes to this parameter
    contributions = np.abs(vt[:, i]) * s
    total_contribution = np.sum(contributions)
    max_contribution = contributions[0]
    sensitivity = max_contribution / total_contribution if total_contribution > 1e-16 else 0
    
    print(f"  {pname:4s}: sensitivity = {sensitivity:.3f}  ", end="")
    if sensitivity > 0.8:
        print("(✓ highly identifiable)")
    elif sensitivity > 0.5:
        print("(⚠ moderately identifiable)")
    else:
        print("(✗ poorly identifiable)")

print("="*70 + "\n")

# 假设已经有 A_stack (N,4)，列顺序: [C11, C22, C66, C12]

# 1) 定义 w = C66 - 1/4*(C11 - 2*C12 + C22)
d_w = np.array([-0.25, -0.25, 1.0, 0.5])  # 对应 (C11, C22, C66, C12)

# 2) 归一化方向（用于解释）
d_w_norm = d_w / np.linalg.norm(d_w)

# 3) 计算沿 w 方向的“有效奇异值”
Aw = A_stack @ d_w_norm              # (N,)
sigma_eff = np.linalg.norm(Aw)       # 标量

print("\n===== Identifiability of R0-violation direction w =====")
print("w = C66 - 1/4*(C11 - 2*C12 + C22)")
print(f"||d_w|| = {np.linalg.norm(d_w):.4e}")
print(f"Effective singular value sigma_eff(w) ≈ {sigma_eff:.4e}")





# ================= Solve least squares =============
x, residuals, rank, s = np.linalg.lstsq(A_stack, b_stack, rcond=None)

C11, C22, C66, C12 = x

print("\n================= Identified stiffness =================")
print(f"C11 = {C11:.6e}")
print(f"C22 = {C22:.6e}")
print(f"C66 = {C66:.6e}")
print(f"C12 = {C12:.6e}")


# --------- diagnostics ---------
print("\n---- Linear Algebra Diagnostics ----")
print(f"rank(A_stack)  = {rank}")
print(f"singular values = {s}")
res_norm = np.linalg.norm(A_stack.dot(x) - b_stack)
print(f"||A x - b|| = {res_norm:.6e}")
print(f"cond(A_stack) ≈ {np.linalg.cond(A_stack):.3e}")



# ================== Print matrices ================
print("\n================= Identified Voigt stiffness matrix =================")
C_mat = np.array([
    [C11, C12, 0.0],
    [C12, C22, 0.0],
    [0.0,  0.0, C66]
])
for row in C_mat:
    print(" ".join(f"{v:12.6e}" for v in row))

# ========== C66 theoretical relation check ==========
print("\n================= C66 Theoretical Relation Check =================")
# for orthotropic materials, the theoretical value of C66 is: C66 = 1/4 * (C11 - 2*C12 + C22)
C66_theory = 0.25 * (C11 - 2.0 * C12 + C22)
print(f"C66 (identified)     = {C66:.6e}")
print(f"C66 (theory: 1/4*(C11-2*C12+C22)) = {C66_theory:.6e}")

if abs(C66) > 1e-10:
    diff_abs = abs(C66 - C66_theory)
    diff_percent = (diff_abs / abs(C66)) * 100.0
    print(f"Absolute difference  = {diff_abs:.6e}")
    print(f"Relative difference  = {diff_percent:.2f} %")
    
    if diff_percent < 1.0:
        print(f"✓ C66 is very close to theoretical value (difference < 1%)")
    elif diff_percent < 5.0:
        print(f"⚠ C66 is reasonably close to theoretical value (difference < 5%)")
    else:
        print(f"✗ C66 deviates significantly from theoretical value (difference >= 5%)")
else:
    print("[WARNING] C66 is too small, cannot compute relative difference")

if HAS_THEORY:
    print("\n================= Theoretical Voigt matrix =================")
    C_mat_th = np.array([
        [C11_th, C12_th, 0.0],
        [C12_th, C22_th, 0.0],
        [0.0,    0.0,   C66_th]
    ])
    for row in C_mat_th:
        print(" ".join(f"{v:12.6e}" for v in row))

    print("\n================= Component-wise error =================")
    print(f"ΔC11 = {C11-C11_th:+.6e}   ({(C11-C11_th)/C11_th*100:+.2f} %)")
    print(f"ΔC22 = {C22-C22_th:+.6e}   ({(C22-C22_th)/C22_th*100:+.2f} %)")
    print(f"ΔC66 = {C66-C66_th:+.6e}   ({(C66-C66_th)/C66_th*100:+.2f} %)")
    print(f"ΔC12 = {C12-C12_th:+.6e}   ({(C12-C12_th)/C12_th*100:+.2f} %)")
