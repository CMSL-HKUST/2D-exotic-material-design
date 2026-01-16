# uc_homogenize_xdmf_from_meshio_2phases.py
from __future__ import print_function
from fenics import *
import numpy as np

# ---------------- user settings ----------------
MESH_XDMF      = "unitcell_2phase.xdmf"      # produced by your meshio script
CELL_DATA_NAME = "name_to_read"       # <-- matches your meshio cell_data key
PLANE_STRESS   = True                  # False -> plane strain
deg            = 1                   # element degree (1 or 2)

# === 两相材料：id=1 与 id=2 ===
E1, nu1 = 1351, 0.32        # phase-1（例如实体相）
E2, nu2 = 1e-6*E1, 0.32    # phase-2（例如孔相：ersatz 材料；若是真两相，改成真实值）
PHASES  = {1: (E1, nu1), 2: (E2, nu2)}   # {phase_id: (E, nu)}

# ------------- read mesh & cell tags (meshio style) ------------
mesh = Mesh()
with XDMFFile(MESH_XDMF) as xf:
    xf.read(mesh)

tdim = mesh.topology().dim()
domains = MeshFunction("size_t", mesh, tdim, 0)
with XDMFFile(MESH_XDMF) as xf:
    try:
        xf.read(domains, CELL_DATA_NAME)
    except RuntimeError:
        mvc = MeshValueCollection("size_t", mesh, tdim)
        with XDMFFile(MESH_XDMF) as xf2:
            xf2.read(mvc, CELL_DATA_NAME)
        domains = cpp.mesh.MeshFunctionSizet(mesh, mvc)

tags = domains.array()
uniq, counts = np.unique(tags, return_counts=True)
print("Cell tags present:", dict(zip(uniq.tolist(), counts.tolist())))
if len(uniq) == 1 and uniq[0] == 0:
    raise RuntimeError("All cells are tag 0. Check XDMF cell_data key.")

# 检查网格中出现的 tag 是否都在 PHASES 里
missing = [int(t) for t in uniq if int(t) not in PHASES]
if missing:
    raise RuntimeError(f"Found cell tags without materials in PHASES: {missing}")
dxm = Measure("dx", domain=mesh, subdomain_data=domains)

# ------------- periodic boundary from bbox -------
coords = mesh.coordinates()
x_min, x_max = coords[:,0].min(), coords[:,0].max()
y_min, y_max = coords[:,1].min(), coords[:,1].max()
Lx, Ly = x_max - x_min, y_max - y_min

class PeriodicBoundary(SubDomain):
    def __init__(self, tol=1e-10):
        super().__init__()
        self.tol = tol
    def inside(self, x, on_boundary):
        # 左边界或下边界为“主边界”；排除右上角重复点
        return on_boundary and (
            near(x[0], x_min, self.tol) or near(x[1], y_min, self.tol)
        ) and not (near(x[0], x_max, self.tol) and near(x[1], y_max, self.tol))
    def map(self, x, y):
        y[0], y[1] = x[0], x[1]
        if near(x[0], x_max, self.tol): y[0] = x[0] - Lx
        if near(x[1], y_max, self.tol): y[1] = x[1] - Ly

pbc = PeriodicBoundary()

# ------------- function spaces -------------------
Ve = VectorElement("CG", mesh.ufl_cell(), deg)
Re = FiniteElement("R",  mesh.ufl_cell(), 0)          # two LMs for mean(u~)=0
W  = FunctionSpace(mesh, MixedElement([Ve, Re, Re]), constrained_domain=pbc)
V_DG0 = FunctionSpace(mesh, "DG", 0)                  # piecewise-constant fields

# ------------- materials (Voigt D fields) --------
# Voigt: [xx, yy, xy], with engineering shear gamma_xy = 2*e_xy
D11 = Function(V_DG0); D12 = Function(V_DG0); D22 = Function(V_DG0)
D16 = Function(V_DG0); D26 = Function(V_DG0); D66 = Function(V_DG0)

def make_D_2D(E, nu, plane_stress=True):
    if plane_stress:
        k = E/(1.0 - nu**2)
        return np.array([[k,      k*nu,   0.0],
                         [k*nu,   k,      0.0],
                         [0.0,    0.0,    k*(1.0 - nu)/2.0]])
    else:
        a  = E*(1.0-nu)/((1.0+nu)*(1.0-2.0*nu))
        b  = a*nu/(1.0-nu)
        G2 = E/(1.0+nu)        # 2μ (since gamma=2e)
        return np.array([[a,  b,  0.0],
                         [b,  a,  0.0],
                         [0.0, 0.0, G2]])

cell2id = domains.array()
dofs = np.arange(len(cell2id), dtype=np.int32)
d11 = D11.vector(); d12 = D12.vector(); d22 = D22.vector()
d16 = D16.vector(); d26 = D26.vector(); d66 = D66.vector()

for pid in np.unique(cell2id).astype(int):
    E, nu = PHASES[int(pid)]
    Dloc = make_D_2D(E, nu, PLANE_STRESS)
    mask = (cell2id == pid)
    idx  = dofs[mask]
    d11[idx] = Dloc[0,0]; d12[idx] = Dloc[0,1]; d22[idx] = Dloc[1,1]
    d16[idx] = Dloc[0,2]; d26[idx] = Dloc[1,2]; d66[idx] = Dloc[2,2]

def D_dot_e(evec):
    s0 = D11*evec[0] + D12*evec[1] + D16*evec[2]
    s1 = D12*evec[0] + D22*evec[1] + D26*evec[2]
    s2 = D16*evec[0] + D26*evec[1] + D66*evec[2]
    return as_vector([s0, s1, s2])

# ------------- strain helpers --------------------
def eps(u):       return sym(grad(u))
def ten2voigt(e): return as_vector([e[0,0], e[1,1], 2.0*e[0,1]])

# ------------- macro-strain bases ----------------
E_basis = [
    ("E1", as_tensor(((1.0, 0.0), (0.0, 0.0)))),  # Exx=1
    ("E2", as_tensor(((0.0, 0.0), (0.0, 1.0)))),  # Eyy=1
    ("E3", as_tensor(((0.0, 0.5), (0.5, 0.0)))),  # Exy=1/2 => gamma_xy=1
]
volume = assemble(1.0*dxm)

# ------------- cell problems (3 load cases) -------
(u, lx, ly) = TrialFunctions(W)
(v, qx, qy) = TestFunctions(W)
C_hom = np.zeros((3,3))

params_direct   = {"linear_solver": "mumps"}
params_fallback = {"linear_solver": "lu", "preconditioner": "none"}

for j, (name, E_macro) in enumerate(E_basis):
    e_ut = ten2voigt(eps(u))
    e_v  = ten2voigt(eps(v))

    a = inner(D_dot_e(e_ut), e_v)*dxm \
        + qx*dot(Constant((1.0,0.0)), u)*dxm \
        + qy*dot(Constant((0.0,1.0)), u)*dxm \
        + lx*dot(Constant((1.0,0.0)), v)*dxm \
        + ly*dot(Constant((0.0,1.0)), v)*dxm

    L = - inner(D_dot_e(ten2voigt(E_macro)), e_v)*dxm

    w = Function(W, name=f"w_{name}")
    try:
        solve(a == L, w, solver_parameters=params_direct)
    except RuntimeError:
        solve(a == L, w, solver_parameters=params_fallback)

    u_sol, _, _ = w.split()

    e_tot   = ten2voigt(eps(u_sol)) + ten2voigt(E_macro)
    s_voigt = D_dot_e(e_tot)
    s_avg = np.array([
        assemble(s_voigt[0]*dxm)/volume,
        assemble(s_voigt[1]*dxm)/volume,
        assemble(s_voigt[2]*dxm)/volume,
    ])
    C_hom[:, j] = s_avg

np.set_printoptions(precision=6, suppress=True)
print("\n=== Homogenized elasticity (2D Voigt [xx, yy, xy], γxy used) ===")
print("Plane stress" if PLANE_STRESS else "Plane strain")
print(C_hom)
np.savetxt("C_homogenized.csv", C_hom, delimiter=",")
print("Saved: C_homogenized.csv")

# --------- symmetry check & ortho identity ----------
C_hom = 0.5*(C_hom + C_hom.T)
C11, C12, C22, C33 = C_hom[0,0], C_hom[0,1], C_hom[1,1], C_hom[2,2]
theory = 0.25*(C11 - 2.0*C12 + C22)
difference = theory - C33
error = abs(difference)/max(abs(theory), 1e-16)*100.0
print(f"C11={C11:.6g}, C12={C12:.6g}, C22={C22:.6g}, C33={C33:.6g}")
print(f"1/4*(C11 - 2*C12 + C22) - C33 = {difference:.6g}")
print(f"error = {error:.6g}%")
