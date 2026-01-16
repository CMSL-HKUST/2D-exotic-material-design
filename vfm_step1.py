"""
Version: final.1
Date: 2025-11-19
Author: mou guangjin
Email: mouguangjin@ust.hk

Step 1: This script is used to calculate the average strain of the coarse grid.
(method: Q4 + 2×2 Gauss)

Input file: displacement field of R0 material during tensile test
Output file: average strain of the coarse grids (10*10 for each frame)

To do: replace the input file "u_tension.xdmf" 
"""



from fenics import *
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ------------------- User settings -------------------
INPUT_DIR              = "inputs"
OUTPUT_DIR             = "outputs"
MESH_BASENAME          = "R0_2D"
U_XDMF_NAME            = os.path.join(INPUT_DIR, "u_tension.xdmf")
U_CHECKPOINT_NAME      = "u"

# coarse grid setting
NX = 10
NY = 10

# save the last 10 frames average strain data
N_LAST_FRAMES = 10

# visualization setting
PLOT_ALPHA = 0.7  # Transparency level for strain distribution plot (0.0 = transparent, 1.0 = opaque)
COLORBAR_FRACTION = 0.05  # Colorbar thickness (width) relative to axes. Smaller value = thinner colorbar
COLORBAR_LABEL_FONTSIZE = 10  # Font size for colorbar label (title)
COLORBAR_TICK_FONTSIZE = 25  # Font size for colorbar tick labels (numbers on the colorbar)
COLORBAR_TICK_FONTFAMILY = "Times New Roman"  # Font family for colorbar tick numbers
COLORBAR_TICK_FONTWEIGHT = "bold"  # Font weight for colorbar tick numbers
CELL_VALUE_FONTSIZE = 10  # Font size for values shown inside each coarse cell


# ========== load mesh ==========
mesh = Mesh()
mesh_file = os.path.join(INPUT_DIR, f"{MESH_BASENAME}.xdmf")
with XDMFFile(mesh_file) as f:
    f.read(mesh)

# Load domains (material regions) for background visualization
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
has_domains = False
try:
    with XDMFFile(mesh_file) as f:
        # Try to read domains (may not exist in all mesh files)
        f.read(domains)
    has_domains = True
except:
    try:
        # Try alternative: read with explicit name
        with XDMFFile(mesh_file) as f:
            f.read(domains, "domains")
        has_domains = True
    except:
        # If domains are not found, create a dummy one (all cells same value)
        domains.set_all(1)
        has_domains = False

mpi_rank = MPI.rank(mesh.mpi_comm())
if mpi_rank == 0:
    print(f"[INFO] Mesh loaded: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")
    print(f"[INFO] Domain information loaded: {has_domains}")

# ========== coarse grid definition (10×10) ==========
coords = mesh.coordinates()
x_min = coords[:, 0].min()
x_max = coords[:, 0].max()
y_min = coords[:, 1].min()
y_max = coords[:, 1].max()

Lx = x_max - x_min
Ly = y_max - y_min

dx_coarse = Lx / NX
dy_coarse = Ly / NY

if mpi_rank == 0:
    print(f"[INFO] Domain bbox: x in [{x_min}, {x_max}], y in [{y_min}, {y_max}]")
    print(f"[INFO] Coarse cell size: dx={dx_coarse}, dy={dy_coarse}")

# 2×2 Gauss points on [-1,1]×[-1,1]
# weights are all 1, here we can directly take the average of 4 points
g = 1.0 / np.sqrt(3.0)
gauss_pts = [(-g, -g),
             ( g, -g),
             ( g,  g),
             (-g,  g)]


def q4_dN_dxi_eta(xi, eta):
    """
    For a Q4 element, given reference coordinates (xi, eta) ∈ [-1, 1]^2,
    return the derivatives of the 4 shape functions with respect to (xi, eta):
    dN/dxi and dN/deta.

    Node ordering:
        1:(-1,-1), 2:(+1,-1), 3:(+1,+1), 4:(-1,+1)
    """
    # dN/dxi
    dN1_dxi = -0.25 * (1 - eta)
    dN2_dxi =  0.25 * (1 - eta)
    dN3_dxi =  0.25 * (1 + eta)
    dN4_dxi = -0.25 * (1 + eta)
    # dN/deta
    dN1_deta = -0.25 * (1 - xi)
    dN2_deta = -0.25 * (1 + xi)
    dN3_deta =  0.25 * (1 + xi)
    dN4_deta =  0.25 * (1 - xi)

    dN_dxi  = np.array([dN1_dxi,  dN2_dxi,  dN3_dxi,  dN4_dxi])
    dN_deta = np.array([dN1_deta, dN2_deta, dN3_deta, dN4_deta])
    return dN_dxi, dN_deta


# since the coarse cell are regular rectangles, the mapping is:
#   x = x0 + (xi+1)/2 * dx_coarse
#   y = y0 + (eta+1)/2 * dy_coarse
# The Jacobian is:
#   dx/dxi = dx_coarse/2,  dy/deta = dy_coarse/2
#   dxi/dx = 2/dx_coarse,  deta/dy = 2/dy_coarse
invJ_x = 2.0 / dx_coarse
invJ_y = 2.0 / dy_coarse


def compute_coarse_strain_Q4(u_fun):
    """
    Given a displacement field u_fun (VectorFunctionSpace, CG1),
    compute the average strain on the 10×10 coarse grid using Q4 + 2×2 Gauss quadrature.

    Returns
    -------
    eps_xx_avg, eps_yy_avg, eps_xy_avg : np.ndarray, shape (NY, NX)
    """
    eps_xx_avg = np.zeros((NY, NX))
    eps_yy_avg = np.zeros((NY, NX))
    eps_xy_avg = np.zeros((NY, NX))

    for j in range(NY):
        for i in range(NX):
            # the 4 corner points of the current coarse cell
            x0 = x_min + i     * dx_coarse
            x1 = x_min + (i+1) * dx_coarse
            y0 = y_min + j     * dy_coarse
            y1 = y_min + (j+1) * dy_coarse

            # the node order is consistent with Q4:
            # 1:(x0,y0), 2:(x1,y0), 3:(x1,y1), 4:(x0,y1)
            p1 = Point(x0, y0, 0.0)
            p2 = Point(x1, y0, 0.0)
            p3 = Point(x1, y1, 0.0)
            p4 = Point(x0, y1, 0.0)

            u1 = u_fun(p1)  # (ux, uy)
            u2 = u_fun(p2)
            u3 = u_fun(p3)
            u4 = u_fun(p4)

            # 4×2 matrix, each row is a node displacement vector
            U = np.array([[u1[0], u1[1]],
                          [u2[0], u2[1]],
                          [u3[0], u3[1]],
                          [u4[0], u4[1]]])

            # calculate the strain at the 2×2 Gauss points, then take the average as the cell average strain
            eps_xx_cell = 0.0
            eps_yy_cell = 0.0
            eps_xy_cell = 0.0

            for (xi, eta) in gauss_pts:
                dN_dxi, dN_deta = q4_dN_dxi_eta(xi, eta)

                # dN/dx = dN/dxi * dxi/dx
                dN_dx = dN_dxi  * invJ_x
                dN_dy = dN_deta * invJ_y

                # calculate the gradient: grad u = [ [∂ux/∂x, ∂ux/∂y],
                #                      [∂uy/∂x, ∂uy/∂y] ]
                dux_dx = np.dot(dN_dx, U[:, 0])
                dux_dy = np.dot(dN_dy, U[:, 0])
                duy_dx = np.dot(dN_dx, U[:, 1])
                duy_dy = np.dot(dN_dy, U[:, 1])

                eps_xx_g = dux_dx
                eps_yy_g = duy_dy
                eps_xy_g = 0.5 * (dux_dy + duy_dx)

                eps_xx_cell += eps_xx_g
                eps_yy_cell += eps_yy_g
                eps_xy_cell += eps_xy_g

            # the average of 4 Gauss points
            eps_xx_avg[j, i] = eps_xx_cell / 4.0
            eps_yy_avg[j, i] = eps_yy_cell / 4.0
            eps_xy_avg[j, i] = eps_xy_cell / 4.0

    return eps_xx_avg, eps_yy_avg, eps_xy_avg


# ========== read all frames from XDMF and compute coarse strain ==========
V_u = VectorFunctionSpace(mesh, "CG", 1)
u_fun = Function(V_u)

eps_xx_frames = []
eps_yy_frames = []
eps_xy_frames = []

with XDMFFile(U_XDMF_NAME) as xdmf:
    k = 0
    while True:
        try:
            xdmf.read_checkpoint(u_fun, U_CHECKPOINT_NAME, k)
        except RuntimeError:
            # no more time steps
            break

        if mpi_rank == 0:
            print(f"[INFO] Processing frame {k} ...")

        exx_avg, eyy_avg, exy_avg = compute_coarse_strain_Q4(u_fun)

        eps_xx_frames.append(exx_avg)
        eps_yy_frames.append(eyy_avg)
        eps_xy_frames.append(exy_avg)

        k += 1

n_frames_total = len(eps_xx_frames)
if mpi_rank == 0:
    print(f"[INFO] Total frames found in XDMF: {n_frames_total}")

if n_frames_total == 0:
    raise RuntimeError("No frames were read from XDMF file. Check U_XDMF_NAME / checkpoint name.")

# only keep the last N_LAST_FRAMES frames
n_use = min(N_LAST_FRAMES, n_frames_total)
eps_xx_arr = np.stack(eps_xx_frames[-n_use:], axis=0)  # shape (n_use, NY, NX)
eps_yy_arr = np.stack(eps_yy_frames[-n_use:], axis=0)
eps_xy_arr = np.stack(eps_xy_frames[-n_use:], axis=0)

if mpi_rank == 0:
    print(f"[INFO] Keeping last {n_use} frames for VFM.")

# ========== save average strain data to npz ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, f"coarse_strain_Q4_last{n_use}.npz")

if mpi_rank == 0:
    np.savez(out_path,
             eps_xx=eps_xx_arr,
             eps_yy=eps_yy_arr,
             eps_xy=eps_xy_arr,
             NX=NX, NY=NY,
             x_min=x_min, x_max=x_max,
             y_min=y_min, y_max=y_max,
             dx_coarse=dx_coarse, dy_coarse=dy_coarse,
             n_frames=n_use)
    print(f"[INFO] Saved coarse strain data to: {out_path}")

                          #postprocessing 
# ========== visualize strain distribution of final step ==========
def plot_strain_distribution(eps_xx, eps_yy, eps_xy, mesh, domains, x_min, y_min, dx_coarse, dy_coarse, output_dir, alpha=0.5):
    """
    Plot the strain distribution for the final step.
    Shows eps_xx, eps_yy, and eps_xy with color mapping and value labels on each cell.
    Each strain component is saved as a separate PDF file.
    The domain (material structure) is plotted as gray background, and the strain distribution
    is overlaid with semi-transparent colors.
    
    Parameters
    ----------
    eps_xx, eps_yy, eps_xy : np.ndarray, shape (NY, NX)
        Strain components for the final step
    mesh : dolfin.Mesh
        The mesh object for domain visualization
    domains : dolfin.MeshFunction
        Domain mesh function for material regions
    x_min, y_min : float
        Minimum coordinates of the domain
    dx_coarse, dy_coarse : float
        Size of each coarse cell
    output_dir : str
        Directory to save the plots
    alpha : float, optional
        Transparency level for the color map (0.0 = fully transparent, 1.0 = fully opaque). Default is 0.5.
    """
    NY, NX = eps_xx.shape
    
    # Strain components, their labels, and file names
    strains = [eps_xx, eps_yy, eps_xy]
    labels = [r'$\varepsilon_{xx}$', r'$\varepsilon_{yy}$', r'$\varepsilon_{xy}$']
    filenames = ['strain_distribution_eps_xx_final_step.pdf', 
                 'strain_distribution_eps_yy_final_step.pdf', 
                 'strain_distribution_eps_xy_final_step.pdf']
    
    for strain, label, filename in zip(strains, labels, filenames):
        # Create a separate figure for each strain component
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle(f'{label} Distribution - Final Step (Average strain per coarse cell)', 
                     fontsize=14, fontweight='bold')
        
        # First, plot the domain (material structure) as gray background
        # FEniCS plot needs to operate on current figure/axes
        try:
            # Save current matplotlib state
            orig_fig = plt.gcf()
            orig_ax = plt.gca()
            
            # Switch to target figure and axes
            plt.figure(fig.number)
            plt.sca(ax)
            
            # Create a gray colormap for domain visualization
            gray_colors = ['#E0E0E0', '#B0B0B0', '#808080', '#606060']
            gray_cmap = mcolors.LinearSegmentedColormap.from_list('gray_domain', gray_colors, N=256)
            
            # Plot domain in gray color as background layer
            # FEniCS plot will use current axes set by plt.sca()
            plot(domains, cmap=gray_cmap, alpha=0.8)
            
            # Restore original figure/axes
            plt.figure(orig_fig.number)
            plt.sca(orig_ax)
            
        except Exception as e:
            print(f"[WARNING] Could not plot domain background: {e}")
            # Try to restore state
            try:
                plt.figure(orig_fig.number)
                plt.sca(orig_ax)
            except:
                pass
        
        # Create coordinate arrays for pcolormesh
        x_edges = np.linspace(x_min, x_min + NX * dx_coarse, NX + 1)
        y_edges = np.linspace(y_min, y_min + NY * dy_coarse, NY + 1)
        X, Y = np.meshgrid(x_edges, y_edges)
        
        # Plot pcolormesh with colormap and semi-transparent colors (on top of domain)
        im = ax.pcolormesh(X, Y, strain, cmap='viridis', shading='flat', 
                          edgecolors='white', linewidth=1.5, alpha=alpha)
        
        # Add colorbar with custom ticks (5 intervals = 6 tick points)
        # We'll manually set position after tight_layout to ensure exact height match
        cbar = plt.colorbar(im, ax=ax, fraction=COLORBAR_FRACTION, pad=0.04)
        cbar.set_label(label, fontsize=COLORBAR_LABEL_FONTSIZE, rotation=0, labelpad=10)
        
        # Set custom colorbar ticks: only show min, middle, and max values
        strain_min = strain.min()
        strain_max = strain.max()
        strain_mid = (strain_min + strain_max) / 2.0
        
        # Create 3 tick values: min, middle, max
        tick_values = [strain_min, strain_mid, strain_max]
        
        # Format labels: use scientific notation for very small/large numbers, otherwise 4 decimal places
        tick_labels = []
        for val in tick_values:
            if abs(val) < 0.001 or abs(val) > 1000:
                tick_labels.append(f'{val:.2e}')
            else:
                tick_labels.append(f'{val:.4f}')
        
        # Set tick positions and labels
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)  # Set tick label font size

        # Make colorbar tick numbers Times New Roman + bold (value range numbers)
        # Most colorbars are vertical -> y tick labels; keep x as fallback.
        for tl in cbar.ax.get_yticklabels():
            tl.set_fontfamily(COLORBAR_TICK_FONTFAMILY)
            tl.set_fontweight(COLORBAR_TICK_FONTWEIGHT)
        for tl in cbar.ax.get_xticklabels():
            tl.set_fontfamily(COLORBAR_TICK_FONTFAMILY)
            tl.set_fontweight(COLORBAR_TICK_FONTWEIGHT)
        
        # Add text annotations with strain values on each cell
        for j in range(NY):
            for i in range(NX):
                # Center of the cell
                x_center = x_min + (i + 0.5) * dx_coarse
                y_center = y_min + (j + 0.5) * dy_coarse
                value = strain[j, i]
                
                # Format the value (use scientific notation for very small/large numbers)
                if abs(value) < 0.001 or abs(value) > 1000:
                    text_str = f'{value:.2e}'
                else:
                    text_str = f'{value:.4f}'
                
                # Add text with black bold font
                ax.text(x_center, y_center, text_str, 
                       ha='center', va='center', 
                       fontsize=CELL_VALUE_FONTSIZE, color='black', fontweight='bold')
        
        # Set labels and title
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Use tight_layout with rect parameter to reserve space for colorbar on the right
        # rect: [left, bottom, right, top] in figure coordinates
        # Reserve extra space on the right for colorbar (about 10% more)
        fig.tight_layout(rect=[0, 0, 0.92, 1])
        
        # Manually adjust colorbar to match axes height exactly (after tight_layout)
        # Get axes position after tight_layout
        ax_pos = ax.get_position()
        
        # Calculate colorbar position: same y0, y1 as axes, x position after axes with padding
        cbar_width = COLORBAR_FRACTION * ax_pos.width
        cbar_x0 = ax_pos.x1 + 0.04  # pad=0.04
        cbar_y0 = ax_pos.y0  # Match bottom of axes exactly
        cbar_y1 = ax_pos.y1  # Match top of axes exactly
        cbar_height = cbar_y1 - cbar_y0
        
        # Set colorbar axes position to exactly match axes height
        cbar.ax.set_position([cbar_x0, cbar_y0, cbar_width, cbar_height])
        
        # Save the figure as PDF
        # Use bbox_inches='tight' with pad_inches to ensure colorbar is included
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"[INFO] Saved {label} strain distribution plot to: {plot_path}")
        plt.close()


if mpi_rank == 0:
    # Get the final step strain data
    eps_xx_final = eps_xx_arr[-1]  # shape (NY, NX)
    eps_yy_final = eps_yy_arr[-1]
    eps_xy_final = eps_xy_arr[-1]
    
    print(f"[INFO] Visualizing strain distribution for final step...")
    plot_strain_distribution(eps_xx_final, eps_yy_final, eps_xy_final,
                            mesh, domains, x_min, y_min, dx_coarse, dy_coarse, OUTPUT_DIR,
                            alpha=PLOT_ALPHA)
