import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import os
import zipfile
import glob
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

# Function to extract and load Lyapunov exponent data from zip files
def load_lyapunov_data(download_path):
    """
    Extract and load Lyapunov exponent data from downloaded zip files.
    
    Parameters:
    download_path: Path to the downloads folder
    
    Returns:
    data_periodic: Dictionary with domain sizes and Lyapunov exponents for periodic case
    data_oddperiodic: Dictionary with domain sizes and Lyapunov exponents for odd-periodic case
    """
    # Paths to the zip files
    periodic_zip = os.path.join(download_path, "lyapexpts_ksperiodic.zip")
    oddperiodic_zip = os.path.join(download_path, "lyapexpts_ksoddperiodic.zip")
    
    # Create temporary extraction directories
    temp_dir_periodic = os.path.join(download_path, "temp_periodic")
    temp_dir_oddperiodic = os.path.join(download_path, "temp_oddperiodic")
    
    os.makedirs(temp_dir_periodic, exist_ok=True)
    os.makedirs(temp_dir_oddperiodic, exist_ok=True)
    
    # Extract zip files
    with zipfile.ZipFile(periodic_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir_periodic)
        
    with zipfile.ZipFile(oddperiodic_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir_oddperiodic)
    
    # Load data from extracted files
    data_periodic = {}
    data_oddperiodic = {}
    
    # Process periodic data
    for filename in glob.glob(os.path.join(temp_dir_periodic, "L*.txt")):
        base = os.path.basename(filename)
        L_str = base[1:-4]  # Remove 'L' and '.txt'
        L = float(L_str.replace('p', '.'))  # Replace 'p' with '.'
        
        # Load Lyapunov exponents
        lyap_exponents = np.loadtxt(filename)
        data_periodic[L] = lyap_exponents
    
    # Process odd-periodic data
    for filename in glob.glob(os.path.join(temp_dir_oddperiodic, "L*.txt")):
        base = os.path.basename(filename)
        L_str = base[1:-4]  # Remove 'L' and '.txt'
        L = float(L_str.replace('p', '.'))  # Replace 'p' with '.'
        
        # Load Lyapunov exponents
        lyap_exponents = np.loadtxt(filename)
        data_oddperiodic[L] = lyap_exponents
    
    return data_periodic, data_oddperiodic

# Function to simulate Kuramoto-Sivashinsky PDE using spectral method
def simulate_ks(L=50, N=512, tmax=100, dt=0.25):
    """
    Simulate the Kuramoto-Sivashinsky equation on a periodic domain.
    
    Parameters:
    L: Domain length
    N: Number of spatial points
    tmax: Maximum simulation time
    dt: Time step for output
    
    Returns:
    t: Time points
    x: Spatial points
    u: Solution array (shape: time x space)
    """
    # Spatial grid
    x = np.linspace(0, L, N, endpoint=False)
    dx = L/N
    
    # Initial condition - small random perturbation
    np.random.seed(42)  # For reproducibility
    u0 = 0.1 * np.cos(x/16) * (1 + np.sin(x/16)) + 0.01 * np.random.randn(N)
    
    # Setup for spectral method
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    
    # Linear operator (in Fourier space)
    L_operator = k**2 - k**4
    
    # Store solution at specified time intervals
    t_eval = np.arange(0, tmax, dt)
    u_store = np.zeros((len(t_eval), N))
    
    # Function for ODE solver
    def rhs(t, u):
        u_hat = np.fft.fft(u)
        # Linear part
        dudt_hat = L_operator * u_hat
        # Nonlinear part
        nonlinear = -0.5 * np.fft.fft(u**2)
        dudt_hat += nonlinear * 1j * k
        return np.real(np.fft.ifft(dudt_hat))
    
    # Solve using SciPy's ODE solver
    sol = solve_ivp(
        rhs, 
        [0, tmax], 
        u0, 
        method='Radau', 
        t_eval=t_eval,
        rtol=1e-3,
        atol=1e-6
    )
    
    return sol.t, x, sol.y.T

# Path to the downloads folder
download_path = os.path.expanduser("~/Downloads")

# Load Lyapunov exponent data from zip files
try:
    data_periodic, data_oddperiodic = load_lyapunov_data(download_path)
    print(f"Loaded data for {len(data_periodic)} periodic domain sizes and {len(data_oddperiodic)} odd-periodic domain sizes")
    
    # Pick a domain size that exists in the data for simulation
    L_values_periodic = sorted(list(data_periodic.keys()))
    if L_values_periodic:
        L = L_values_periodic[len(L_values_periodic)//2]  # Choose a middle value
        print(f"Using domain size L = {L} for simulation")
    else:
        L = 100  # Default if no data is found
        print(f"No domain sizes found in data, using default L = {L}")
    
    # Flag indicating we successfully loaded data
    data_loaded = True
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Using synthetic data instead...")
    
    # Create synthetic data for demonstration
    L = 100  # Default domain size
    data_loaded = False
    
    # Synthetic data dictionaries
    data_periodic = {}
    data_oddperiodic = {}
    
    # Create a range of domain sizes and synthetic Lyapunov exponents
    for L_val in np.linspace(20, 200, 20):
        # Generate synthetic Lyapunov spectrum with realistic properties
        np.random.seed(int(L_val*10))  # Different seed for each size
        exponents = np.zeros(24)
        for i in range(24):
            # Pattern: a few positive, many near zero, then increasingly negative
            if i < 3:
                exponents[i] = 0.2 * (1 - i/3) + 0.05 * np.random.randn()
            elif i < 8:
                exponents[i] = 0.05 * (1 - (i-3)/5) + 0.02 * np.random.randn()
            else:
                exponents[i] = -0.1 * ((i-8)/16) + 0.01 * np.random.randn()
        
        data_periodic[L_val] = np.sort(exponents)[::-1]
        
        # Similar but different for odd-periodic
        np.random.seed(int(L_val*10) + 100)
        exponents_odd = exponents * 0.9 + 0.02 * np.random.randn(24)
        data_oddperiodic[L_val] = np.sort(exponents_odd)[::-1]

# Simulate KS equation with the chosen domain size
t, x, u = simulate_ks(L=L, N=512, tmax=100, dt=0.25)

# ==================== CREATE VISUALIZATIONS ====================

# Set up figure
fig = plt.figure(figsize=(15, 12), facecolor='black')
if data_loaded:
    fig.suptitle(f'Kuramoto-Sivashinsky Dynamics & Lyapunov Analysis (Real Data)', color='white', fontsize=20)
else:
    fig.suptitle(f'Kuramoto-Sivashinsky Dynamics & Lyapunov Analysis (Synthetic Data)', color='white', fontsize=20)

# 3D Surface plot (top left)
ax1 = fig.add_subplot(221, projection='3d')
X, T = np.meshgrid(x, t)
norm = colors.Normalize(vmin=u.min(), vmax=u.max())
surf = ax1.plot_surface(T, X, u, cmap=cm.plasma, 
                        norm=norm, linewidth=0, antialiased=True)
ax1.set_xlabel('Time', color='white')
ax1.set_ylabel('Space', color='white')
ax1.set_zlabel('u(x,t)', color='white')
ax1.set_title(f'Spatio-temporal Chaos (L = {L})', color='white')
ax1.set_facecolor('black')
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('w')
ax1.yaxis.pane.set_edgecolor('w')
ax1.zaxis.pane.set_edgecolor('w')
ax1.tick_params(colors='white')

# Heatmap (top right)
ax2 = fig.add_subplot(222)
heatmap = ax2.imshow(u, 
                     aspect='auto', 
                     extent=[0, L, t[-1], 0], 
                     cmap='inferno', 
                     interpolation='bicubic')
ax2.set_xlabel('Space', color='white')
ax2.set_ylabel('Time', color='white')
ax2.set_title('Space-time Evolution', color='white')
ax2.tick_params(colors='white')
cbar = plt.colorbar(heatmap, ax=ax2)
cbar.ax.yaxis.set_tick_params(color='white')
cbar.set_label('u(x,t)', color='white')
for tick in cbar.ax.get_yticklabels():
    tick.set_color('white')

# Lyapunov Exponents comparison (bottom left)
ax3 = fig.add_subplot(223)

# Find L values close to the simulation L for both datasets
L_close_periodic = min(data_periodic.keys(), key=lambda x: abs(x - L))
L_close_oddperiodic = min(data_oddperiodic.keys(), key=lambda x: abs(x - L))

lyap_indices = np.arange(1, 25)
width = 0.4

# Plot periodic case
bars1 = ax3.bar(lyap_indices - width/2, data_periodic[L_close_periodic], 
                width, color=plt.cm.viridis(np.linspace(0, 1, 24)), label=f'Periodic (L≈{L_close_periodic})')

# Plot odd-periodic case
bars2 = ax3.bar(lyap_indices + width/2, data_oddperiodic[L_close_oddperiodic], 
                width, color=plt.cm.plasma(np.linspace(0, 1, 24)), label=f'Odd-periodic (L≈{L_close_oddperiodic})')

ax3.axhline(y=0, color='white', linestyle='--', alpha=0.7)
ax3.set_xlabel('Index', color='white')
ax3.set_ylabel('λ (Lyapunov Exponent)', color='white')
ax3.set_title('Lyapunov Exponent Spectra Comparison', color='white')
ax3.tick_params(colors='white')
ax3.legend(facecolor='black', edgecolor='white', labelcolor='white')
for pos in ['top', 'bottom', 'right', 'left']:
    ax3.spines[pos].set_color('white')

# Domain size vs. First Lyapunov Exponent (bottom right)
ax4 = fig.add_subplot(224)

# Get data for plotting
Ls_periodic = sorted(data_periodic.keys())
lyap1_periodic = [data_periodic[L][0] for L in Ls_periodic]  # First (largest) exponent

Ls_oddperiodic = sorted(data_oddperiodic.keys())
lyap1_oddperiodic = [data_oddperiodic[L][0] for L in Ls_oddperiodic]  # First (largest) exponent

# Plot data
scatter1 = ax4.scatter(Ls_periodic, lyap1_periodic, 
                       c=lyap1_periodic, cmap='viridis', s=80, alpha=0.8, label='Periodic')
scatter2 = ax4.scatter(Ls_oddperiodic, lyap1_oddperiodic, 
                       c=lyap1_oddperiodic, cmap='plasma', s=80, alpha=0.8, label='Odd-periodic', marker='s')

# Highlight the domain size used for simulation
ax4.axvline(x=L, color='white', linestyle='--', alpha=0.7, label=f'Simulation L={L}')

ax4.set_xlabel('Domain Size (L)', color='white')
ax4.set_ylabel('Largest Lyapunov Exponent', color='white')
ax4.set_title('Chaos Strength vs. Domain Size', color='white')
ax4.tick_params(colors='white')
ax4.grid(True, alpha=0.3, color='white')
ax4.legend(facecolor='black', edgecolor='white', labelcolor='white')
for pos in ['top', 'bottom', 'right', 'left']:
    ax4.spines[pos].set_color('white')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the static visualization
plt.savefig('ks_visualization_static.png', dpi=300, facecolor='black')

# Create animation (timepoint snapshots)
fig_anim, ax_anim = plt.subplots(figsize=(10, 6), facecolor='black')
ax_anim.set_facecolor('black')

line, = ax_anim.plot(x, u[0], color='cyan', linewidth=2)
ax_anim.set_xlim(0, L)
ax_anim.set_ylim(np.min(u) - 0.5, np.max(u) + 0.5)
ax_anim.set_xlabel('Space', color='white')
ax_anim.set_ylabel('u(x,t)', color='white')
ax_anim.tick_params(colors='white')
title = ax_anim.set_title('Time = 0.00', color='white')
for pos in ['top', 'bottom', 'right', 'left']:
    ax_anim.spines[pos].set_color('white')

def update(frame):
    line.set_ydata(u[frame])
    title.set_text(f'Time = {t[frame]:.2f}')
    return line, title

ani = animation.FuncAnimation(fig_anim, update, frames=range(0, len(t), 4), 
                              blit=True, interval=50)
plt.tight_layout()

# Save the animation
ani.save('ks_dynamics.gif', writer='pillow', fps=15, dpi=100)

print("Visualization complete!")
print("Created static image 'ks_visualization_static.png' and animation 'ks_dynamics.gif'")
print(f"Used {'real' if data_loaded else 'synthetic'} Lyapunov exponent data")
