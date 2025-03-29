import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft

# Path to the saved model
model_path = r"c:\Kaustubh\ChaosPINN\baseline\results\ks_model"

# Create output directory for visualizations
vis_dir = r"c:\Kaustubh\ChaosPINN\baseline\results\visualizations"
os.makedirs(vis_dir, exist_ok=True)

print("Generating synthetic Kuramoto-Sivashinsky solution...")

# Solve the Kuramoto-Sivashinsky equation using pseudo-spectral method
def ks_solution(L=40, N=200, dt=0.1, T=10):
    """
    Solve the Kuramoto-Sivashinsky equation using pseudo-spectral method
    u_t + u*u_x + u_xx + u_xxxx = 0
    
    Parameters:
    L: Domain size [-L/2, L/2]
    N: Number of spatial points
    dt: Time step
    T: Total simulation time
    """
    # Spatial grid
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    dx = L/N
    
    # Wave numbers
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    
    # Initial condition: small amplitude cosine
    u = 0.1 * np.cos(x/16)
    
    # Time steps
    t_points = np.arange(0, T+dt, dt)
    nt = len(t_points)
    
    # Store solution
    u_solution = np.zeros((nt, N))
    u_solution[0] = u
    
    # Time stepping using pseudo-spectral method
    for i in range(1, nt):
        # FFT of u
        u_hat = fft(u)
        
        # Linear part: exact integration
        # exp(dt * (-k^2 + k^4))
        E = np.exp(dt * (-k**2 - k**4))
        
        # Nonlinear part: explicit Euler
        # -0.5 * d(u^2)/dx
        u_x = ifft(1j * k * u_hat).real
        nonlinear = -u * u_x
        nonlinear_hat = fft(nonlinear)
        
        # Combine linear and nonlinear parts
        u_hat = E * u_hat + dt * E * nonlinear_hat
        
        # Inverse FFT to get u at next time step
        u = ifft(u_hat).real
        u_solution[i] = u
    
    return x, t_points, u_solution

# Generate synthetic data
x, t, u_pred = ks_solution()
print("Synthetic data generated")

# 1. 2D Heatmap visualization
plt.figure(figsize=(12, 8))
plt.pcolormesh(x, t, u_pred, shading='auto', cmap='viridis')
plt.colorbar(label='u(x,t)')
plt.xlabel('x', fontsize=14)
plt.ylabel('t', fontsize=14)
plt.title('Kuramoto-Sivashinsky Equation Solution', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "ks_heatmap.png"), dpi=300)
print("Heatmap saved")

# 2. 3D Surface plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
X, T = np.meshgrid(x, t)
surf = ax.plot_surface(X, T, u_pred, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('t', fontsize=14)
ax.set_zlabel('u(x,t)', fontsize=14)
ax.set_title('Kuramoto-Sivashinsky Equation - 3D Surface', fontsize=16)
fig.colorbar(surf, shrink=0.6, aspect=10)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "ks_3d_surface.png"), dpi=300)
print("3D surface plot saved")

# 3. Multiple time slices
plt.figure(figsize=(12, 8))
time_slices = [0, 2, 4, 6, 8, 10]
for i, time in enumerate(time_slices):
    idx = min(int(time / 10 * (len(t) - 1)), len(t) - 1)
    plt.plot(x, u_pred[idx, :], label=f't = {time}', linewidth=2)
plt.xlabel('x', fontsize=14)
plt.ylabel('u(x,t)', fontsize=14)
plt.title('KS Solution at Different Time Steps', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "ks_time_slices.png"), dpi=300)
print("Time slices plot saved")

# 4. Space-time evolution animation
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, u_pred[0], 'k-', linewidth=2)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(u_pred.min() - 0.1, u_pred.max() + 0.1)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('u(x,t)', fontsize=14)
title = ax.set_title('t = 0.0', fontsize=16)
ax.grid(True)

def update(frame):
    line.set_ydata(u_pred[frame])
    title.set_text(f't = {t[frame]:.2f}')
    return line, title

ani = FuncAnimation(fig, update, frames=range(0, len(t), 2), blit=True)
ani.save(os.path.join(vis_dir, "ks_evolution.gif"), writer='pillow', fps=10, dpi=100)
print("Animation saved")

# 5. Contour plot
plt.figure(figsize=(12, 8))
contour = plt.contourf(X, T, u_pred, 50, cmap='viridis')
plt.colorbar(label='u(x,t)')
plt.xlabel('x', fontsize=14)
plt.ylabel('t', fontsize=14)
plt.title('Kuramoto-Sivashinsky Equation - Contour Plot', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "ks_contour.png"), dpi=300)
print("Contour plot saved")

print(f"All visualizations saved to {vis_dir}")