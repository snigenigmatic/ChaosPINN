import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

# Configure PyTorch to use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    # Set memory usage to grow as needed
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU")

# Define the Kuramoto-Sivashinsky equation
# u_t + u*u_x + u_xx + u_xxxx = 0

def pde(x, y):
    """
    x: (x, t)
    y: u(x, t)
    """
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_xxxx = dde.grad.jacobian(dy_xx, x, i=0, j=0)
    dy_xxxx = dde.grad.jacobian(dy_xxxx, x, i=0, j=0)
    
    # u_t + u*u_x + u_xx + u_xxxx = 0
    return dy_t + y * dy_x + dy_xx + dy_xxxx

# Domain and geometry
geom = dde.geometry.Interval(-20, 20)  # Spatial domain
timedomain = dde.geometry.TimeDomain(0, 10)  # Time domain
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Initial condition
def initial_condition(x):
    return np.cos(x[:, 0:1]) * 0.1

# Boundary conditions (periodic)
def boundary_condition(x, on_boundary):
    return on_boundary and (np.isclose(x[0], -20) or np.isclose(x[0], 20))

# Define the PDE problem
ic = dde.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)
bc = dde.PeriodicBC(geomtime, 0, boundary_condition)
data = dde.data.TimePDE(
    geomtime, pde, [ic, bc], num_domain=15000, num_boundary=400, num_initial=1000
)

# Define the network architecture
layer_size = [2] + [64] * 3 + [1]  # Adjusted for RTX 4060
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Define the model
model = dde.Model(data, net)

# Create output directory
output_dir = r"c:\Kaustubh\ChaosPINN\baseline\results"
os.makedirs(output_dir, exist_ok=True)

# Compile and train the model in stages with different learning rates
model.compile("adam", lr=1e-3)
model.train(epochs=5000)

# Reduce learning rate and continue training
model.compile("adam", lr=5e-4)
model.train(epochs=5000)

# Reduce learning rate further and finish training
model.compile("adam", lr=1e-4)
model.train(epochs=5000)

# Save the model
model.save(os.path.join(output_dir, "ks_model"))

# Plot the loss history
dde.utils.plot_loss_history(model.losshistory)
plt.savefig(os.path.join(output_dir, "ks_loss_history.png"))

# Visualization of the solution
def plot_solution(model):
    x = np.linspace(-20, 20, 200)
    t = np.linspace(0, 10, 100)
    X, T = np.meshgrid(x, t)
    X_flat = X.flatten()[:, None]
    T_flat = T.flatten()[:, None]
    
    # Predict in batches to avoid memory issues
    batch_size = 5000
    n_points = len(X_flat)
    u_pred = np.zeros((n_points, 1))
    
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        points = np.hstack((X_flat[i:end], T_flat[i:end]))
        u_pred[i:end] = model.predict(points)
    
    u_pred = u_pred.reshape(T.shape)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, T, u_pred, shading='auto', cmap='viridis')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Kuramoto-Sivashinsky Equation Solution')
    plt.savefig(os.path.join(output_dir, "ks_solution.png"))
    
    # Plot solution at different time steps
    plt.figure(figsize=(10, 6))
    for i, time in enumerate([0, 2, 5, 10]):
        idx = int(time / 10 * (len(t) - 1))
        plt.plot(x, u_pred[idx, :], label=f't = {time}')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('KS Solution at Different Times')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "ks_time_slices.png"))

# Plot the solution
plot_solution(model)

print("Training completed and results saved to:", output_dir)