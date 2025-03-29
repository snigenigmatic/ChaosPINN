import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import os

# Set the random seed for reproducibility
np.random.seed(1234)
dde.config.set_random_seed(1234)

# Physical parameters
L = 32  # Domain length
T = 50  # Total simulation time
alpha = 1.0  # Coefficient for the equation

# Create the computational domain
geom_x = dde.geometry.Interval(0, L)
geom_time = dde.geometry.TimeDomain(0, T)
geom = dde.geometry.GeometryXTime(geom_x, geom_time)

# Boundary conditions (periodic in space)
def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)

# Initial condition
def initial_condition(x):
    # Starting with a cosine wave with some noise
    return 0.1*np.cos(2*np.pi*x[:, 0:1]/L) + 0.01*np.random.randn(*x[:, 0:1].shape)

# Define the PDE residual
def pde(x, y):
    """Define the PDE residual for Kuramoto-Sivashinsky equation.
    
    u_t + u*u_x + u_xx + u_xxxx = 0
    """
    y_x = dde.grad.jacobian(y, x, i=0, j=0)
    y_t = dde.grad.jacobian(y, x, i=0, j=1)
    y_xx = dde.grad.hessian(y, x, i=0, j=0)
    
    # Computing the fourth derivative
    y_xxx = dde.grad.jacobian(y_xx, x, i=0, j=0)
    y_xxxx = dde.grad.jacobian(y_xxx, x, i=0, j=0)
    
    # KS equation: u_t + u*u_x + u_xx + u_xxxx = 0
    return y_t + y * y_x + y_xx + y_xxxx

# Create the boundary conditions
# Periodic BC: u(0,t) = u(L,t) and u_x(0,t) = u_x(L,t)
bc_l = dde.icbc.PeriodicBC(geom, 0, boundary_l, boundary_r)

# Initial condition
ic = dde.icbc.IC(geom, initial_condition, lambda _, on_initial: on_initial)

# Create data for training
data = dde.data.TimePDE(
    geom,
    pde,
    [bc_l, ic],
    num_domain=10000,
    num_boundary=500,
    num_initial=500,
    solution=None,
    num_test=10000,
)

# Create the network
layer_size = [2] + [50] * 5 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Create the model with Adam optimizer
model = dde.Model(data, net)

# Define learning rate scheduler
def learning_rate_schedule(epoch):
    if epoch < 2000:
        return 1e-3
    elif epoch < 5000:
        return 1e-4
    else:
        return 1e-5

# Define the training hyperparameters
loss_weights = [1, 10, 10]  # Weights for PDE, BC, and IC
model.compile("adam", lr=1e-3, loss_weights=loss_weights)

# Enable checkpoint saving
ckpt_dir = "./model_checkpoints"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Select a variable to monitor for checkpointing
variable = dde.callbacks.VariableValue(
    "loss", period=1000, filename=os.path.join(ckpt_dir, "loss_history.dat")
)

# Create a ModelCheckpoint callback
model_checkpoint = dde.callbacks.ModelCheckpoint(
    os.path.join(ckpt_dir, "model.ckpt"),
    save_better_only=True,
    period=1000
)

# Define LBFGS callback
lbfgs = dde.callbacks.LossHistory()
lbfgs_callback = dde.callbacks.LBFGSCallback(
    minimum_iterations=500,
    maximum_iterations=2000,
    tolerance=1e-4,
    factor=0.3,
    frequency=100,
)

# Train the model with adaptive learning rate
losshistory, train_state = model.train(
    iterations=10000,
    callbacks=[
        variable,
        model_checkpoint,
        dde.callbacks.LearningRateScheduler(learning_rate_schedule, verbose=1),
    ],
    display_every=100
)

# Fine-tune with L-BFGS
model.compile("L-BFGS")
model.train(display_every=100, callbacks=[lbfgs])

# Visualize the results
def visualize_solution(model):
    # Generate a grid of points for visualization
    t_values = np.linspace(0, T, 101)
    x_values = np.linspace(0, L, 256)
    
    # Create meshgrid
    X, T = np.meshgrid(x_values, t_values)
    X_flat = X.flatten()[:, None]
    T_flat = T.flatten()[:, None]
    points = np.hstack((X_flat, T_flat))
    
    # Predict solution at grid points
    u_pred = model.predict(points)
    u_pred = u_pred.reshape(X.shape)
    
    # Create a figure for the heatmap
    plt.figure(figsize=(12, 8))
    plt.contourf(X, T, u_pred, 100, cmap='RdBu')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('PINN Solution of 1D Kuramoto-Sivashinsky Equation')
    plt.savefig('ks_solution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create an animation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Kuramoto-Sivashinsky Equation Solution')
    line, = ax.plot([], [], 'r-', lw=2)
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        y = u_pred[i]
        line.set_data(x_values, y)
        ax.set_title(f'Time t = {t_values[i]:.2f}')
        return line,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t_values), interval=100, blit=True)
    anim.save('ks_solution_animation.mp4', writer='ffmpeg', fps=10, dpi=200)
    
    plt.close()
    
    # Save specific time snapshots
    for t_idx in [0, int(len(t_values)/4), int(len(t_values)/2), int(3*len(t_values)/4), -1]:
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, u_pred[t_idx])
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f'Snapshot at t = {t_values[t_idx]:.2f}')
        plt.grid(True)
        plt.savefig(f'ks_snapshot_t{t_values[t_idx]:.2f}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Evaluate and visualize the solution
visualize_solution(model)

# Error analysis function
def compute_error_metrics(model):
    """Compute error metrics for the trained model."""
    # We'll need a reference solution to calculate the error
    # For KS, we can generate a high-fidelity numerical solution using spectral methods
    # or compare with test data points if available
    
    # For this example, we'll generate error statistics based on PDE residual
    t_values = np.linspace(0, T, 101)
    x_values = np.linspace(0, L, 256)
    
    X, T = np.meshgrid(x_values, t_values)
    X_flat = X.flatten()[:, None]
    T_flat = T.flatten()[:, None]
    points = np.hstack((X_flat, T_flat))
    
    # Calculate PDE residual at each point
    pde_residuals = model.predict(points, operator=pde)
    
    # Compute error statistics
    mean_residual = np.mean(np.abs(pde_residuals))
    max_residual = np.max(np.abs(pde_residuals))
    
    # Create histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(np.abs(pde_residuals), bins=50, alpha=0.7)
    plt.xlabel('Absolute PDE Residual')
    plt.ylabel('Frequency')
    plt.title('Distribution of PDE Residuals')
    plt.grid(True)
    plt.savefig('ks_residual_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap of residuals
    residual_map = pde_residuals.reshape(X.shape)
    plt.figure(figsize=(12, 8))
    plt.contourf(X, T, np.abs(residual_map), 100, cmap='viridis')
    plt.colorbar(label='|PDE Residual|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('PDE Residual Map')
    plt.savefig('ks_residual_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "mean_residual": mean_residual,
        "max_residual": max_residual
    }

# Compute error metrics
error_metrics = compute_error_metrics(model)
print(f"Mean PDE Residual: {error_metrics['mean_residual']}")
print(f"Max PDE Residual: {error_metrics['max_residual']}")

# Save the trained model
model.save("ks_equation_model.dat")

# Write a summary report
with open("KS_PINN_Summary.md", "w") as f:
    f.write("# Kuramoto-Sivashinsky Equation PINN Solution Summary\n\n")
    f.write(f"## Model Architecture\n")
    f.write(f"- Network: FNN with {len(layer_size)-2} hidden layers\n")
    f.write(f"- Hidden layers size: {layer_size[1]}\n")
    f.write(f"- Activation function: {activation}\n\n")
    f.write(f"## Training Parameters\n")
    f.write(f"- Domain: x ∈ [0, {L}], t ∈ [0, {T}]\n")
    f.write(f"- Training points: {data.num_train} total\n")
    f.write(f"  - Domain points: {data.num_domain}\n")
    f.write(f"  - Boundary points: {data.num_boundary}\n")
    f.write(f"  - Initial points: {data.num_initial}\n\n")
    f.write(f"## Error Analysis\n")
    f.write(f"- Mean PDE Residual: {error_metrics['mean_residual']}\n")
    f.write(f"- Max PDE Residual: {error_metrics['max_residual']}\n\n")
    f.write(f"## Results\n")
    f.write(f"See the attached visualization files:\n")
    f.write(f"- ks_solution_heatmap.png: 2D visualization of the solution\n")
    f.write(f"- ks_solution_animation.mp4: Animation of the solution evolving over time\n")
    f.write(f"- ks_snapshot_t*.png: Snapshots of the solution at specific time points\n")
    f.write(f"- ks_residual_histogram.png: Distribution of PDE residuals\n")
    f.write(f"- ks_residual_map.png: Spatial and temporal distribution of residuals\n")
