import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class KuramotoSivashinskyNet(nn.Module):
    """Neural network model for solving the Kuramoto-Sivashinsky equation using physics-enhanced supervised learning."""

    def __init__(self, hidden_layers=4, hidden_dim=50):
        super(KuramotoSivashinskyNet, self).__init__()

        # Network architecture: spatial coordinates and time as input, solution as output
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Input: (x, t)
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh()) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, 1)   # Output: u(x, t)
        )

    def forward(self, x, t):
        # Combine spatial and temporal inputs
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


class KuramotoSivashinskyPhysicsEnhanced:
    """Physics-enhanced supervised learning approach for the Kuramoto-Sivashinsky equation."""

    def __init__(self, domain_size=32.0, device='cuda'):
        self.domain_size = domain_size
        # Use CPU if CUDA is not available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize neural network
        self.model = KuramotoSivashinskyNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=True
        )
        self.criterion = nn.MSELoss()  # Mean squared error loss for supervised learning
        self.physics_weight = 0.1      # Weight for physics-informed loss component

    def u(self, x, t):
        """Compute the solution u at points (x, t)."""
        x = x.to(self.device)
        t = t.to(self.device)
        return self.model(x, t)

    def compute_derivatives(self, x, t):
        """Compute the derivatives for the KS equation using automatic differentiation."""
        # Create tensor variables that require gradients
        x_var = x.clone().requires_grad_(True)
        t_var = t.clone().requires_grad_(True)
        
        # Forward pass to get u(x,t)
        u = self.model(x_var, t_var)
        
        # First derivatives
        u_x = torch.autograd.grad(
            u, x_var, 
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_t = torch.autograd.grad(
            u, t_var,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(
            u_x, x_var,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        
        # Fourth derivatives (computed through second derivatives)
        u_xxxx = torch.autograd.grad(
            u_xx, x_var,
            grad_outputs=torch.ones_like(u_xx),
            create_graph=True, retain_graph=True
        )[0]
        
        return u, u_t, u_x, u_xx, u_xxxx

    def ks_equation_residual(self, x, t):
        """Compute the residual of the KS equation: u_t + u*u_x + u_xx + u_xxxx = 0."""
        u, u_t, u_x, u_xx, u_xxxx = self.compute_derivatives(x, t)
        
        # Compute the residual of the KS equation
        residual = u_t + u * u_x + u_xx + u_xxxx
        
        return residual

    def generate_dataset(self, nx=100, nt=50, t_max=5.0):
        """Generate a labeled dataset using the pseudo-spectral method."""
        # Generate reference solution
        reference_solution = self.generate_reference_solution(nx=nx, nt=nt, t_max=t_max)
        
        # Create grid points
        x = np.linspace(0, self.domain_size, nx, endpoint=False)
        t = np.linspace(0, t_max, nt)
        
        # Create meshgrid for all (x,t) points
        X, T = np.meshgrid(x, t)
        
        # Flatten the arrays for training
        x_flat = X.flatten().reshape(-1, 1)
        t_flat = T.flatten().reshape(-1, 1)
        u_flat = reference_solution.flatten().reshape(-1, 1)
        
        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=self.device)
        t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=self.device)
        u_tensor = torch.tensor(u_flat, dtype=torch.float32, device=self.device)
        
        return x_tensor, t_tensor, u_tensor

    def generate_reference_solution(self, nx=100, nt=50, t_max=5.0):
        """
        Generate a reference solution using a pseudo-spectral method.
        This is a more traditional numerical approach to solving the KS equation.
        """
        # Space discretization
        dx = self.domain_size / nx
        x = np.linspace(0, self.domain_size, nx, endpoint=False)

        # Time discretization
        dt = t_max / (nt - 1)

        # Initial condition
        u = 0.1 * np.cos(x * 2 * np.pi / self.domain_size) + 0.01 * np.random.randn(nx)

        # Wavenumbers for spectral differentiation
        k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)

        # Precompute operators for spectral method
        L = k**2 - k**4  # Linear operator

        # Storage for the solution
        solution = np.zeros((nt, nx))
        solution[0, :] = u

        # Time-stepping using a semi-implicit scheme (ETDRK4)
        E = np.exp(dt * L)
        E_2 = np.exp(dt * L / 2)

        M = 16  # Number of points for complex means
        r = np.exp(2j * np.pi * (np.arange(M) + 0.5) / M)
        
        # Fix potential issues with LR calculation for ETDRK4
        LR = np.outer(dt * np.ones(M), L)
        LR = LR.reshape(M, -1)

        # Precomputed coefficients for ETDRK4
        Q = dt * np.mean(((np.exp(LR/2) - 1) / (LR + 1e-16)), axis=0)
        f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / ((LR + 1e-16)**3), axis=0)
        f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / ((LR + 1e-16)**3), axis=0)
        f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / ((LR + 1e-16)**3), axis=0)

        # Main time-stepping loop
        for n in range(1, nt):
            # Compute the nonlinear term
            u_hat = np.fft.fft(u)
            N = -0.5 * np.fft.fft(u**2)

            # ETDRK4 scheme with better numerical stability
            a = E_2 * u_hat + Q * N
            Na = -0.5 * np.fft.fft((np.fft.ifft(a))**2)
            b = E_2 * u_hat + Q * Na
            Nb = -0.5 * np.fft.fft((np.fft.ifft(b))**2)
            c = E_2 * a + Q * (2*Nb - N)
            Nc = -0.5 * np.fft.fft((np.fft.ifft(c))**2)

            u_hat = E * u_hat + N*f1 + 2*(Na + Nb)*f2 + Nc*f3
            u = np.real(np.fft.ifft(u_hat))

            solution[n, :] = u

        return solution

    def generate_physics_points(self, n_physics=1000):
        """Generate random collocation points for evaluating the physics residual."""
        # Random points in the domain for physics evaluation
        x = torch.rand(n_physics, 1, device=self.device) * self.domain_size
        t = torch.rand(n_physics, 1, device=self.device) * 5.0  # Assuming t_max = 5.0
        
        return x, t

    def create_data_loaders(self, x, t, u, batch_size=1024, train_split=0.8):
        """Create training and validation dataloaders."""
        # Create dataset indices
        dataset_size = x.shape[0]
        indices = torch.randperm(dataset_size)
        
        # Split into train and validation
        train_size = int(train_split * dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create tensor datasets
        train_data = torch.utils.data.TensorDataset(
            x[train_indices], t[train_indices], u[train_indices]
        )
        val_data = torch.utils.data.TensorDataset(
            x[val_indices], t[val_indices], u[val_indices]
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader

    def train(self, epochs=1000, batch_size=1024, nx=100, nt=50, t_max=5.0, n_physics=1000):
        """Train the model using physics-enhanced supervised learning."""
        print("Generating training data...")
        x, t, u = self.generate_dataset(nx=nx, nt=nt, t_max=t_max)
        
        print(f"Dataset size: {x.shape[0]} points")
        train_loader, val_loader = self.create_data_loaders(x, t, u, batch_size=batch_size)
        
        train_losses = []
        val_losses = []
        physics_losses = []
        
        print("Starting training...")
        for epoch in tqdm(range(epochs)):
            # Training phase
            self.model.train()
            train_loss = 0.0
            physics_loss_epoch = 0.0
            
            for batch_x, batch_t, batch_u in train_loader:
                self.optimizer.zero_grad()
                
                # 1. Data loss - supervised component
                outputs = self.u(batch_x, batch_t)
                data_loss = self.criterion(outputs, batch_u)
                
                # 2. Physics loss - random collocation points
                x_phys, t_phys = self.generate_physics_points(n_physics=n_physics//10)  # Smaller batch
                physics_residuals = self.ks_equation_residual(x_phys, t_phys)
                physics_loss = torch.mean(physics_residuals**2)
                
                # Combined loss
                loss = data_loss + self.physics_weight * physics_loss
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += data_loss.item() * batch_x.size(0)
                physics_loss_epoch += physics_loss.item()
            
            train_loss /= len(train_loader.dataset)
            physics_loss_epoch /= len(train_loader)
            train_losses.append(train_loss)
            physics_losses.append(physics_loss_epoch)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_t, batch_u in val_loader:
                    outputs = self.u(batch_x, batch_t)
                    loss = self.criterion(outputs, batch_u)
                    val_loss += loss.item() * batch_x.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Print progress and update scheduler
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6e}, Physics Loss = {physics_loss_epoch:.6e}, Val Loss = {val_loss:.6e}")
                self.scheduler.step(val_loss)
            
            # Early stopping condition
            if train_loss < 1e-6:
                print(f"Converged at epoch {epoch}")
                break
        
        return train_losses, val_losses, physics_losses

    def save_model(self, path="ks_physics_enhanced_model.pt"):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'domain_size': self.domain_size
        }, path)

    def load_model(self, path="ks_physics_enhanced_model.pt"):
        """Load a trained model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.domain_size = checkpoint['domain_size']

    def evaluate(self, nx=100, nt=50, t_max=5.0):
        """Generate a solution grid for visualization."""
        self.model.eval()

        x = torch.linspace(0, self.domain_size, nx, device=self.device).reshape(-1, 1)
        t = torch.linspace(0, t_max, nt, device=self.device)

        solution = torch.zeros((nt, nx), device=self.device)

        with torch.no_grad():
            for i, t_i in enumerate(t):
                t_points = t_i.repeat(nx, 1)
                solution[i, :] = self.u(x, t_points).squeeze()

        return solution.cpu().numpy(), x.cpu().numpy(), t.cpu().numpy()
    
    def evaluate_physics_residual(self, nx=50, nt=20, t_max=5.0):
        """Evaluate the physics residual across the domain."""
        self.model.eval()
        
        # Create grid for evaluation
        x = torch.linspace(0, self.domain_size, nx, device=self.device).reshape(-1, 1)
        t = torch.linspace(0, t_max, nt, device=self.device).reshape(-1, 1)
        
        # Create meshgrid
        T, X = torch.meshgrid(t.squeeze(), x.squeeze(), indexing='ij')
        
        # Reshape for evaluation
        x_grid = X.reshape(-1, 1).requires_grad_(True)
        t_grid = T.reshape(-1, 1).requires_grad_(True)
        
        # Compute residuals
        residuals = self.ks_equation_residual(x_grid, t_grid)
        
        # Reshape back to grid
        residual_grid = residuals.detach().cpu().numpy().reshape(nt, nx)
        
        return residual_grid, X.cpu().numpy(), T.cpu().numpy()


def visualize_solution(solution, x, t, title="Kuramoto-Sivashinsky Solution"):
    """Plot the solution as a 2D heatmap."""
    X, T = np.meshgrid(x.flatten(), t)

    plt.figure(figsize=(10, 6))
    plt.contourf(T, X, solution, 100, cmap='viridis')
    plt.colorbar(label='u(x,t)')
    plt.title(title)
    plt.xlabel('Time (t)')
    plt.ylabel('Space (x)')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()


def compare_solutions(model_solution, reference_solution, x, t):
    """Compare the model and reference solutions visually."""
    # Plot the solutions
    visualize_solution(model_solution, x, t, "Physics-Enhanced Model Solution")
    visualize_solution(reference_solution, x, t, "Reference Solution")

    # Plot the absolute difference
    diff = np.abs(model_solution - reference_solution)

    X, T = np.meshgrid(x.flatten(), t)

    plt.figure(figsize=(10, 6))
    plt.contourf(T, X, diff, 100, cmap='hot')
    plt.colorbar(label='Absolute Error')
    plt.title("Absolute Error")
    plt.xlabel('Time (t)')
    plt.ylabel('Space (x)')
    plt.tight_layout()
    plt.savefig("error_map.png")
    plt.close()

    # Plot a few time slices
    time_slices = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]

    plt.figure(figsize=(12, 8))
    for i, tidx in enumerate(time_slices):
        plt.subplot(len(time_slices), 1, i+1)
        plt.plot(x, model_solution[tidx], 'b-', label='Physics-Enhanced NN')
        plt.plot(x, reference_solution[tidx], 'r--', label='Reference')
        plt.title(f"t = {t[tidx]:.2f}")
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.grid(True)
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig("time_slices_comparison.png")
    plt.close()


def visualize_physics_residual(residual, x, t):
    """Visualize the physics residual across the domain."""
    plt.figure(figsize=(10, 6))
    plt.contourf(t, x, residual, 100, cmap='coolwarm')
    plt.colorbar(label='Residual')
    plt.title("Physics Residual: u_t + u*u_x + u_xx + u_xxxx")
    plt.xlabel('Time (t)')
    plt.ylabel('Space (x)')
    plt.tight_layout()
    plt.savefig("physics_residual.png")
    plt.close()


def test_against_reference_solution(model_solution, reference_solution, threshold=0.1):
    """
    Test the model solution against a reference solution.
    Returns True if the model performs adequately, False otherwise.
    """
    # Calculate the mean absolute error
    abs_error = np.abs(model_solution - reference_solution)
    mae = np.mean(abs_error)

    print(f"Mean Absolute Error: {mae:.6f}")

    # Calculate the normalized root mean square error
    rmse = np.sqrt(np.mean((model_solution - reference_solution)**2))
    norm_rmse = rmse / (np.max(reference_solution) - np.min(reference_solution))

    print(f"Normalized RMSE: {norm_rmse:.6f}")

    # Calculate the temporal correlation at each spatial point
    nt, nx = model_solution.shape
    temporal_correlations = np.zeros(nx)

    for i in range(nx):
        correlation = np.corrcoef(model_solution[:, i], reference_solution[:, i])[0, 1]
        temporal_correlations[i] = correlation if not np.isnan(correlation) else 0

    avg_temporal_correlation = np.mean(temporal_correlations)
    print(f"Average Temporal Correlation: {avg_temporal_correlation:.6f}")

    # Calculate the spatial correlation at each time point
    spatial_correlations = np.zeros(nt)

    for i in range(nt):
        correlation = np.corrcoef(model_solution[i, :], reference_solution[i, :])[0, 1]
        spatial_correlations[i] = correlation if not np.isnan(correlation) else 0

    avg_spatial_correlation = np.mean(spatial_correlations)
    print(f"Average Spatial Correlation: {avg_spatial_correlation:.6f}")

    # Overall score
    overall_score = (
        (1 - min(norm_rmse, 1)) * 0.4 +
        avg_temporal_correlation * 0.3 +
        avg_spatial_correlation * 0.3
    )

    print(f"Overall Score: {overall_score:.6f} (higher is better)")

    return overall_score > 0.7  # Threshold for passing the test


def plot_training_history(train_losses, val_losses, physics_losses):
    """Plot the training and validation loss history."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses, label='Training Loss')
    plt.semilogy(val_losses, label='Validation Loss')
    plt.semilogy(physics_losses, label='Physics Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Physics Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history.png")
    plt.close()


def main():
    """Main function to run the physics-enhanced KS equation solver and tests."""
    print("Starting Kuramoto-Sivashinsky physics-enhanced learning")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create the supervised learning model
    domain_size = 32.0  # Domain size is important for KS dynamics
    ks_model = KuramotoSivashinskyPhysicsEnhanced(domain_size=domain_size)

    # Train the model with physics-enhanced supervised learning
    print("Training the model with physics-enhanced supervised learning...")
    train_losses, val_losses, physics_losses = ks_model.train(epochs=500, batch_size=1024, nx=100, nt=50)

    # Save the trained model
    ks_model.save_model()

    # Generate the model solution
    nx, nt = 128, 50
    t_max = 5.0
    model_solution, x_grid, t_grid = ks_model.evaluate(nx=nx, nt=nt, t_max=t_max)

    print("Generating reference solution for comparison...")
    # Generate a reference solution
    reference_solution = ks_model.generate_reference_solution(
        nx=nx, nt=nt, t_max=t_max
    )

    # Visualize and compare the solutions
    print("Comparing solutions...")
    compare_solutions(model_solution, reference_solution, x_grid, t_grid)

    # Evaluate and visualize physics residual
    print("Evaluating physics residual...")
    residual, X, T = ks_model.evaluate_physics_residual(nx=50, nt=20, t_max=t_max)
    visualize_physics_residual(residual, X, T)

    # Test against the reference solution
    print("\nTesting model against reference solution:")
    test_result = test_against_reference_solution(model_solution, reference_solution)
    print(f"Test {'passed' if test_result else 'failed'}")

    # Plot the loss history
    plot_training_history(train_losses, val_losses, physics_losses)

    print("\nResults saved as PNG files.")
    print("Model saved as 'ks_physics_enhanced_model.pt'")


if __name__ == "__main__":
    main()
