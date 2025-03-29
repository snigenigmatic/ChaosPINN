import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import tempfile
import re

# PINN model for Kuramoto-Sivashinsky equation
class KSPINN(nn.Module):
    def __init__(self, hidden_layers=4, neurons=50):
        super(KSPINN, self).__init__()
        
        # Neural network architecture
        layers = [nn.Linear(2, neurons), nn.Tanh()]  # Input: (x, t)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons, 1))  # Output: u(x, t)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t):
        # Concatenate x and t
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)
    
    def u_x(self, x, t):
        # Compute ∂u/∂x
        x.requires_grad_(True)
        u = self.forward(x, t)
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        return u_x
    
    def u_xx(self, x, t):
        # Compute ∂²u/∂x²
        x.requires_grad_(True)
        u_x = self.u_x(x, t)
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        return u_xx
    
    def u_xxxx(self, x, t):
        # Compute ∂⁴u/∂x⁴
        x.requires_grad_(True)
        u_xx = self.u_xx(x, t)
        u_xxx = torch.autograd.grad(
            u_xx, x, grad_outputs=torch.ones_like(u_xx),
            create_graph=True, retain_graph=True
        )[0]
        u_xxxx = torch.autograd.grad(
            u_xxx, x, grad_outputs=torch.ones_like(u_xxx),
            create_graph=True, retain_graph=True
        )[0]
        return u_xxxx
    
    def u_t(self, x, t):
        # Compute ∂u/∂t
        t.requires_grad_(True)
        u = self.forward(x, t)
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        return u_t
    
    def pde_residual(self, x, t):
        # Kuramoto-Sivashinsky equation: u_t + u*u_x + u_xx + u_xxxx = 0
        u = self.forward(x, t)
        u_t = self.u_t(x, t)
        u_x = self.u_x(x, t)
        u_xx = self.u_xx(x, t)
        u_xxxx = self.u_xxxx(x, t)
        
        # Compute the residual
        residual = u_t + u * u_x + u_xx + u_xxxx
        return residual

# Load and process Lyapunov exponent data from zip files
def load_lyapunov_data(downloads_dir, domain_type='periodic'):
    """
    Load Lyapunov exponent data from zip files in downloads directory
    
    Args:
        downloads_dir: Directory containing the zip files
        domain_type: 'periodic' or 'oddperiodic'
    
    Returns:
        Dictionary with L values as keys and arrays of Lyapunov exponents as values
    """
    if domain_type == 'periodic':
        zip_file = os.path.join(downloads_dir, 'lyapexpts_ksperiodic.zip')
    else:
        zip_file = os.path.join(downloads_dir, 'lyapexpts_ksoddperiodic.zip')
    
    lyapunov_data = {}
    
    # Create a temporary directory to extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the zip file
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            # Process the extracted files
            for file in os.listdir(temp_dir):
                if file.endswith('.txt'):
                    # Parse L value from filename (e.g., L097p4.txt -> 97.4)
                    match = re.match(r'L(\d+)p(\d+)\.txt', file)
                    if match:
                        integer_part = match.group(1)
                        decimal_part = match.group(2)
                        L = float(f"{integer_part}.{decimal_part}")
                        
                        # Load the Lyapunov exponents
                        file_path = os.path.join(temp_dir, file)
                        try:
                            lyap_exps = np.loadtxt(file_path)
                            lyapunov_data[L] = lyap_exps
                            print(f"Loaded Lyapunov exponents for L={L}")
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        except Exception as e:
            print(f"Error extracting or processing zip file {zip_file}: {e}")
    
    print(f"Loaded data for {len(lyapunov_data)} different L values")
    return lyapunov_data

# Check contents of the other zip file (Z6184566.zip)
def check_extra_zip_contents(downloads_dir):
    """
    Check and report contents of the extra zip file
    
    Args:
        downloads_dir: Directory containing the zip files
    """
    extra_zip = os.path.join(downloads_dir, 'Z6184566.zip')
    
    try:
        with zipfile.ZipFile(extra_zip, 'r') as zip_ref:
            print(f"\nContents of {extra_zip}:")
            for item in zip_ref.namelist():
                print(f"  - {item}")
            
            # If it contains MATLAB code, report that
            matlab_files = [f for f in zip_ref.namelist() if f.endswith('.m')]
            if matlab_files:
                print("\nFound MATLAB files that might contain the simulation code:")
                for m_file in matlab_files:
                    print(f"  - {m_file}")
    except Exception as e:
        print(f"Error examining {extra_zip}: {e}")

# Generate IC and boundary conditions from Lyapunov spectrum
def generate_initial_condition(L, lyap_exps, nx=100):
    """
    Generate an initial condition consistent with the Lyapunov spectrum
    
    Args:
        L: Domain length
        lyap_exps: Array of Lyapunov exponents
        nx: Number of spatial points
    
    Returns:
        x and u arrays representing the initial condition
    """
    x = np.linspace(0, L, nx)
    
    # Create a simple initial condition (could be refined based on the specific dynamics)
    # For demonstration, using a sum of sinusoids with amplitudes related to Lyapunov exponents
    u = np.zeros_like(x)
    for i, lyap in enumerate(lyap_exps[:min(5, len(lyap_exps))]):
        # Use positive Lyapunov exponents to determine the amplitudes
        amplitude = 0.1 * np.exp(abs(lyap))
        u += amplitude * np.sin((i+1) * np.pi * x / L)
    
    return x, u

# Training function with Lyapunov spectrum regularization
def train_ks_pinn(model, L, lyap_exps, n_epochs=10000, lr=1e-4, domain_type='periodic'):
    """
    Train the PINN model with regularization based on Lyapunov spectrum
    
    Args:
        model: KSPINN model
        L: Domain length
        lyap_exps: Array of Lyapunov exponents
        n_epochs: Number of training epochs
        lr: Learning rate
        domain_type: 'periodic' or 'oddperiodic'
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, verbose=True)
    
    # Generate training points
    nx, nt = 50, 40
    x_domain = torch.linspace(0, L, nx).reshape(-1, 1)
    t_domain = torch.linspace(0, 5.0, nt).reshape(-1, 1)
    
    # Create meshgrid for collocation points
    X, T = torch.meshgrid(x_domain.squeeze(), t_domain.squeeze(), indexing='ij')
    x_collocation = X.flatten().reshape(-1, 1)
    t_collocation = T.flatten().reshape(-1, 1)
    
    # Generate initial condition from Lyapunov spectrum
    x_ic, u_ic = generate_initial_condition(L, lyap_exps)
    x_ic_tensor = torch.tensor(x_ic, dtype=torch.float32).reshape(-1, 1)
    u_ic_tensor = torch.tensor(u_ic, dtype=torch.float32).reshape(-1, 1)
    t_ic_tensor = torch.zeros_like(x_ic_tensor)
    
    # Setup boundary conditions based on domain type
    if domain_type == 'periodic':
        # Periodic boundary conditions
        x_left = torch.zeros(nt, 1)
        x_right = torch.ones(nt, 1) * L
        t_boundary = t_domain.repeat(2, 1)
        
    else:  # 'oddperiodic'
        # Odd-periodic boundary (u = uxx = 0 at x=0,L)
        x_left = torch.zeros(nt, 1)
        x_right = torch.ones(nt, 1) * L
        t_boundary = t_domain.repeat(2, 1)
    
    # Loss weights
    w_pde = 1.0
    w_ic = 10.0
    w_bc = 10.0
    w_lyap = 0.1
    
    # Training loop
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # PDE residual loss
        residual = model.pde_residual(x_collocation, t_collocation)
        loss_pde = torch.mean(torch.square(residual))
        
        # Initial condition loss
        u_pred_ic = model(x_ic_tensor, t_ic_tensor)
        loss_ic = torch.mean(torch.square(u_pred_ic - u_ic_tensor))
        
        # Boundary condition loss
        if domain_type == 'periodic':
            # Enforce u(0,t) = u(L,t) and u_x(0,t) = u_x(L,t)
            u_left = model(x_left, t_domain)
            u_right = model(x_right, t_domain)
            
            u_x_left = model.u_x(x_left, t_domain)
            u_x_right = model.u_x(x_right, t_domain)
            
            loss_bc = torch.mean(torch.square(u_left - u_right)) + \
                      torch.mean(torch.square(u_x_left - u_x_right))
        else:  # 'oddperiodic'
            # Enforce u = 0 and u_xx = 0 at x=0,L
            u_boundary = model(torch.cat([x_left, x_right]), t_boundary)
            u_xx_boundary = model.u_xx(torch.cat([x_left, x_right]), t_boundary)
            
            loss_bc = torch.mean(torch.square(u_boundary)) + \
                      torch.mean(torch.square(u_xx_boundary))
        
        # Lyapunov regularization
        # In a complete implementation, this would involve estimating Lyapunov exponents
        # from the model predictions and comparing with the provided data
        # Here we use a placeholder
        loss_lyap = torch.tensor(0.0)
        
        # Total loss
        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc + w_lyap * loss_lyap
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # Record the loss
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, "
                 f"PDE: {loss_pde.item():.6f}, IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f}")
    
    return losses

# Function to visualize the solution
def visualize_solution(model, L, t_values=[0, 1, 2, 3, 4]):
    """
    Visualize the PINN solution for the Kuramoto-Sivashinsky equation
    
    Args:
        model: Trained KSPINN model
        L: Domain length
        t_values: List of time points to visualize
    """
    model.eval()
    nx = 200
    x = torch.linspace(0, L, nx).reshape(-1, 1)
    
    plt.figure(figsize=(12, 8))
    for t_val in t_values:
        t = torch.ones_like(x) * t_val
        u = model(x, t).detach().numpy()
        plt.plot(x.numpy(), u, label=f't = {t_val}')
    
    plt.title(f'Kuramoto-Sivashinsky Solution (L = {L})')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ks_solution.png')
    plt.close()
    
    # Also create a spacetime plot
    t_values = np.linspace(0, 5, 50)
    x_values = np.linspace(0, L, 100)
    
    T, X = np.meshgrid(t_values, x_values)
    U = np.zeros_like(T)
    
    for i, t_val in enumerate(t_values):
        t_tensor = torch.ones(len(x_values), 1) * t_val
        x_tensor = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
        
        u_pred = model(x_tensor, t_tensor).detach().numpy().flatten()
        U[:, i] = u_pred
    
    plt.figure(figsize=(12, 8))
    plt.contourf(T, X, U, 100, cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.title(f'Kuramoto-Sivashinsky Spacetime Plot (L = {L})')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.savefig('ks_spacetime.png')
    plt.close()

# Function to calculate Lyapunov exponents from PINN predictions
def calculate_lyapunov_from_pinn(model, L, domain_type='periodic', n_exponents=24):
    """
    Calculate approximate Lyapunov exponents from the PINN model
    
    Args:
        model: Trained KSPINN model
        L: Domain length
        domain_type: 'periodic' or 'oddperiodic'
        n_exponents: Number of Lyapunov exponents to calculate
        
    Returns:
        Array of estimated Lyapunov exponents
    """
    # This is a simplified placeholder - a full implementation would:
    # 1. Generate long trajectories using the PINN
    # 2. Implement Algorithm 1 from the paper to calculate Lyapunov exponents
    # 3. Compare with the provided dataset
    
    # Placeholder calculation
    model.eval()
    nx = 100
    x = torch.linspace(0, L, nx).reshape(-1, 1)
    t = torch.ones_like(x) * 10.0  # After sufficient time for dynamics to develop
    
    # Get the solution and its derivatives
    u = model(x, t).detach().numpy()
    
    # Simplified estimation (in practice would implement the full algorithm)
    lyap_estimates = np.zeros(n_exponents)
    for i in range(n_exponents):
        # Placeholder calculation based on Fourier modes
        k = i + 1
        wave_number = 2 * np.pi * k / L
        
        # For KS equation, the formula for Lyapunov exponents in linear regime is:
        # λ ≈ wave_number² - wave_number⁴
        # (simplified and not accurate for the full nonlinear dynamics)
        lyap_estimates[i] = wave_number**2 - wave_number**4
    
    # Sort in descending order (most positive first)
    lyap_estimates = np.sort(lyap_estimates)[::-1]
    
    return lyap_estimates

# Compare Lyapunov spectra
def compare_lyapunov_spectra(true_lyap, pred_lyap):
    """
    Compare true and predicted Lyapunov spectra
    
    Args:
        true_lyap: Ground truth Lyapunov exponents
        pred_lyap: Predicted Lyapunov exponents
    """
    plt.figure(figsize=(10, 6))
    plt.plot(true_lyap, 'b-o', label='Ground Truth')
    plt.plot(pred_lyap, 'r-x', label='PINN Estimated')
    plt.title('Lyapunov Exponent Spectrum Comparison')
    plt.xlabel('Index')
    plt.ylabel('Lyapunov Exponent')
    plt.legend()
    plt.grid(True)
    plt.savefig('lyapunov_comparison.png')
    plt.close()
    
    # Also plot the error
    plt.figure(figsize=(10, 6))
    min_len = min(len(true_lyap), len(pred_lyap))
    error = np.abs(true_lyap[:min_len] - pred_lyap[:min_len])
    plt.semilogy(error, 'k-o')
    plt.title('Absolute Error in Lyapunov Exponent Prediction')
    plt.xlabel('Index')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    plt.savefig('lyapunov_error.png')
    plt.close()

# Main function to execute the workflow
def main(downloads_dir, L=None, domain_type='periodic', hidden_layers=4, neurons=50, n_epochs=5000):
    """
    Main function to execute the full workflow
    
    Args:
        downloads_dir: Directory containing the zip files
        L: Domain length to analyze (if None, will use first available)
        domain_type: 'periodic' or 'oddperiodic'
        hidden_layers: Number of hidden layers in the PINN
        neurons: Number of neurons per hidden layer
        n_epochs: Number of training epochs
    """
    print(f"Processing {domain_type} domain Kuramoto-Sivashinsky data...")
    
    # Check extra zip file contents
    check_extra_zip_contents(downloads_dir)
    
    # Load Lyapunov data
    lyapunov_data = load_lyapunov_data(downloads_dir, domain_type)
    
    if not lyapunov_data:
        print("No Lyapunov data loaded. Please check the zip file paths.")
        return
    
    # If L is not specified, use the first available value
    if L is None:
        L = list(lyapunov_data.keys())[0]
        print(f"No L value specified. Using L={L}")
    
    # Check if we have data for the requested L value
    if L not in lyapunov_data:
        print(f"No Lyapunov data found for L={L}. Available values: {sorted(lyapunov_data.keys())}")
        return
    
    lyap_exps = lyapunov_data[L]
    print(f"Loaded {len(lyap_exps)} Lyapunov exponents for L={L}")
    
    # Create and train the model
    model = KSPINN(hidden_layers=hidden_layers, neurons=neurons)
    print(f"Training PINN model for {n_epochs} epochs...")
    losses = train_ks_pinn(model, L, lyap_exps, n_epochs=n_epochs, domain_type=domain_type)
    
    # Save the model
    torch.save(model.state_dict(), f'ks_pinn_model_{domain_type}_L{L}.pth')
    print(f"Model saved as ks_pinn_model_{domain_type}_L{L}.pth")
    
    # Visualize losses
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    print("Training loss plot saved as training_loss.png")
    
    # Visualize solution
    visualize_solution(model, L)
    print("Solution visualizations saved as ks_solution.png and ks_spacetime.png")
    
    # Calculate and compare Lyapunov exponents
    est_lyap_exps = calculate_lyapunov_from_pinn(model, L, domain_type)
    compare_lyapunov_spectra(lyap_exps, est_lyap_exps)
    print("Lyapunov spectra comparison saved as lyapunov_comparison.png and lyapunov_error.png")
    
    print("Analysis complete!")

if __name__ == "__main__":
    # Path to your downloads directory
    downloads_dir = os.path.expanduser("~/ChaosPINN/Data")
    
    # You can specify a particular L value or leave as None to use the first available
    L = None  # e.g., 97.4
    
    # Choose domain type: 'periodic' or 'oddperiodic'
    domain_type = 'periodic'
    
    # For faster execution on CPU, reduce these parameters
    hidden_layers = 3
    neurons = 40
    n_epochs = 1000  # Reduce for quicker testing, increase for better results
    
    main(downloads_dir, L, domain_type, hidden_layers, neurons, n_epochs)
