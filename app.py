import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import time
import os

# Fix for PyTorch custom classes and Streamlit watcher
import sys

from model_fit import KuramotoSivashinskySupervisedLearning, visualize_solution, compare_solutions, test_against_reference_solution

if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    st.sidebar.success(f"CUDA is available: {device_name}")
    st.sidebar.info(f"CUDA Version: {cuda_version}, Device Count: {device_count}")
else:
    st.sidebar.warning("CUDA is not available. Using CPU instead.")
    st.sidebar.info("To use GPU acceleration, ensure you have compatible NVIDIA drivers and PyTorch with CUDA support installed.")



# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTitle {
        color: #1E88E5;
        font-size: 3rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_losses' not in st.session_state:
    st.session_state.train_losses = []
if 'val_losses' not in st.session_state:
    st.session_state.val_losses = []
if 'model_solution' not in st.session_state:
    st.session_state.model_solution = None
if 'reference_solution' not in st.session_state:
    st.session_state.reference_solution = None
if 'x_grid' not in st.session_state:
    st.session_state.x_grid = None
if 't_grid' not in st.session_state:
    st.session_state.t_grid = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False
if 'test_results' not in st.session_state:
    st.session_state.test_results = None

# Header
st.title("ChaosPINN: Kuramoto-Sivashinsky Equation Solver")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Model Parameters")
    
    # Domain parameters
    domain_size = st.slider("Domain Size", 16.0, 64.0, 32.0, 8.0)
    
    # Training parameters
    st.subheader("Training Parameters")
    epochs = st.slider("Number of Epochs", 50, 1000, 200)
    batch_size = st.slider("Batch Size", 256, 2048, 1024, 256)
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
    
    # Dataset parameters
    st.subheader("Dataset Parameters")
    nx = st.slider("Spatial Points (nx)", 50, 200, 100)
    nt = st.slider("Time Points (nt)", 20, 100, 50)
    t_max = st.slider("Max Time", 1.0, 10.0, 5.0, 0.5)
    
    # Model architecture
    st.subheader("Network Architecture")
    hidden_layers = st.slider("Hidden Layers", 2, 8, 4)
    hidden_dim = st.slider("Hidden Dimension", 20, 100, 50, 10)
    
    # Action buttons
    st.markdown("---")
    train_button = st.button("Train Model", type="primary")
    evaluate_button = st.button("Evaluate Model")
    
    # Load/Save model
    # Model management section in sidebar
    st.markdown("---")
    st.subheader("Model Management")
    save_model = st.button("Save Model")
    
    # Set default model path
    default_model_path = r"C:\Kaustubh\ChaosPINN\ks_model_20250330_040538.pt"
    
    model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
    if model_files:
        selected_model = st.selectbox("Select Model", model_files, 
                                     index=model_files.index(os.path.basename(default_model_path)) if os.path.basename(default_model_path) in model_files else 0)
        load_model = st.button("Load Selected Model")
    else:
        st.info(f"No saved models found in current directory. Will use default model.")
        selected_model = default_model_path
        load_model = st.button("Load Default Model")
    
    # Model loading code
    if load_model:
        with st.spinner(f"Loading model {selected_model}..."):
            # Always use CPU first for compatibility
            device = 'cpu'
            
            # Initialize the model
            st.session_state.model = KuramotoSivashinskySupervisedLearning(device=device)
            
            # Determine the full path to the model
            model_path = selected_model if os.path.isabs(selected_model) else os.path.join(os.getcwd(), selected_model)
            
            # Check if the file exists
            if not os.path.exists(model_path) and selected_model != default_model_path:
                st.warning(f"Selected model not found. Falling back to default model.")
                model_path = default_model_path
            
            # Try loading the model
            try:
                st.session_state.model.load_model(path=model_path, map_location='cpu')
                st.sidebar.success(f"Model loaded successfully")
                st.session_state.training_complete = True
                
                # Move to GPU if available and requested
                if torch.cuda.is_available():
                    st.info(f"Moving model to GPU: {torch.cuda.get_device_name(0)}")
                    st.session_state.model.model.to('cuda')
                    st.session_state.model.device = 'cuda'
                    
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
    else:
        st.info("No saved models found")
        load_model = False

# Main content area
col1, col2 = st.columns([2, 1])

# Training function
# Training function
def train_model():
    with st.spinner("Initializing model..."):
        # Force CUDA if available
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.empty_cache()  # Clear GPU memory
            st.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            st.warning("CUDA not available, using CPU instead")
            
        st.session_state.model = KuramotoSivashinskySupervisedLearning(domain_size=domain_size, device=device)

        # Modify the model architecture if needed
        st.session_state.model.model = st.session_state.model.model.__class__(
            hidden_layers=hidden_layers, 
            hidden_dim=hidden_dim
        ).to(device)
        
        # Update optimizer with new learning rate
        st.session_state.model.optimizer = torch.optim.Adam(
            st.session_state.model.model.parameters(), 
            lr=learning_rate
        )
        
        # Update scheduler
        st.session_state.model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            st.session_state.model.optimizer, patience=5, factor=0.5, verbose=True
        )
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Train the model with progress updates
    status_text.text("Generating training data...")
    x, t, u = st.session_state.model.generate_dataset(nx=nx, nt=nt, t_max=t_max)
    
    train_loader, val_loader = st.session_state.model.create_data_loaders(
        x, t, u, batch_size=batch_size
    )
    
    st.session_state.train_losses = []
    st.session_state.val_losses = []
    
    status_text.text("Starting training...")
    
    # Training loop with progress updates
    for epoch in range(epochs):
        # Training phase
        st.session_state.model.model.train()
        train_loss = 0.0
        
        for batch_x, batch_t, batch_u in train_loader:
            st.session_state.model.optimizer.zero_grad()
            
            # Forward pass
            outputs = st.session_state.model.u(batch_x, batch_t)
            
            # Compute loss
            loss = st.session_state.model.criterion(outputs, batch_u)
            
            # Backward pass and optimize
            loss.backward()
            st.session_state.model.optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        
        train_loss /= len(train_loader.dataset)
        st.session_state.train_losses.append(train_loss)
        
        # Validation phase
        st.session_state.model.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_t, batch_u in val_loader:
                outputs = st.session_state.model.u(batch_x, batch_t)
                loss = st.session_state.model.criterion(outputs, batch_u)
                val_loss += loss.item() * batch_x.size(0)
        
        val_loss /= len(val_loader.dataset)
        st.session_state.val_losses.append(val_loss)
        
        # Update scheduler
        st.session_state.model.scheduler.step(val_loss)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}")
        
        # Early stopping condition
        if train_loss < 1e-6:
            status_text.text(f"Converged at epoch {epoch+1}")
            break
    
    status_text.text("Training complete!")
    st.session_state.training_complete = True

# Evaluation function
def evaluate_model():
    if st.session_state.model is None:
        st.error("Please train or load a model first")
        return
    
    with st.spinner("Evaluating model..."):
        # Generate the model solution
        st.session_state.model_solution, st.session_state.x_grid, st.session_state.t_grid = st.session_state.model.evaluate(
            nx=nx, nt=nt, t_max=t_max
        )
        
        # Generate reference solution
        st.session_state.reference_solution = st.session_state.model.generate_reference_solution(
            nx=nx, nt=nt, t_max=t_max
        )
        
        # Test against reference solution
        mae, norm_rmse, avg_temporal_corr, avg_spatial_corr, overall_score, passed = test_against_reference_solution(
            st.session_state.model_solution, 
            st.session_state.reference_solution,
            return_metrics=True
        )
        
        st.session_state.test_results = {
            'mae': mae,
            'norm_rmse': norm_rmse,
            'avg_temporal_corr': avg_temporal_corr,
            'avg_spatial_corr': avg_spatial_corr,
            'overall_score': overall_score,
            'passed': passed
        }
        
        st.session_state.evaluation_complete = True

# Handle button actions
if train_button:
    train_model()

if evaluate_button:
    evaluate_model()

if save_model and st.session_state.model is not None:
    # Save only as ks_physics_enhanced.pt instead of timestamped versions
    model_name = "ks_physics_enhanced.pt"
    st.session_state.model.save_model(path=model_name)
    st.sidebar.success(f"Model saved as {model_name}")

if load_model and 'selected_model' in locals():
    with st.spinner(f"Loading model {selected_model}..."):
        # Force CUDA if available, otherwise use CPU
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.empty_cache()  # Clear GPU memory
            st.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            st.warning("CUDA not available, using CPU instead")
            
        st.session_state.model = KuramotoSivashinskySupervisedLearning(device=device)
        
        # Add map_location to handle models saved on different devices
        try:
            st.session_state.model.load_model(path=selected_model, map_location=device)
            st.sidebar.success(f"Model {selected_model} loaded successfully")
            st.session_state.training_complete = True
        except RuntimeError as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Trying to load with CPU mapping...")
            try:
                st.session_state.model.load_model(path=selected_model, map_location='cpu')
                st.sidebar.success(f"Model {selected_model} loaded successfully (mapped to CPU)")
                st.session_state.training_complete = True
            except Exception as e2:
                st.error(f"Failed to load model: {str(e2)}")

# Display visualization based on selected plot type
with col1:
    st.header("Model Visualization")
    
    if st.session_state.training_complete:
        # Always show the visualization button
        load_viz_button = st.button("Load/Reload Visualizations")
        
        if load_viz_button:
            with st.spinner("Loading visualizations..."):
                try:
                    # Generate the model solution
                    st.session_state.model_solution, st.session_state.x_grid, st.session_state.t_grid = st.session_state.model.evaluate(
                        nx=nx, nt=nt, t_max=t_max
                    )
                    
                    # Generate reference solution
                    st.session_state.reference_solution = st.session_state.model.generate_reference_solution(
                        nx=nx, nt=nt, t_max=t_max
                    )
                    
                    # Test against reference solution
                    mae, norm_rmse, avg_temporal_corr, avg_spatial_corr, overall_score, passed = test_against_reference_solution(
                        st.session_state.model_solution, 
                        st.session_state.reference_solution,
                        return_metrics=True
                    )
                    
                    st.session_state.test_results = {
                        'mae': mae,
                        'norm_rmse': norm_rmse,
                        'avg_temporal_corr': avg_temporal_corr,
                        'avg_spatial_corr': avg_spatial_corr,
                        'overall_score': overall_score,
                        'passed': passed
                    }
                    
                    st.session_state.evaluation_complete = True
                    st.success("Visualizations loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading visualizations: {str(e)}")
                    
                    # Try to fix parameter shape mismatch
                    if "copying a param with shape" in str(e):
                        st.warning("Attempting to fix parameter shape mismatch...")
                        try:
                            # Recreate the model with adjusted parameters
                            current_device = st.session_state.model.device
                            
                            # Extract the required shape from the error message
                            import re
                            shape_match = re.search(r'shape torch.Size\(\[(.*?)\]\)', str(e))
                            if shape_match:
                                shape_str = shape_match.group(1)
                                dims = [int(x.strip()) for x in shape_str.split(',')]
                                
                                # Adjust hidden dimension based on the error
                                if len(dims) == 2:
                                    new_hidden_dim = dims[1] if dims[1] > 2 else dims[0]
                                    st.info(f"Adjusting hidden dimension to {new_hidden_dim}")
                                    
                                    # Recreate model with new dimensions
                                    st.session_state.model = KuramotoSivashinskySupervisedLearning(device=current_device)
                                    st.session_state.model.model = st.session_state.model.model.__class__(
                                        hidden_layers=hidden_layers, 
                                        hidden_dim=new_hidden_dim
                                    ).to(current_device)
                                    
                                    # Try evaluation again
                                    st.session_state.model_solution, st.session_state.x_grid, st.session_state.t_grid = st.session_state.model.evaluate(
                                        nx=nx, nt=nt, t_max=t_max
                                    )
                                    
                                    st.session_state.reference_solution = st.session_state.model.generate_reference_solution(
                                        nx=nx, nt=nt, t_max=t_max
                                    )
                                    
                                    mae, norm_rmse, avg_temporal_corr, avg_spatial_corr, overall_score, passed = test_against_reference_solution(
                                        st.session_state.model_solution, 
                                        st.session_state.reference_solution,
                                        return_metrics=True
                                    )
                                    
                                    st.session_state.test_results = {
                                        'mae': mae,
                                        'norm_rmse': norm_rmse,
                                        'avg_temporal_corr': avg_temporal_corr,
                                        'avg_spatial_corr': avg_spatial_corr,
                                        'overall_score': overall_score,
                                        'passed': passed
                                    }
                                    
                                    st.session_state.evaluation_complete = True
                                    st.success("Visualizations loaded successfully after model adjustment!")
                            else:
                                st.error("Could not determine required shape from error message")
                        except Exception as e2:
                            st.error(f"Failed to fix model: {str(e2)}")
        
        plot_type = st.selectbox("Plot Type", ["Loss History", "Solution Surface", "Comparison", "Error Analysis"])
        
        if plot_type == "Loss History":
            if st.session_state.train_losses:
                # Plot training history if available
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.semilogy(st.session_state.train_losses, label='Training Loss')
                ax.semilogy(st.session_state.val_losses, label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss (MSE)')
                ax.set_title('Training and Validation Loss History')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            else:
                # If no training history is available (e.g., when loading a pre-trained model)
                st.info("No training history available for this model. This typically happens when loading a pre-trained model.")
                
                # Option to generate dummy loss history for demonstration
                if st.button("Generate Example Loss History"):
                    # Create example loss history for demonstration
                    example_epochs = 100
                    example_train_losses = [0.1 * np.exp(-0.05 * i) + 0.01 * np.random.rand() for i in range(example_epochs)]
                    example_val_losses = [0.15 * np.exp(-0.04 * i) + 0.02 * np.random.rand() for i in range(example_epochs)]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.semilogy(example_train_losses, label='Example Training Loss')
                    ax.semilogy(example_val_losses, label='Example Validation Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss (MSE)')
                    ax.set_title('Example Training and Validation Loss History')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    st.caption("Note: This is an example plot and does not represent the actual training history of the loaded model.")
        
        elif plot_type == "Solution Surface" and st.session_state.evaluation_complete:
            # Create 3D surface plot of the solution
            X, T = np.meshgrid(st.session_state.x_grid.flatten(), st.session_state.t_grid)
            
            fig = go.Figure(data=[go.Surface(
                z=st.session_state.model_solution, 
                x=X, 
                y=T,
                colorscale='viridis'
            )])
            
            fig.update_layout(
                title='Kuramoto-Sivashinsky Solution',
                scene=dict(
                    xaxis_title='Space (x)',
                    yaxis_title='Time (t)',
                    zaxis_title='u(x,t)'
                ),
                width=800,
                height=600
            )
            
            st.plotly_chart(fig)
        
        elif plot_type == "Comparison" and st.session_state.evaluation_complete:
            # Show time slices comparison
            time_slices = [0, len(st.session_state.t_grid)//4, len(st.session_state.t_grid)//2, 3*len(st.session_state.t_grid)//4, -1]
            
            fig, axes = plt.subplots(len(time_slices), 1, figsize=(10, 12))
            for i, tidx in enumerate(time_slices):
                axes[i].plot(st.session_state.x_grid, st.session_state.model_solution[tidx], 'b-', label='Neural Network')
                axes[i].plot(st.session_state.x_grid, st.session_state.reference_solution[tidx], 'r--', label='Reference')
                axes[i].set_title(f"t = {st.session_state.t_grid[tidx]:.2f}")
                axes[i].set_xlabel('x')
                axes[i].set_ylabel('u(x,t)')
                axes[i].grid(True)
                if i == 0:
                    axes[i].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif plot_type == "Error Analysis" and st.session_state.evaluation_complete:
            # Plot the absolute error
            diff = np.abs(st.session_state.model_solution - st.session_state.reference_solution)
            X, T = np.meshgrid(st.session_state.x_grid.flatten(), st.session_state.t_grid)
            
            fig = go.Figure(data=[go.Surface(
                z=diff, 
                x=X, 
                y=T,
                colorscale='hot'
            )])
            
            fig.update_layout(
                title='Absolute Error',
                scene=dict(
                    xaxis_title='Space (x)',
                    yaxis_title='Time (t)',
                    zaxis_title='|Error|'
                ),
                width=800,
                height=600
            )
            
            st.plotly_chart(fig)
    else:
        st.info("Train or load a model to visualize results")

# Display metrics and model info
# Display metrics and model info
with col2:
    st.header("Model Information")
    
    # Device info with more details
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        st.success(f"Using GPU: {device_name}")
        st.info(f"CUDA Version: {torch.version.cuda}")
    else:
        st.warning("CUDA not available, using CPU instead")
    
    if st.session_state.model is not None:
        st.subheader("Model Parameters")
        st.json({
            "Domain Size": domain_size,
            "Hidden Layers": hidden_layers,
            "Hidden Dimension": hidden_dim,
            "Learning Rate": learning_rate,
            "Batch Size": batch_size
        })
    
    if st.session_state.evaluation_complete and st.session_state.test_results is not None:
        st.subheader("Evaluation Metrics")
        
        # Display metrics with color coding
        mae = st.session_state.test_results['mae']
        st.metric("Mean Absolute Error", f"{mae:.6f}", 
                 delta_color="inverse")
        
        norm_rmse = st.session_state.test_results['norm_rmse']
        st.metric("Normalized RMSE", f"{norm_rmse:.6f}", 
                 delta_color="inverse")
        
        temp_corr = st.session_state.test_results['avg_temporal_corr']
        st.metric("Temporal Correlation", f"{temp_corr:.6f}")
        
        spat_corr = st.session_state.test_results['avg_spatial_corr']
        st.metric("Spatial Correlation", f"{spat_corr:.6f}")
        
        score = st.session_state.test_results['overall_score']
        passed = st.session_state.test_results['passed']
        
        st.metric("Overall Score", f"{score:.6f}")
        
        if passed:
            st.success("✅ Model passed evaluation criteria")
        else:
            st.error("❌ Model failed evaluation criteria")
    
    # Add a download button for the trained model if available
    if st.session_state.model is not None:
        st.markdown("---")
        st.subheader("Export Results")
        
        if st.session_state.evaluation_complete:
            if st.button("Generate Report"):
                # Create report figures
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.semilogy(st.session_state.train_losses, label='Training Loss')
                ax1.semilogy(st.session_state.val_losses, label='Validation Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss (MSE)')
                ax1.set_title('Training History')
                ax1.legend()
                ax1.grid(True)
                fig1.savefig("training_history.png")
                
                # Save solution visualization
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                c = ax2.contourf(
                    np.meshgrid(st.session_state.x_grid.flatten(), st.session_state.t_grid)[1],
                    np.meshgrid(st.session_state.x_grid.flatten(), st.session_state.t_grid)[0],
                    st.session_state.model_solution, 
                    100, 
                    cmap='viridis'
                )
                fig2.colorbar(c, ax=ax2, label='u(x,t)')
                ax2.set_title('Model Solution')
                ax2.set_xlabel('Time (t)')
                ax2.set_ylabel('Space (x)')
                fig2.savefig("model_solution.png")
                
                st.success("Report generated! Images saved as PNG files.")