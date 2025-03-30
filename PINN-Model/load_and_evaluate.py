import torch
import numpy as np
import matplotlib.pyplot as plt

# Import the model class
from paste import KuramotoSivashinskyPhysicsEnhanced

# Create a new model instance 
ks_model = KuramotoSivashinskyPhysicsEnhanced(device='cpu')

# Modify the load_model method to specifically handle loading from CUDA to CPU
def load_model_cpu(self, path):
    """Load a trained model with CPU mapping."""
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.domain_size = checkpoint['domain_size']
    self.device = 'cpu'
    self.model = self.model.to('cpu')

# Monkey patch the load_model method
ks_model.load_model = lambda path: load_model_cpu(ks_model, path)

# Load the saved model weights with the patched method
ks_model.load_model(path="./ks_physics_enhanced_model.pt")


# Now let's run inference
nx, nt = 128, 50  # Same parameters as in the training
t_max = 5.0

# Get the model solution
print("Generating model solution...")
model_solution, x_grid, t_grid = ks_model.evaluate(nx=nx, nt=nt, t_max=t_max)

# Generate a reference solution for comparison
print("Generating reference solution for comparison...")
reference_solution = ks_model.generate_reference_solution(nx=nx, nt=nt, t_max=t_max)

# Compare the solutions
print("Comparing solutions...")
from paste import visualize_solution, compare_solutions, test_against_reference_solution

# Compare the model solution against the reference solution
compare_solutions(model_solution, reference_solution, x_grid, t_grid)

# Run quantitative tests
print("\nTesting model against reference solution:")
test_result = test_against_reference_solution(model_solution, reference_solution)
print(f"Test {'passed' if test_result else 'failed'}")
