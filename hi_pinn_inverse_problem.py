#!/usr/bin/env python3
"""
HI-PINN: Hybrid Iterative Physics-Informed Neural Networks for Inverse Source Problems

Submitted to: Mathematical Methods in the Applied Sciences
Special Issue: Inverse Problems, Numerical Methods and Artificial Intelligence

Author: [Your Name]
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# PROBLEM SETUP
# ============================================

nx = 100
L = 1.0
dx = L / nx
x = np.linspace(0, L, nx)

# True source (single Gaussian peak)
def true_source(x):
    return 5.0 * np.exp(-((x - 0.5)/0.15)**2)

source_true = true_source(x)

# Build finite difference matrix for -u'' = f
A = np.zeros((nx, nx))
for i in range(nx):
    A[i, i] = 2.0 / dx**2
    if i > 0:
        A[i, i-1] = -1.0 / dx**2
    if i < nx-1:
        A[i, i+1] = -1.0 / dx**2

# Dirichlet BCs: u(0)=0, u(1)=0
A[0, :] = 0
A[0, 0] = 1.0
A[-1, :] = 0
A[-1, -1] = 1.0

def solve_forward(source):
    """Solve -u'' = f with Dirichlet BCs"""
    b = source.copy()
    b[0] = 0
    b[-1] = 0
    return np.linalg.solve(A, b)

# Generate measurements
u_true = solve_forward(source_true)
n_sensors = 15
sensor_indices = np.linspace(10, nx-11, n_sensors, dtype=int)
sensor_positions = x[sensor_indices]
measurements_clean = u_true[sensor_indices]
noise_level = 0.03
noise = noise_level * np.std(measurements_clean) * np.random.randn(n_sensors)
measurements_noisy = measurements_clean + noise

print("="*60)
print("PROBLEM SETUP")
print("="*60)
print(f"Grid points: {nx}")
print(f"Sensors: {n_sensors}")
print(f"Noise level: {noise_level*100}%")
print(f"True source max: {source_true.max():.2f}")

# ============================================
# PINN MODEL
# ============================================

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
sensor_indices_tensor = torch.tensor(sensor_indices, dtype=torch.long).to(device)
measurements_tensor = torch.tensor(measurements_noisy, dtype=torch.float32).to(device)

class SimplePINN(nn.Module):
    """Physics-Informed Neural Network for source function representation"""
    def __init__(self):
        super(SimplePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def solve_forward_torch(source):
    """Differentiable forward solver in PyTorch"""
    b = source.clone()
    b[0] = 0.0
    b[-1] = 0.0
    return torch.linalg.solve(A_tensor, b)

# ============================================
# TRAINING HI-PINN
# ============================================

print("\n" + "="*60)
print("TRAINING HI-PINN")
print("="*60 + "\n")

model = SimplePINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300)

loss_history = []

for epoch in range(2000):
    optimizer.zero_grad()
    
    # Predict source
    source_pred = model(x_tensor).flatten()
    
    # Solve forward problem
    u_pred = solve_forward_torch(source_pred)
    u_at_sensors = u_pred[sensor_indices_tensor]
    
    # Data loss
    data_loss = torch.mean((u_at_sensors - measurements_tensor)**2)
    
    # Smoothness regularization
    reg_loss = 0.001 * torch.mean((source_pred[1:] - source_pred[:-1])**2)
    
    total_loss = data_loss + reg_loss
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss)
    
    loss_history.append(total_loss.item())
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: Loss={total_loss.item():.6f}, Data={data_loss.item():.6f}")

# Final reconstruction
model.eval()
with torch.no_grad():
    source_reconstructed = model(x_tensor).cpu().numpy().flatten()

l2_error_pinn = np.linalg.norm(source_reconstructed - source_true) / np.linalg.norm(source_true)
print(f"\n✅ HI-PINN Error: {l2_error_pinn*100:.2f}%")

# ============================================
# CLASSICAL LANDWEBER BASELINE
# ============================================

def landweber_simple(iterations=800, tau=0.0005):
    """Classical Landweber iteration without machine learning"""
    source_est = np.zeros(nx)
    errors = []
    
    for k in range(iterations):
        u_pred = solve_forward(source_est)
        residual = measurements_noisy - u_pred[sensor_indices]
        
        # Simple gradient (adjoint operator)
        grad = np.zeros(nx)
        for i, idx in enumerate(sensor_indices):
            grad[idx] += residual[i]
        
        source_est = source_est + tau * grad
        
        if k % 200 == 0:
            error = np.linalg.norm(source_est - source_true) / np.linalg.norm(source_true)
            errors.append(error)
            print(f"Landweber iter {k}: error={error*100:.2f}%")
    
    final_error = np.linalg.norm(source_est - source_true) / np.linalg.norm(source_true)
    return source_est, final_error, errors

print("\n" + "="*60)
print("CLASSICAL LANDWEBER BASELINE")
print("="*60 + "\n")

source_landweber, l2_error_landweber, landweber_errors = landweber_simple(iterations=800, tau=0.0005)
print(f"\nLandweber Error: {l2_error_landweber*100:.2f}%")
print(f"Improvement: {(l2_error_landweber - l2_error_pinn)*100:.2f} percentage points")

# ============================================
# UNCERTAINTY QUANTIFICATION (ENSEMBLE)
# ============================================

def uncertainty_ensemble(n_models=8, iterations_per_model=400):
    """Ensemble-based uncertainty quantification"""
    all_preds = []
    
    print("\n" + "="*60)
    print("UNCERTAINTY QUANTIFICATION (ENSEMBLE)")
    print("="*60 + "\n")
    
    for m in range(n_models):
        torch.manual_seed(m * 111)
        model_m = SimplePINN().to(device)
        opt = optim.Adam(model_m.parameters(), lr=0.001)
        
        for epoch in range(iterations_per_model):
            opt.zero_grad()
            source_pred = model_m(x_tensor).flatten()
            u_pred = solve_forward_torch(source_pred)
            u_at_sensors = u_pred[sensor_indices_tensor]
            loss = torch.mean((u_at_sensors - measurements_tensor)**2)
            loss.backward()
            opt.step()
        
        with torch.no_grad():
            pred = model_m(x_tensor).cpu().numpy().flatten()
            all_preds.append(pred)
        
        print(f"  Model {m+1}/{n_models} done")
    
    all_preds = np.array(all_preds)
    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)

mean_source, std_source = uncertainty_ensemble(n_models=8, iterations_per_model=400)

# Calculate correlation between uncertainty and true error
true_error = np.abs(mean_source - source_true)
correlation = np.corrcoef(std_source, true_error)[0,1] if len(std_source) > 1 else 0
print(f"\n✓ Mean uncertainty: {np.mean(std_source):.6f}")
print(f"✓ Max uncertainty: {np.max(std_source):.6f}")
print(f"✓ Pearson correlation: {correlation:.3f}")

# ============================================
# GENERATE FIGURES
# ============================================

print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60 + "\n")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: True source
axes[0,0].plot(x, source_true, 'b-', linewidth=2.5, label='True $f(x)$')
axes[0,0].fill_between(x, 0, source_true, alpha=0.2, color='blue')
axes[0,0].set_xlabel('$x$', fontsize=12)
axes[0,0].set_ylabel('$f(x)$', fontsize=12)
axes[0,0].set_title('Ground Truth Source', fontsize=12)
axes[0,0].legend(fontsize=10)
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Reconstruction comparison
axes[0,1].plot(x, source_true, 'b-', linewidth=2.5, label='True')
axes[0,1].plot(x, source_reconstructed, 'r--', linewidth=2, label=f'HI-PINN ({l2_error_pinn*100:.1f}%)')
axes[0,1].plot(x, source_landweber, 'g:', linewidth=2, label=f'Landweber ({l2_error_landweber*100:.1f}%)')
axes[0,1].scatter(sensor_positions, measurements_noisy, c='k', s=40, marker='x', label='Noisy data')
axes[0,1].set_xlabel('$x$', fontsize=12)
axes[0,1].set_ylabel('$f(x)$', fontsize=12)
axes[0,1].set_title('Reconstruction Comparison', fontsize=12)
axes[0,1].legend(fontsize=9)
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Training loss convergence
axes[0,2].semilogy(loss_history, 'b-', linewidth=1.5)
axes[0,2].set_xlabel('Epoch', fontsize=12)
axes[0,2].set_ylabel('Loss', fontsize=12)
axes[0,2].set_title('Training Loss Convergence', fontsize=12)
axes[0,2].grid(True, alpha=0.3)

# Plot 4: Uncertainty quantification
axes[1,0].plot(x, mean_source, 'r-', linewidth=2, label='Mean prediction')
axes[1,0].fill_between(x, mean_source - 2*std_source, mean_source + 2*std_source, 
                        alpha=0.3, color='red', label='95% confidence')
axes[1,0].plot(x, source_true, 'b--', linewidth=2, label='True')
axes[1,0].set_xlabel('$x$', fontsize=12)
axes[1,0].set_ylabel('$f(x)$', fontsize=12)
axes[1,0].set_title('Uncertainty Quantification', fontsize=12)
axes[1,0].legend(fontsize=10)
axes[1,0].grid(True, alpha=0.3)

# Plot 5: Error comparison bar chart
methods = ['Landweber', 'HI-PINN']
errors = [l2_error_landweber*100, l2_error_pinn*100]
bars = axes[1,1].bar(methods, errors, color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)
axes[1,1].set_ylabel('Relative L2 Error (%)', fontsize=12)
axes[1,1].set_title('Error Comparison', fontsize=12)
axes[1,1].set_ylim(0, max(errors) * 1.2)
for bar, err in zip(bars, errors):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{err:.1f}%', ha='center', fontsize=12, fontweight='bold')

# Plot 6: Improvement pie chart
improvement = (l2_error_landweber - l2_error_pinn) / l2_error_landweber * 100
axes[1,2].pie([improvement, 100-improvement], 
              labels=[f'Improvement\n{improvement:.1f}%', f'Remaining\n{100-improvement:.1f}%'],
              colors=['#27ae60', '#e74c3c'], autopct='%1.1f%%', startangle=90)
axes[1,2].set_title(f'Error Reduction: {improvement:.1f}%', fontsize=12)

plt.tight_layout()
plt.savefig('manuscript_figures.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure saved as 'manuscript_figures.png'")

# ============================================
# SAVE RESULTS
# ============================================

results = {
    'l2_error_pinn_percent': float(l2_error_pinn * 100),
    'l2_error_landweber_percent': float(l2_error_landweber * 100),
    'improvement_absolute': float((l2_error_landweber - l2_error_pinn) * 100),
    'improvement_relative': float(improvement),
    'final_loss': float(loss_history[-1]),
    'mean_uncertainty': float(np.mean(std_source)),
    'max_uncertainty': float(np.max(std_source)),
    'correlation': float(correlation),
    'n_sensors': n_sensors,
    'noise_level_percent': noise_level * 100,
    'nx': nx,
    'iterations': 2000
}

with open('final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Results saved to 'final_results.json'")

# ============================================
# FINAL OUTPUT
# ============================================

print("\n" + "="*70)
print("FINAL RESULTS FOR MANUSCRIPT")
print("="*70)
print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    TABLE 1: QUANTITATIVE RESULTS                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Method                  │ Relative L2 Error (%)                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Classical Landweber     │ {l2_error_landweber*100:>6.2f}%                          ║
║  HI-PINN (Ours)          │ {l2_error_pinn*100:>6.2f}%                          ║
╚══════════════════════════════════════════════════════════════════════╝

📊 Improvement: {(l2_error_landweber - l2_error_pinn)*100:.2f} percentage points ({improvement:.1f}% relative reduction)

📈 CONVERGENCE METRICS:
   ├── PINN final loss: {loss_history[-1]:.8f}
   ├── Landweber iterations: 800
   └── PINN iterations: 2000

🎲 UNCERTAINTY METRICS:
   ├── Mean uncertainty: {np.mean(std_source):.6f}
   ├── Max uncertainty: {np.max(std_source):.6f}
   └── Pearson correlation: {correlation:.3f}

🔬 EXPERIMENTAL SETUP:
   ├── Grid points: {nx}
   ├── Sensors: {n_sensors}
   ├── Noise level: {noise_level*100}%
   └── Domain: [0, {L}]
""")

print("✅ ALL EXPERIMENTS COMPLETE!")