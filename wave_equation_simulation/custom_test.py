import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0     # Particle mass
k = 1.0     # Constant in potential
alpha = 1.0 # Constant in potential

# Spatial grid parameters
Nx = 100          # Number of grid points in x
Ny = 100          # Number of grid points in y
Lx = 10.0         # Length of the x domain
Ly = 10.0         # Length of the y domain
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

# Initial wave function Psi(x, y, 0)
sigma = 1.0  # Width of the Gaussian packet
k0 = 5.0     # Initial wave number
Psi = np.exp(- (X**2 + Y**2) / (2 * sigma**2)) * np.exp(1j * k0 * X)

# Potential function V(x, y, t)
def V_func(X, Y, t):
    return np.sin(k * (X**2 + Y**2) * t) - np.exp(- alpha / (t + 1))

# Function to compute Laplacian using finite differences
def compute_laplacian(Psi, dx, dy):
    Psi_xx = (np.roll(Psi, -1, axis=1) - 2 * Psi + np.roll(Psi, 1, axis=1)) / dx**2
    Psi_yy = (np.roll(Psi, -1, axis=0) - 2 * Psi + np.roll(Psi, 1, axis=0)) / dy**2
    return Psi_xx + Psi_yy

# Time evolution parameters
dt = 0.0001  # Time step
Nt = 1000    # Number of time steps
t = 0.0      # Initial time

# Prepare for animation
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(np.abs(Psi)**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
               origin='lower', cmap='viridis', vmin=0, vmax=np.max(np.abs(Psi)**2))
ax.set_xlabel('Position x')
ax.set_ylabel('Position y')
title = ax.set_title(f'Probability Density at t = {t:.4f}')
cbar = fig.colorbar(im, ax=ax, label='Probability Density |Ψ|²')

def init():
    im.set_data(np.abs(Psi)**2)
    title.set_text(f'Probability Density at t = {t:.4f}')
    return [im, title]

def animate(n):
    global Psi, t
    # Compute Laplacian of Psi
    Laplacian_Psi = compute_laplacian(Psi, dx, dy)
    
    # Compute potential V at current time
    V_t = V_func(X, Y, t)
    
    # Compute time derivative of Psi
    dPsi_dt = (1j / hbar) * (- (hbar**2) / (2 * m) * Laplacian_Psi + V_t * Psi)
    
    # Update Psi
    Psi += dt * dPsi_dt
    
    # Apply boundary conditions (Psi = 0 at boundaries)
    Psi[0, :] = Psi[-1, :] = Psi[:, 0] = Psi[:, -1] = 0
    
    # Update time
    t += dt
    
    # Update image data and title
    im.set_array(np.abs(Psi)**2)
    title.set_text(f'Probability Density at t = {t:.4f}')
    return [im, title]

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=Nt, interval=20, blit=True)

# Display the animation
plt.show()
