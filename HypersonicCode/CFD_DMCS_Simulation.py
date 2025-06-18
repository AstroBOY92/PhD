# Load mesh file (e.g., .stl, .obj)
mesh = pv.read("your_mesh_file.stl")  # Replace with your file path

# Plot original 3D mesh
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, color='black', show_edges=True)
plotter.set_background("skyblue")
plotter.screenshot("mesh_3d.png")
plotter.close()

# Get mesh points
points = mesh.points

# Project onto a plane (e.g., the XY plane, or rotate first)
# Optionally rotate the mesh for a skewed projection
angle = np.deg2rad(30)
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle),  np.cos(angle), 0],
                            [0,              0,             1]])

rotated_points = points @ rotation_matrix.T

# Project to 2D (XY projection)
projected = rotated_points[:, :2]

# Create 2D grid for plotting
x = projected[:, 0]
y = projected[:, 1]

# Plot 2D projection
plt.figure(figsize=(8, 8))
plt.tricontourf(x, y, np.zeros_like(x), levels=1, colors='lightcoral', alpha=0.6)
plt.triplot(x, y, mesh.faces.reshape(-1, 4)[:, 1:], color='k', linewidth=0.2)

plt.grid(True)
plt.xlabel("mm")
plt.ylabel("mm")
plt.axis('equal')
plt.savefig("projection_2d.png_")

import numpy as np
import matplotlib.pyplot as plt

# Input parameters
U = 1815  # Free Stream Velocity (m/s)
T = 220   # Free Stream Temperature (K)
p = 0.88  # Free Stream Pressure (Pa)
rho = 1.36e-5  # Free Stream Density (Kg/m^3)
lambda_path = 4.45e-3  # Free Stream Mean Path (m)
Tw = 1000  # Wall Temperature (K)

# Constants
kB = 1.38e-23  # Boltzmann constant (J/K)
R = 287  # Specific gas constant for air (J/(kg·K))

# Simulation parameters
num_simulations = 1000

# Initialize arrays to store results
temperature_distribution = np.zeros(num_simulations)
pressure_distribution = np.zeros(num_simulations)
heat_flux = np.zeros(num_simulations)

# Function to calculate heat flux
def calculate_heat_flux(T, Tw, rho, U, kB, R):
    # Heat flux equation (simplified for this example)
    q = rho * U * kB * (Tw - T) / R
    return q

# Monte Carlo simulation
for i in range(num_simulations):
    # Random perturbations for temperature and pressure
    temp_variation = np.random.normal(0, 5)
    pressure_variation = np.random.normal(0, 0.1)

    # Calculate temperature and pressure distributions
    T_sim = T + temp_variation
    p_sim = p + pressure_variation

    temperature_distribution[i] = T_sim
    pressure_distribution[i] = p_sim

    # Calculate heat flux
    heat_flux[i] = calculate_heat_flux(T_sim, Tw, rho, U, kB, R)

# Plot results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(temperature_distribution, bins=30, alpha=0.7, color='blue')
plt.title('Temperature Distribution')
plt.xlabel('Temperature (K)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(pressure_distribution, bins=30, alpha=0.7, color='green')
plt.title('Pressure Distribution')
plt.xlabel('Pressure (Pa)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Calculate and print average heat flux
average_heat_flux = np.mean(heat_flux)
print(f'Average Heat Flux: {average_heat_flux:.2f} W/m^2')

# Save the results
import pandas as pd

results = pd.DataFrame({
    'Temperature (K)': temperature_distribution,
    'Pressure (Pa)': pressure_distribution,
    'Heat Flux (W/m^2)': heat_flux
})

results.to_csv('monte_carlo_cfd_results.csv', index=False)

print("Simulation results saved to 'monte_carlo_cfd_results.csv'")


