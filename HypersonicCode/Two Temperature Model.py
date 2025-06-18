#Two Temperature Model Code
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
electron_temperature_distribution = np.zeros(num_simulations)
ion_temperature_distribution = np.zeros(num_simulations)
pressure_distribution = np.zeros(num_simulations)
heat_flux = np.zeros(num_simulations)

# Function to calculate heat flux
def calculate_heat_flux(Te, Ti, Tw, rho, U, kB, R):
    # Heat flux equation (simplified for this example)
    q = rho * U * kB * (Tw - (Te + Ti) / 2) / R
    return q

# Monte Carlo simulation for TTM
for i in range(num_simulations):
    # Random perturbations for electron and ion temperatures and pressure
    Te_variation = np.random.normal(0, 5)
    Ti_variation = np.random.normal(0, 5)
    pressure_variation = np.random.normal(0, 0.1)

    # Calculate electron and ion temperature distributions
    Te_sim = T + Te_variation
    Ti_sim = T + Ti_variation
    p_sim = p + pressure_variation

    electron_temperature_distribution[i] = Te_sim
    ion_temperature_distribution[i] = Ti_sim
    pressure_distribution[i] = p_sim

    # Calculate heat flux
    heat_flux[i] = calculate_heat_flux(Te_sim, Ti_sim, Tw, rho, U, kB, R)

# Plot results
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.hist(electron_temperature_distribution, bins=30, alpha=0.7, color='blue')
plt.title('Electron Temperature Distribution')
plt.xlabel('Temperature (K)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(ion_temperature_distribution, bins=30, alpha=0.7, color='red')
plt.title('Ion Temperature Distribution')
plt.xlabel('Temperature (K)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
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
    'Electron Temperature (K)': electron_temperature_distribution,
    'Ion Temperature (K)': ion_temperature_distribution,
    'Pressure (Pa)': pressure_distribution,
    'Heat Flux (W/m^2)': heat_flux
})

results.to_csv('two_temperature_model_results.csv', index=False)

print("Simulation results saved to 'two_temperature_model_results.csv'")

Hypersonic Temperature Distribution
import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 1.4
R = 287.05

# Function to calculate properties for a given Mach number
def hypersonic_properties(mach):
    # Isentropic relations
    temperature_ratio = 1 / (1 + (gamma - 1) * 0.5 * mach**2)
    pressure_ratio = temperature_ratio**(gamma / (gamma - 1))
    velocity = mach * np.sqrt(gamma * R * 1.0)  # Assuming T0=1.0 for simplicity

    # Calculate actual properties
    temperature = temperature_ratio * 1.0
    pressure = pressure_ratio * 1.0
    density = pressure / (R * temperature)

    return temperature, pressure, density, velocity

# Calculate properties for a range of Mach numbers from 0 to 20
mach_numbers = np.linspace(0, 20, 100)
properties = [hypersonic_properties(mach) for mach in mach_numbers]

# Extract properties
temperatures = [prop[0] for prop in properties]
pressures = [prop[1] for prop in properties]
velocities = [prop[3] for prop in properties]

# Create a plot for temperature
plt.figure(figsize=(10, 6))
plt.plot(mach_numbers, temperatures, label='Temperature (T0=1.0)')
plt.xlabel('Mach Number')
plt.ylabel('Temperature (T0=1.0)')
plt.title('Hypersonic Temperature Distribution')
plt.grid(True)
plt.legend()
plt.show()

# Create a plot for pressure
plt.figure(figsize=(10, 6))
plt.plot(mach_numbers, pressures, label='Pressure (P0=1.0)')
plt.xlabel('Mach Number')
plt.ylabel('Pressure (P0=1.0)')
plt.title('Hypersonic Pressure Distribution')
plt.grid(True)
plt.legend()
plt.show()

# Create a plot for velocity
plt.figure(figsize=(10, 6))
plt.plot(mach_numbers, velocities, label='Velocity (U0)')
plt.xlabel('Mach Number')
plt.ylabel('Velocity (U0)')
plt.title('Hypersonic Velocity Distribution')
plt.grid(True)
plt.legend()
plt.show()
