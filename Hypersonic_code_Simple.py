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
