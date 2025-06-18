import numpy as np
import pandas as pd

# Define parameters for the hypersonic flow dataset
num_points = 500

# Altitude profile (meters)
altitude = np.linspace(20000, 80000, num_points)

# Generate atmospheric temperature (Kelvin) decreasing with altitude
temperature = 288.15 - 0.0065 * altitude

# Pressure (Pascals) decreasing exponentially with altitude
pressure = 101325 * np.exp(-altitude / 7500)

# Density (kg/m^3) using ideal gas law: rho = P / (R_specific * T)
R_specific = 287.05  # J/(kg·K) for air
density = pressure / (R_specific * temperature)

# Mach number, typically hypersonic regime (Mach 5 to Mach 25)
mach_number = np.random.uniform(5, 25, num_points)

# Velocity (m/s), approximated using speed of sound and Mach number
speed_of_sound = np.sqrt(1.4 * R_specific * temperature)
velocity = mach_number * speed_of_sound

# Enthalpy (J/kg), approximated using specific heat capacity
Cp = 1005  # J/(kg·K), approximate specific heat for air
enthalpy = Cp * temperature

# Compile data into DataFrame
data = pd.DataFrame({
    'Altitude(m)': altitude,
    'Temperature(K)': temperature,
    'Pressure(Pa)': pressure,
    'Density(kg/m^3)': density,
    'Mach_Number': mach_number,
    'Velocity(m/s)': velocity,
    'Enthalpy(J/kg)': enthalpy
})

# Save dataset to CSV
data.to_csv('hypersonic_thermo_fluid_data.csv', index=False)

print("Thermo-fluid data for hypersonic flow generated and saved to CSV.")
