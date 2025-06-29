import numpy as np
import pandas as pd

# Constants (edit as appropriate for your material)
rho_0 = 2.33       # Initial density (g/cm^3) — example for Si
rho_m = 2.65       # Oxide density (g/cm^3) — example for SiO2
A = 1.0            # Area (cm^2)
k = 8.617e-5       # Boltzmann constant (eV/K)
D = 1.0            # Set D=1 (unitless) unless specified
x = 0              # Assume x=0 so cos(Dx)=1

# Time array (hours)
time_hr = np.linspace(0, 20, 100)
time_s = time_hr * 3600  # convert to seconds if needed

# Calculate W(t)
W = (rho_0 * rho_m * np.exp((rho_0/rho_m)*time_hr) * np.cos(D*x) * A * time_hr) / k

# Create DataFrame
df_wt = pd.DataFrame({
    'Time_hr': time_hr,
    'Weight_Change': W
})

print(df_wt.head())
df_wt.to_csv('weight_change_vs_time.csv', index=False)
