import numpy as np
import pandas as pd

# Constants for Dry Oxidation (from your data)
C1 = 7.72e2    # μm^2/hr
C2 = 6.23e6    # μm^2/hr
k = 8.617e-5   # eV/K
T = 1100 + 273 # Temperature in Kelvin (example: 1100°C furnace)
# Activation energy for dry oxidation (parabolic)
E = 2.00       # eV
a2 = 1         # Assume a2=1 unless specified

# Time array (hours)
time_hr = np.linspace(0, 20, 100)

# Calculate β and α
beta = C1 * np.exp(-E/(k*T))
alpha = beta / (C2 * np.exp(-E/(k*T)))

# Calculate oxide thickness for each t (choose + root for physical solution)
thickness = []
for t in time_hr:
    discriminant = alpha**2 - 4*beta*t
    if discriminant < 0:
        x = np.nan  # Non-physical, skip or set NaN
    else:
        x = (alpha + np.sqrt(discriminant)) / (2*a2)
    thickness.append(x)

# Create DataFrame
df_thickness = pd.DataFrame({
    'Time_hr': time_hr,
    'Oxide_Thickness_um': thickness
})

print(df_thickness.head())
df_thickness.to_csv('oxide_thickness_vs_time.csv', index=False)
