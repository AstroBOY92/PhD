#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
Du, Dv = 0.16, 0.08  # Diffusion coefficients
rho, alpha, beta = 0.06, 0.075, 0.02  # Reaction terms

# Domain
L = 100  # Length of the square domain
dx = dy = 2.0  # Space step
T = 10.0  # Total time
dt = 0.01  # Time step
n = int(L/dx)  # Number of points in each direction
m = int(T/dt)  # Number of time points

# Initial conditions
U = np.random.rand(n, n)
V = np.random.rand(n, n)

def laplacian(Z):
    Ztop = np.roll(Z, -1, axis=0)
    Zleft = np.roll(Z, -1, axis=1)
    Zbottom = np.roll(Z, 1, axis=0)
    Zright = np.roll(Z, 1, axis=1)
    return (Ztop + Zleft + Zbottom + Zright - 4 * Z) / dx**2

def update(U, V, Du, Dv, rho, alpha, beta, dt):
    # Apply the reaction-diffusion model
    dUdt = Du * laplacian(U) - U * V**2 + rho * (1 - U)
    dVdt = Dv * laplacian(V) + U * V**2 - (rho + alpha) * V
    U += dUdt * dt
    V += dVdt * dt
    return U, V

# Time-stepping loop
pattern_history = []  # To store patterns at different time steps for plotting
time_points = []  # To store time points for plotting
for t in range(m):
    if t % (m // 10) == 0:  # Store every 10th step
        pattern_history.append(U.copy())
        time_points.append(t*dt)
    U, V = update(U, V, Du, Dv, rho, alpha, beta, dt)

# Plot the results at different time steps
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, pattern, time_point in zip(axes.flatten(), pattern_history, time_points):
    ax.imshow(pattern, cmap='Spectral', interpolation='bilinear')
    ax.set_title(f'Pattern at t = {time_point:.2f}')
    ax.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




