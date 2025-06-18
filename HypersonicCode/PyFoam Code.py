#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
gamma = 1.4  # Specific heat ratio for air
nx, ny = 200, 100  # Grid resolution
Lx, Ly = 2.0, 1.0  # Domain size in meters
dx, dy = Lx/nx, Ly/ny
cfl = 0.5

# Initial conditions: hypersonic inflow
M_inf = 7.0  # Hypersonic Mach number
rho_inf = 1.225
p_inf = 101325 / 10  # Lower pressure (high altitude)
T_inf = p_inf / (rho_inf * 287.0)
a_inf = np.sqrt(gamma * 287.0 * T_inf)
u_inf = M_inf * a_inf
v_inf = 0.0
e_inf = p_inf / (gamma - 1) + 0.5 * rho_inf * (u_inf**2 + v_inf**2)


# In[4]:


# State variables
rho = np.ones((ny, nx)) * rho_inf
u = np.ones((ny, nx)) * u_inf
v = np.ones((ny, nx)) * v_inf
E = np.ones((ny, nx)) * e_inf

# Boundary condition: a wedge at the bottom center
def apply_boundary_conditions():
    # Reflective bottom wedge
    for i in range(nx):
        for j in range(0, ny//2 - int(i*0.2) if i < nx//2 else ny//2 - int((nx - i - 1)*0.2)):
            u[j, i] = -u[j, i]
            v[j, i] = -v[j, i]

# Time integration (Euler explicit)
def compute_time_step():
    a = np.sqrt(gamma * (E/rho - 0.5 * (u**2 + v**2)) * (gamma - 1))
    dt = cfl * min(dx / (np.abs(u) + a).max(), dy / (np.abs(v) + a).max())
    return dt

# Fluxes
def compute_fluxes():
    p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
    fx = np.zeros((ny, nx, 4))
    fy = np.zeros((ny, nx, 4))
    fx[:, :, 0] = rho * u
    fx[:, :, 1] = rho * u**2 + p
    fx[:, :, 2] = rho * u * v
    fx[:, :, 3] = (E + p) * u
    fy[:, :, 0] = rho * v
    fy[:, :, 1] = rho * u * v
    fy[:, :, 2] = rho * v**2 + p
    fy[:, :, 3] = (E + p) * v
    return fx, fy

# Update function
def update():
    global rho, u, v, E
    fx, fy = compute_fluxes()
    dt = compute_time_step()

    # Finite volume update
    rho[:, 1:-1] -= dt/dx * (fx[:, 1:-1, 0] - fx[:, :-2, 0])
    u[:, 1:-1] -= dt/dx * (fx[:, 1:-1, 1] - fx[:, :-2, 1]) / rho[:, 1:-1]
    v[1:-1, :] -= dt/dy * (fy[1:-1, :, 2] - fy[:-2, :, 2]) / rho[1:-1, :]
    E[:, 1:-1] -= dt/dx * (fx[:, 1:-1, 3] - fx[:, :-2, 3])
    E[1:-1, :] -= dt/dy * (fy[1:-1, :, 3] - fy[:-2, :, 3])

    apply_boundary_conditions()

# Visualization
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((ny, nx)), cmap='inferno', origin='lower', extent=[0, Lx, 0, Ly])
plt.title("Hypersonic 2D Simulation - Density Field")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

def animate(i):
    for _ in range(5):  # Run multiple steps between frames
        update()
    im.set_data(rho)
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.show()


# In[ ]:




