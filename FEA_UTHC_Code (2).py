#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fenics
from fenics import *
import matplotlib.pyplot as plt

# Create mesh and define function space
nx = ny = 50
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('300 + 100*x[0]', degree=1)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define source term
f = Constant(0)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution to file in VTK format
vtkfile = File('heat_solution.pvd')
vtkfile << u

# Plot solution
p = plot(u)
plt.colorbar(p)
plt.title('Temperature distribution in UHTC sample')
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Define the problem domain
nx, ny = 50, 50
dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)

# Define thermal conductivity and source term
k = 1.0
Q = 0.0

# Initialize temperature array
T = np.zeros((nx, ny))

# Boundary conditions
T[:, 0] = 300  # Left side
T[:, -1] = 300  # Right side
T[0, :] = 300  # Bottom side
T[-1, :] = 400  # Top side

# Setup coefficient matrix and right-hand side vector
A = np.zeros((nx * ny, nx * ny))
b = np.zeros(nx * ny)

for i in range(1, nx - 1):
    for j in range(1, ny - 1):
        p = i * ny + j
        A[p, p] = -4 * k
        A[p, p - 1] = k
        A[p, p + 1] = k
        A[p, p - ny] = k
        A[p, p + ny] = k
        b[p] = -Q * dx * dy

# Solve the system of equations
T_flat = solve(A, b)

# Reshape the solution to a 2D array
T[1:-1, 1:-1] = T_flat.reshape((nx - 2, ny - 2))

# Plot the solution
plt.imshow(T, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
plt.colorbar(label='Temperature (K)')
plt.title('Temperature distribution in UHTC sample')
plt.show()


# In[ ]:




