##Machine Learning in Material Science example

Non linear Gerkin
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
Stress Distribution Code
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


# In[6]:


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

Predicting Material Hardness

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the environment
# For simplicity, let's consider a 2D grid where each cell represents a state of the material
# Actions: 0=up, 1=right, 2=down, 3=left
grid_size = 5
num_actions = 4
rewards = np.zeros((grid_size, grid_size))
rewards[4, 4] = 1  # Target state with a positive reward

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, num_actions))

# Helper functions
def get_next_state(state, action):
    x, y = state
    if action == 0 and x > 0:
        x -= 1
    elif action == 1 and y < grid_size - 1:
        y += 1
    elif action == 2 and x < grid_size - 1:
        x += 1
    elif action == 3 and y > 0:
        y -= 1
    return x, y

def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # Explore
    else:
        return np.argmax(Q[state[0], state[1], :])  # Exploit

# Training the agent
for episode in range(num_episodes):
    state = (0, 0)  # Start state
    while state != (4, 4):  # Until reaching the target state
        action = choose_action(state)
        next_state = get_next_state(state, action)
        reward = rewards[next_state[0], next_state[1]]
        Q[state[0], state[1], action] = (1 - alpha) * Q[state[0], state[1], action] +             alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], :]))
        state = next_state

# Visualizing the learned Q-values
plt.figure(figsize=(10, 10))
sns.heatmap(np.max(Q, axis=2), annot=True, cmap='viridis')
plt.title('Learned Q-values for each state')
plt.xlabel('State (y)')
plt.ylabel('State (x)')
plt.show()

Predictive State Material
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the environment
# For simplicity, let's consider a 2D grid where each cell represents a state of the material
# Actions: 0=up, 1=right, 2=down, 3=left
grid_size = 5
num_actions = 4
rewards = np.zeros((grid_size, grid_size))
rewards[4, 4] = 1  # Target state with a positive reward

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, num_actions))

# Helper functions
def get_next_state(state, action):
    x, y = state
    if action == 0 and x > 0:
        x -= 1
    elif action == 1 and y < grid_size - 1:
        y += 1
    elif action == 2 and x < grid_size - 1:
        x += 1
    elif action == 3 and y > 0:
        y -= 1
    return x, y

def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # Explore
    else:
        return np.argmax(Q[state[0], state[1], :])  # Exploit

# Training the agent
for episode in range(num_episodes):
    state = (0, 0)  # Start state
    while state != (4, 4):  # Until reaching the target state
        action = choose_action(state)
        next_state = get_next_state(state, action)
        reward = rewards[next_state[0], next_state[1]]
        Q[state[0], state[1], action] = (1 - alpha) * Q[state[0], state[1], action] +             alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], :]))
        state = next_state

# Visualizing the learned Q-values
plt.figure(figsize=(10, 10))
sns.heatmap(np.max(Q, axis=2), annot=True, cmap='viridis')
plt.title('Learned Q-values for each state')
plt.xlabel('State (y)')
plt.ylabel('State (x)')
plt.show()
