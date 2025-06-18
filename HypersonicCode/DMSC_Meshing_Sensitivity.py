#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

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
plt.savefig("projection_2d.png_


# In[ ]:




