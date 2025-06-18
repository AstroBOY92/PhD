#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Increase resolution
nx, ny = 300, 300
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# ----- Von Mises Stress Distribution -----
von_mises = (
    200 * np.exp(-((X - 0.6)**2 + (Y - 0.4)**2) / 0.005) +
    100 * np.exp(-((X - 0.75)**2 + (Y - 0.65)**2) / 0.003) +
    160 * np.exp(-((X - 0.55)**2 + (Y - 0.85)**2) / 0.0025)
)
von_mises = gaussian_filter(von_mises, sigma=2)

# ----- Total Stress Distribution -----
total_stress = (
    400 * np.exp(-((X - 0.2)**2 + (Y - 0.2)**2) / 0.005) +
    350 * np.exp(-((X - 0.25)**2 + (Y - 0.85)**2) / 0.005)
)
total_stress = gaussian_filter(total_stress, sigma=2)

# ----- Plotting -----
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Von Mises
vm = axs[0].imshow(von_mises, extent=[0, 1, 0, 1], origin='lower', cmap='magma', vmin=0, vmax=200)
axs[0].set_title('Von Mises Stress Distribution', fontsize=14)
axs[0].set_xlabel('Approximate Sample Location x axis (m)', fontsize=11)
axs[0].set_ylabel('Approximate Sample Location z axis (m)', fontsize=11)
axs[0].grid(True, linestyle='--', alpha=0.5)
cbar_vm = plt.colorbar(vm, ax=axs[0])
cbar_vm.set_label('Stress (MPa)')

# Total Stress
ts = axs[1].imshow(total_stress, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', vmin=0, vmax=400)
axs[1].set_title('Total Stress Distribution', fontsize=14)
axs[1].set_xlabel('Approximate Sample Location x axis (m)', fontsize=11)
axs[1].set_ylabel('Approximate Sample Location z axis (m)', fontsize=11)
axs[1].grid(True, linestyle='--', alpha=0.5)
cbar_ts = plt.colorbar(ts, ax=axs[1])
cbar_ts.set_label('Stress (MPa)')

plt.tight_layout()
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev

# High-resolution grid
nx, ny = 300, 300
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Simulated Von Mises stress field
von_mises = (
    200 * np.exp(-((X - 0.6)**2 + (Y - 0.4)**2) / 0.005) +
    100 * np.exp(-((X - 0.75)**2 + (Y - 0.65)**2) / 0.003) +
    160 * np.exp(-((X - 0.55)**2 + (Y - 0.85)**2) / 0.0025)
)
von_mises = gaussian_filter(von_mises, sigma=2)

# Peak coordinates (X, Y) that correspond to high-stress points
peak_points = np.array([
    [0.55, 0.85],
    [0.75, 0.65],
    [0.6, 0.4]
]).T  # Shape: (2, N)

# Fit a smooth spline curve through those points
tck, u = splprep(peak_points, s=0, k=2)
u_fine = np.linspace(0, 1, 500)
x_smooth, y_smooth = splev(u_fine, tck)

# Plot stress map and overlay the curve
plt.figure(figsize=(8, 7))
stress_map = plt.imshow(von_mises, extent=[0, 1, 0, 1], origin='lower', cmap='magma', vmin=0, vmax=200)
plt.plot(x_smooth, y_smooth, color='cyan', linewidth=2, label='High-Stress Path')
plt.scatter(*peak_points, color='white', s=50, zorder=5, label='Stress Peaks')
plt.colorbar(stress_map, label='Stress (MPa)')
plt.title('Von Mises Stress with High-Stress Curve')
plt.xlabel('Approximate Sample Location x axis (m)')
plt.ylabel('Approximate Sample Location z axis (m)')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import io, color
from scipy.interpolate import splprep, splev

# Load the shaped geometry
image_path = r"C:\Users\carmi\Documents\Kingston University\PhD\Susmitha Thesis\image.png"
img = io.imread(image_path)
if img.shape[-1] == 4:
    img = color.rgba2rgb(img)
gray = color.rgb2gray(img)

# Create a mask for the domain
domain_mask = gray < 0.95

# Create a synthetic stress field
stress = np.zeros_like(gray)
height, width = stress.shape

# Define a curved path through the center and leading edge
curve_points = np.array([
    [100, 100],
    [180, 180],
    [260, 250],
    [350, 300],
    [430, 350]
]).T  # Shape: (2, N)

# Fit a spline through the selected points
tck, u = splprep(curve_points, s=0, k=3)
u_fine = np.linspace(0, 1, 500)
x_smooth, y_smooth = splev(u_fine, tck)

# Add synthetic stress along the curve path
def add_stress_peak(field, x, y, magnitude, sigma):
    X, Y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    blob = magnitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return field + blob

for x, y in zip(x_smooth, y_smooth):
    stress = add_stress_peak(stress, x, y, magnitude=250, sigma=8)

# Smooth the entire field for realism
stress = gaussian_filter(stress, sigma=2)

# Apply mask
stress[~domain_mask] = np.nan

# Plot result
plt.figure(figsize=(8, 7))
plt.imshow(stress, cmap='plasma')
plt.plot(x_smooth, y_smooth, color='cyan', linewidth=2, label='Stress Curve')
plt.scatter(*curve_points, color='white', s=40, zorder=5)
plt.colorbar(label='Simulated Stress (MPa)')
plt.title('Stress Distribution Along Curved Path of Geometry')
plt.axis('off')
plt.legend()
plt.tight_layout()
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
import cv2
from scipy.ndimage import gaussian_filter

# Load the image
image_path = r"C:\Users\carmi\Documents\Kingston University\PhD\Susmitha Thesis\image.png"
img = io.imread(image_path)
if img.shape[-1] == 4:
    img = color.rgba2rgb(img)
gray = color.rgb2gray(img)

# Create a binary mask
mask = gray < 0.95
binary = mask.astype(np.uint8) * 255

# Find and draw contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_canvas = np.ones_like(binary) * 255
cv2.drawContours(contour_canvas, contours, -1, (0, 0, 0), 2)

# Simulate stress field
stress = np.zeros_like(gray)

def add_stress_peak(field, x, y, magnitude, sigma):
    X, Y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    blob = magnitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return field + blob

# Apply multiple stress peaks
stress = add_stress_peak(stress, 180, 150, 200, 15)
stress = add_stress_peak(stress, 300, 300, 180, 12)
stress = gaussian_filter(stress, sigma=3)
stress[~mask] = np.nan

# Plot side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Panel 1: Outer Contour
axs[0].imshow(contour_canvas, cmap='gray')
axs[0].set_title("Detected Outer Contour")
axs[0].axis('off')

# Panel 2: Stress Distribution
stress_img = axs[1].imshow(stress, cmap='plasma')
axs[1].set_title("Simulated Stress Distribution Inside Geometry")
axs[1].axis('off')
plt.colorbar(stress_img, ax=axs[1], label='Stress (MPa)')

plt.tight_layout()
plt.show()


# In[ ]:




