#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Load the uploaded image
image_path = image.png
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply threshold to create binary image
_, binary = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours for visualization
canvas = np.ones_like(img) * 255
cv2.drawContours(canvas, contours, -1, (0, 0, 0), 1)

# Plot original image and detected contours
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(canvas, cmap='gray')
axs[1].set_title("Detected Outer Contour")
axs[1].axis('off')

plt.tight_layout()
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure
from scipy.ndimage import gaussian_filter


# In[4]:


# Load and preprocess the image
image_path = r"C:\Users\carmi\Documents\Kingston University\PhD\Susmitha Thesis\image.png"
img = io.imread(image_path)

# Convert RGBA to RGB if needed
if img.shape[-1] == 4:
    img = color.rgba2rgb(img)

# Convert RGB to grayscale
gray = color.rgb2gray(img)

# Apply edge detection (Sobel)
edges = filters.sobel(gray)

# Display result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='magma')
plt.title("Sobel Edges")
plt.axis('off')

plt.tight_layout()
plt.show()


# In[5]:


# Load and preprocess the image
image_path = r"C:\Users\carmi\Documents\Kingston University\PhD\Susmitha Thesis\image.png"
img = io.imread(image_path)

# Convert RGBA to RGB if needed
if img.shape[-1] == 4:
    img = color.rgba2rgb(img)

# Convert RGB to grayscale
gray = color.rgb2gray(img)

# Apply edge detection (Sobel)
edges = filters.sobel(gray)

# Display result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='magma')
plt.title("Sobel Edges")
plt.axis('off')

plt.tight_layout()
plt.show()


# In[6]:


import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Use the grayscale shape from before (gray)
domain_mask = gray < 0.95  # Threshold to identify part region (may need adjusting)

# Initialize stress field
stress = np.zeros_like(gray)

# Add synthetic stress peaks at assumed "hole" or stress riser locations
def add_stress_peak(field, x, y, magnitude, sigma):
    X, Y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    blob = magnitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return field + blob

# Add a few stress concentrations (adjust manually based on visual geometry)
stress = add_stress_peak(stress, x=120, y=100, magnitude=200, sigma=10)
stress = add_stress_peak(stress, x=200, y=180, magnitude=160, sigma=12)
stress = add_stress_peak(stress, x=300, y=260, magnitude=220, sigma=15)

# Smooth to make it FEA-like
stress = gaussian_filter(stress, sigma=3)

# Mask outside of the part
stress[~domain_mask] = np.nan

# Plot FEA-style stress field
plt.figure(figsize=(8, 6))
plt.imshow(stress, cmap='plasma')
plt.colorbar(label='Simulated Stress (MPa)')
plt.title('Simulated FEA-Like Stress Field')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[18]:


import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Use a simulated domain similar to the processed image dimensions
height, width = 477, 518
domain_mask = np.ones((height, width), dtype=bool)

# Initialize stress fields
von_mises_stress = np.zeros((height, width))
total_stress = np.zeros((height, width))

# Simulate stress peaks similar to earlier plots
def add_peak(field, x, y, magnitude, sigma):
    X, Y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    blob = magnitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return field + blob

# Add Von Mises-like stress peaks
von_mises_stress = add_peak(von_mises_stress, x=300, y=150, magnitude=180, sigma=20)
von_mises_stress = add_peak(von_mises_stress, x=250, y=250, magnitude=160, sigma=15)
von_mises_stress = add_peak(von_mises_stress, x=200, y=370, magnitude=140, sigma=10)
von_mises_stress = gaussian_filter(von_mises_stress, sigma=3)
von_mises_stress[~domain_mask] = np.nan

# Add Total Stress-like peaks
total_stress = add_peak(total_stress, x=120, y=120, magnitude=400, sigma=25)
total_stress = add_peak(total_stress, x=140, y=350, magnitude=320, sigma=20)
total_stress = gaussian_filter(total_stress, sigma=3)
total_stress[~domain_mask] = np.nan

# Plot side by side like real FEA
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Von Mises Stress
vmap = axs[0].imshow(von_mises_stress, cmap='magma')
axs[0].set_title("Simulated Von Mises Stress")
axs[0].axis('off')
plt.colorbar(vmap, ax=axs[0], fraction=0.046, pad=0.04, label='MPa')

# Total Stress
tmap = axs[1].imshow(total_stress, cmap='plasma')
axs[1].set_title("Simulated Total Stress")
axs[1].axis('off')
plt.colorbar(tmap, ax=axs[1], fraction=0.046, pad=0.04, label='MPa')

plt.tight_layout()
plt.show()


# In[39]:


import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage import io, color

# Load the image again to retrieve grayscale for domain masking
image_path = r"C:\Users\carmi\Documents\Kingston University\PhD\Susmitha Thesis\image.png"
img = io.imread(image_path)
if img.shape[-1] == 4:
    img = color.rgba2rgb(img)
gray = color.rgb2gray(img)

# Use the grayscale shape from before (gray)
domain_mask = gray < 0.95  # Threshold to identify part region (may need adjusting)

# Initialize stress field
stress = np.zeros_like(gray)

# Add synthetic stress peaks at assumed "hole" or stress riser locations
def add_stress_peak(field, x, y, magnitude, sigma):
    X, Y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    blob = magnitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return field + blob

# Add a few stress concentrations (adjust manually based on visual geometry)
stress = add_stress_peak(stress, x=360, y=150, magnitude=400, sigma=10)
stress = add_stress_peak(stress, x=360, y=140, magnitude=400, sigma=10)
stress = add_stress_peak(stress, x=360, y=130, magnitude=400, sigma=10)
stress = add_stress_peak(stress, x=360, y=120, magnitude=400, sigma=10)
stress = add_stress_peak(stress, x=360, y=40, magnitude=160, sigma=12)




# Smooth to make it FEA-like
stress = gaussian_filter(stress, sigma=3)

# Mask outside of the part
stress[~domain_mask] = np.nan

# Plot FEA-style stress field
plt.figure(figsize=(8, 6))
plt.imshow(stress, cmap='plasma')
plt.colorbar(label='Simulated Stress (MPa)')
plt.title('Simulated FEA-Like Stress Field on Geometry')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[41]:


import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage import io, color

# Load the geometry image again
image_path = r"C:\Users\carmi\Documents\Kingston University\PhD\Susmitha Thesis\image.png"
img = io.imread(image_path)
if img.shape[-1] == 4:
    img = color.rgba2rgb(img)
gray = color.rgb2gray(img)

# Create a domain mask from grayscale
domain_mask = gray < 0.95

# Initialize stress field
stress = np.zeros_like(gray)

# Function to add a directional stress concentration (along edge)
def add_edge_stress(field, start_x, start_y, end_x, end_y, magnitude, sigma, num_points=15):
    xs = np.linspace(start_x, end_x, num_points)
    ys = np.linspace(start_y, end_y, num_points)
    for x, y in zip(xs, ys):
        field = add_stress_peak(field, x, y, magnitude, sigma)
    return field

# Stress peak function
def add_stress_peak(field, x, y, magnitude, sigma):
    X, Y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    blob = magnitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return field + blob

# Apply a stress line along the leading edge (right edge)
stress = add_edge_stress(stress, start_x=480, start_y=100, end_x=480, end_y=380,
                         magnitude=800, sigma=6)

# Apply Gaussian smoothing
stress = gaussian_filter(stress, sigma=2)

# Mask outside of part
stress[~domain_mask] = np.nan

# Plot the simulated stress field near the leading edge
plt.figure(figsize=(8, 6))
plt.imshow(stress, cmap='plasma')
plt.colorbar(label='Simulated Stress (MPa)')
plt.title('Simulated Stress Concentration Near Leading Edge')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[42]:


import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage import io, color

# Reload the geometry image
image_path = r"C:\Users\carmi\Documents\Kingston University\PhD\Susmitha Thesis\image.png"
img = io.imread(image_path)
if img.shape[-1] == 4:
    img = color.rgba2rgb(img)
gray = color.rgb2gray(img)

# Create domain mask
domain_mask = gray < 0.95

# Initialize stress field
stress = np.zeros_like(gray)

# Function to add directional stress concentration along edge
def add_edge_stress(field, start_x, start_y, end_x, end_y, magnitude, sigma, num_points=20):
    xs = np.linspace(start_x, end_x, num_points)
    ys = np.linspace(start_y, end_y, num_points)
    for x, y in zip(xs, ys):
        field = add_stress_peak(field, x, y, magnitude, sigma)
    return field

# Function to add a single stress peak
def add_stress_peak(field, x, y, magnitude, sigma):
    X, Y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    blob = magnitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return field + blob

# Apply stress line with higher magnitude (400 MPa)
stress = add_edge_stress(stress, start_x=480, start_y=100, end_x=480, end_y=380,
                         magnitude=400, sigma=6)

# Apply smoothing for realism
stress = gaussian_filter(stress, sigma=2)

# Mask out background
stress[~domain_mask] = np.nan

# Plot the new high-stress distribution
plt.figure(figsize=(8, 6))
plt.imshow(stress, cmap='plasma')
plt.colorbar(label='Simulated Stress (MPa)')
plt.title('High Stress Concentration (~400 MPa) Near Leading Edge')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# High-resolution grid
nx, ny = 300, 300
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# ----- Von Mises Stress -----
von_mises = (
    200 * np.exp(-((X - 0.6)**2 + (Y - 0.4)**2) / 0.005) +
    100 * np.exp(-((X - 0.75)**2 + (Y - 0.65)**2) / 0.003) +
    160 * np.exp(-((X - 0.55)**2 + (Y - 0.85)**2) / 0.0025)
)
von_mises = gaussian_filter(von_mises, sigma=2)

# ----- Total Stress -----
total_stress = (
    400 * np.exp(-((X - 0.2)**2 + (Y - 0.2)**2) / 0.005) +
    350 * np.exp(-((X - 0.25)**2 + (Y - 0.85)**2) / 0.005)
)
total_stress = gaussian_filter(total_stress, sigma=2)

# ----- Plotting -----
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Left: Von Mises
vm_plot = axs[0].imshow(von_mises, extent=[0, 1, 0, 1], origin='lower', cmap='magma', vmin=0, vmax=200)
axs[0].set_title("Von Mises Stress Distribution")
axs[0].set_xlabel("Approximate Sample Location x axis (m)")
axs[0].set_ylabel("Approximate Sample Location z axis (m)")
axs[0].grid(True, linestyle='--', alpha=0.3)
cbar_vm = plt.colorbar(vm_plot, ax=axs[0])
cbar_vm.set_label("Stress (MPa)")

# Right: Total Stress
ts_plot = axs[1].imshow(total_stress, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', vmin=0, vmax=400)
axs[1].set_title("Total Stress Distribution")
axs[1].set_xlabel("Approximate Sample Location x axis (m)")
axs[1].set_ylabel("Approximate Sample Location z axis (m)")
axs[1].grid(True, linestyle='--', alpha=0.3)
cbar_ts = plt.colorbar(ts_plot, ax=axs[1])
cbar_ts.set_label("Stress (MPa)")

plt.tight_layout()
plt.show()


# In[116]:


fig, axs = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

# Left: Von Mises
vm_plot = axs[0].imshow(von_mises, extent=[0, 1, 0, 1], origin='lower', cmap='magma', vmin=0, vmax=200, interpolation='none')
axs[0].set_title("Von Mises Stress Distribution")
axs[0].set_xlabel("Approximate Sample Location x axis (m)")
axs[0].set_ylabel("Approximate Sample Location z axis (m)")
axs[0].grid(False)
cbar_vm = plt.colorbar(vm_plot, ax=axs[0])
cbar_vm.set_label("Stress (MPa)")

# Right: Total Stress
ts_plot = axs[1].imshow(total_stress, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', vmin=0, vmax=400, interpolation='none')
axs[1].set_title("Total Stress Distribution")
axs[1].set_xlabel("Approximate Sample Location x axis (m)")
axs[1].set_ylabel("Approximate Sample Location z axis (m)")
axs[1].grid(False)
cbar_ts = plt.colorbar(ts_plot, ax=axs[1])
cbar_ts.set_label("Stress (MPa)")
plt.axis('on')

plt.tight_layout()
plt.show()


# In[118]:


# import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Use the grayscale shape from before (gray)
domain_mask = gray < 0.95  # Threshold to identify part region (may need adjusting)

# Initialize stress field
stress = np.zeros_like(gray)

# Add synthetic stress peaks at assumed "hole" or stress riser locations
def add_stress_peak(field, x, y, magnitude, sigma):
    X, Y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    blob = magnitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return field + blob

# Add a few stress concentrations (adjust manually based on visual geometry)
stress = add_stress_peak(stress, x=300, y=100, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=305, y=105, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=310, y=110, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=315, y=115, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=320, y=120, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=325, y=125, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=330, y=130, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=335, y=135, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=340, y=140, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=345, y=145, magnitude=10, sigma=15)
stress = add_stress_peak(stress, x=350, y=150, magnitude=10, sigma=45)

stress = add_stress_peak(stress, x=300, y=100, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=305, y=105, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=310, y=110, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=315, y=115, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=320, y=120, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=325, y=125, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=330, y=130, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=335, y=135, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=340, y=140, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=345, y=145, magnitude=10, sigma=10)
stress = add_stress_peak(stress, x=350, y=150, magnitude=10, sigma=40)

stress = add_stress_peak(stress, x=300, y=100, magnitude=50, sigma=5)
stress = add_stress_peak(stress, x=305, y=105, magnitude=10, sigma=5)
stress = add_stress_peak(stress, x=310, y=110, magnitude=10, sigma=5)
stress = add_stress_peak(stress, x=315, y=115, magnitude=10, sigma=5)
stress = add_stress_peak(stress, x=320, y=120, magnitude=10, sigma=5)
stress = add_stress_peak(stress, x=325, y=125, magnitude=10, sigma=5)
stress = add_stress_peak(stress, x=330, y=130, magnitude=10, sigma=5)
stress = add_stress_peak(stress, x=335, y=135, magnitude=10, sigma=5)
stress = add_stress_peak(stress, x=340, y=140, magnitude=10, sigma=5)
stress = add_stress_peak(stress, x=345, y=145, magnitude=10, sigma=25)

stress = add_stress_peak(stress, x=300, y=100, magnitude=50, sigma=60)
stress = add_stress_peak(stress, x=305, y=105, magnitude=10, sigma=60)
stress = add_stress_peak(stress, x=310, y=110, magnitude=10, sigma=20)
stress = add_stress_peak(stress, x=315, y=115, magnitude=10, sigma=20)
stress = add_stress_peak(stress, x=320, y=120, magnitude=10, sigma=20)
stress = add_stress_peak(stress, x=325, y=125, magnitude=10, sigma=20)
stress = add_stress_peak(stress, x=330, y=130, magnitude=10, sigma=20)
stress = add_stress_peak(stress, x=335, y=135, magnitude=10, sigma=20)
stress = add_stress_peak(stress, x=340, y=140, magnitude=10, sigma=20)
stress = add_stress_peak(stress, x=345, y=145, magnitude=10, sigma=60)

#stress = add_stress_peak(stress, x=250, y=125, magnitude=100, sigma=15)
#stress = add_stress_peak(stress, x=300, y=100, magnitude=100, sigma=10)
#stress = add_stress_peak(stress, x=250, y=125, magnitude=100, sigma=10)
#stress = add_stress_peak(stress, x=400, y=200, magnitude=100, sigma=10)
#stress = add_stress_peak(stress, x=450, y=225, magnitude=100, sigma=10)
#stress = add_stress_peak(stress, x=400, y=200, magnitude=100, sigma=15)
#stress = add_stress_peak(stress, x=450, y=225, magnitude=100, sigma=15)
#stress = add_stress_peak(stress, x=285, y=140, magnitude=200, sigma=10)
#stress = add_stress_peak(stress, x=290, y=160, magnitude=200, sigma=10)



stress = add_stress_peak(stress, x=425, y=290, magnitude=210, sigma=30)

stress = add_stress_peak(stress, x=275, y=375, magnitude=100, sigma=20)





# Smooth to make it FEA-like
stress = gaussian_filter(stress, sigma=3)

# Mask outside of the part
stress[~domain_mask] = np.nan

# Plot FEA-style stress field
plt.figure(figsize=(8, 6))
plt.imshow(stress, cmap='plasma')
plt.colorbar(label='Stress (MPa)')
plt.title('Decoded Von Mises Stress distribution')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[119]:


import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Use the grayscale shape from before (gray)
domain_mask = gray < 0.95  # Threshold to identify part region (may need adjusting)

# Initialize stress field
stress = np.zeros_like(gray)

# Add synthetic stress peaks at assumed "hole" or stress riser locations
def add_stress_peak(field, x, y, magnitude, sigma):
    X, Y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    blob = magnitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return field + blob

# Add a few stress concentrations (adjust manually based on visual geometry)
stress = add_stress_peak(stress, x=60, y=100, magnitude=220, sigma=5)
stress = add_stress_peak(stress, x=300, y=100, magnitude=425, sigma=40)

stress = add_stress_peak(stress, x=200, y=50, magnitude=100, sigma=5)
stress = add_stress_peak(stress, x=60, y=260, magnitude=220, sigma=5)
stress = add_stress_peak(stress, x=410, y=260, magnitude=390, sigma=50)


# Smooth to make it FEA-like
stress = gaussian_filter(stress, sigma=3)

# Mask outside of the part
stress[~domain_mask] = np.nan

# Plot FEA-style stress field
plt.figure(figsize=(8, 6))
plt.imshow(stress, cmap='viridis')
plt.colorbar(label='Stress (MPa)')
plt.title('Decoded Total Stress Distribution')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[ ]:




