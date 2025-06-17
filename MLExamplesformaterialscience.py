#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches

# Data for supervised learning (Linear Regression example)
X_supervised, y_supervised = make_blobs(n_samples=100, centers=2, n_features=1, random_state=42)
model_supervised = LinearRegression().fit(X_supervised, y_supervised)
y_pred_supervised = model_supervised.predict(X_supervised)

# Data for unsupervised learning (Clustering example)
X_unsupervised, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
model_unsupervised = KMeans(n_clusters=3).fit(X_unsupervised)
y_pred_unsupervised = model_unsupervised.predict(X_unsupervised)

# Data for reinforcement learning (Sample example, no actual RL implementation)
states = np.arange(1, 11)
actions = np.random.choice(['left', 'right'], size=10)
rewards = np.random.randint(-10, 10, size=10)

# Plotting Supervised Learning (Linear Regression)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_supervised, y_supervised, color='blue', label='Data Points')
plt.plot(X_supervised, y_pred_supervised, color='red', label='Linear Fit')
plt.title('Supervised Learning\n(Linear Regression)')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()

# Plotting Unsupervised Learning (KMeans Clustering)
plt.subplot(1, 3, 2)
plt.scatter(X_unsupervised[:, 0], X_unsupervised[:, 1], c=y_pred_unsupervised, cmap='viridis')
plt.title('Unsupervised Learning\n(KMeans Clustering)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plotting Reinforcement Learning (Example)
plt.subplot(1, 3, 3)
colors = {'left': 'blue', 'right': 'green'}
plt.bar(states, rewards, color=[colors[action] for action in actions])
blue_patch = mpatches.Patch(color='blue', label='Left')
green_patch = mpatches.Patch(color='green', label='Right')
plt.legend(handles=[blue_patch, green_patch])
plt.title('Reinforcement Learning\n(Example States and Rewards)')
plt.xlabel('State')
plt.ylabel('Reward')

plt.tight_layout()
plt.show()


# In[ ]:




