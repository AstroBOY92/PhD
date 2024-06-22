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


# In[ ]:




