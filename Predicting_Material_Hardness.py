#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
num_samples = 100

# Features: Let's say we have two features for simplicity
feature_1 = np.random.uniform(0, 100, num_samples)
feature_2 = np.random.uniform(0, 50, num_samples)

# Hardness (target variable): a linear combination of feature_1 and feature_2 with some noise
hardness = 3 * feature_1 + 2 * feature_2 + np.random.normal(0, 10, num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Feature_1': feature_1,
    'Feature_2': feature_2,
    'Hardness': hardness
})

# Split the data into training and testing sets
X = data[['Feature_1', 'Feature_2']]
y = data['Hardness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel('Actual Hardness')
plt.ylabel('Predicted Hardness')
plt.title('Actual vs Predicted Hardness')
plt.show()


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
num_samples = 100

# Features: Let's say we have two features for simplicity
feature_1 = np.random.uniform(0, 100, num_samples)
feature_2 = np.random.uniform(0, 50, num_samples)

# Hardness (target variable): a linear combination of feature_1 and feature_2 with some noise
hardness = 3 * feature_1 + 2 * feature_2 + np.random.normal(0, 10, num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Feature_1': feature_1,
    'Feature_2': feature_2,
    'Hardness': hardness
})

# Split the data into training and testing sets
X = data[['Feature_1', 'Feature_2']]
y = data['Hardness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a standard linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Train a Ridge regression model (regularized least squares)
ridge_model = Ridge(alpha=1.0)  # alpha is the regularization strength
ridge_model.fit(X_train, y_train)

# Predict on the test set using both models
y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the models
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Output the results
print("Linear Regression:")
print(f"Mean Squared Error: {mse_linear:.2f}")
print(f"R-squared: {r2_linear:.2f}")

print("\nRidge Regression:")
print(f"Mean Squared Error: {mse_ridge:.2f}")
print(f"R-squared: {r2_ridge:.2f}")

# Plot the results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel('Actual Hardness')
plt.ylabel('Predicted Hardness')
plt.title('Linear Regression: Actual vs Predicted Hardness')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel('Actual Hardness')
plt.ylabel('Predicted Hardness')
plt.title('Ridge Regression: Actual vs Predicted Hardness')

plt.tight_layout()
plt.show()


# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
num_samples = 100

# Features: Let's say we have two features for simplicity
feature_1 = np.random.uniform(0, 100, num_samples)
feature_2 = np.random.uniform(0, 50, num_samples)

# Hardness (target variable): a linear combination of feature_1 and feature_2 with some noise
hardness = 3 * feature_1 + 2 * feature_2 + np.random.normal(0, 10, num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Feature_1': feature_1,
    'Feature_2': feature_2,
    'Hardness': hardness
})

# Split the data into training and testing sets
X = data[['Feature_1', 'Feature_2']]
y = data['Hardness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a standard linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Train a Ridge regression model (regularized least squares)
ridge_model = Ridge(alpha=1.0)  # alpha is the regularization strength
ridge_model.fit(X_train, y_train)

# Predict on the test set using both models
y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the models
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Output the results
print("Linear Regression:")
print(f"Mean Squared Error: {mse_linear:.2f}")
print(f"R-squared: {r2_linear:.2f}")

print("\nRidge Regression:")
print(f"Mean Squared Error: {mse_ridge:.2f}")
print(f"R-squared: {r2_ridge:.2f}")

# Plot the results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel('Actual Hardness')
plt.ylabel('Predicted Hardness')
plt.title('Linear Regression: Actual vs Predicted Hardness')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel('Actual Hardness')
plt.ylabel('Predicted Hardness')
plt.title('Ridge Regression: Actual vs Predicted Hardness')

plt.tight_layout()
plt.show()


# In[ ]:




