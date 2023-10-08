import matplotlib.pyplot as plt
import random
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([i for i in range(-200, 200)]).reshape(-1, 1)
y = [i*i + random.randrange(-50, 50) for i in range(-200, 200)]

# Create polynomial features
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)


# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

print(model.predict(poly_features.fit_transform( np.array( [3] ).reshape(-1, 1) )))

# Plot the data
plt.scatter(X, y)
plt.plot(X, model.predict(X_poly), color='red')
plt.show()