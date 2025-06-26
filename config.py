import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Sample data: two classes
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2], [4, 2]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train a linear SVM
model = SVC(kernel='linear')
model.fit(X, y)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='bwr', label='Data points')

# Plot support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
            s=150, facecolors='none', edgecolors='k', linewidths=2, label='Support Vectors')

# Plot decision boundary
w = model.coef_[0]
b = model.intercept_[0]
x_plot = np.linspace(0, 5, 100)
y_plot = -(w[0] * x_plot + b) / w[1]
plt.plot(x_plot, y_plot, 'k--', label='Decision boundary')

plt.legend()
plt.title("Support Vectors in SVM")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()