import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Class +1
X1 = np.random.randn(50, 2) + np.array([2, 2])
y1 = np.ones(50)

# Class -1
X2 = np.random.randn(50, 2) + np.array([-2, -2])
y2 = -np.ones(50)

X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# --> Complete the code by plotting the points with colors correponding
# to their classes

class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialization
        self.w = np.zeros(n_features)
        self.b = 0

        # --> Complete the code by implementing the update loop here

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

model = Perceptron(lr=0.1, epochs=50)
model.fit(X, y)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.2)
    # --> Complete by plotting the points colored by class as before
    plt.show()
plot_decision_boundary(model, X, y)

X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

y_xor = np.array([-1, 1, 1, -1])

model = Perceptron(lr=0.1, epochs=100)
model.fit(X_xor, y_xor)

predictions = model.predict(X_xor)
print("Predictions:", predictions)
