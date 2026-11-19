import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size, hidden_size, output_size = 2, 4, 1

np.random.seed(42)
w1 = np.random.randn(input_size, hidden_size) * 0.5
b1 = np.zeros([1, 4])
w2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.zeros([1, 1])

def forward_pass(X):
    z1 = X @ w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    return a2, [z1, a1, z2, a2]

def cost(pred, actual):
    return (np.mean((actual-pred)**2)) / 2

def cost_derivative(pred, actual):
    return pred-actual

def sigmoid_derivative(pred):
    return pred * (1-pred)

def backprop(X, y, cache, w1, w2):
    z1, a1, z2, a2 = cache
    m = X.shape[0]
    dL_da2 = a2-y
    dL_dz2 = dL_da2 * sigmoid_derivative(a2)
    dL_dw2 = (a1.T @ dL_dz2) / m
    dL_db2 = np.sum(dL_dz2, axis = 0, keepdims = True)
    dL_da1 = dL_dz2 @ w2.T
    dL_dz1 = dL_da1 * sigmoid_derivative(a1)
    dL_dw1 = (X.T @ dL_dz1) / m
    dL_db1 = np.sum(dL_dz1, axis = 0, keepdims = True) / m
    return dL_dw1, dL_db1, dL_dw2, dL_db2

learning_rate = 0.01
iterations = 10000

for i in range(iterations):
    a2, cache = forward_pass(X)
    dw1, db1, dw2, db2 = backprop(X, y, cache, w1, w2)
    w1 -= dw1 * learning_rate
    b1 -= db1 * learning_rate
    w2 -= dw2 * learning_rate
    b2 -= db2 * learning_rate

final_predictions, _ = forward_pass(X, w1, b1, w2, b2)
print("\nTarget y:\n", y)
print("Predictions:\n", np.round(final_predictions, 3))