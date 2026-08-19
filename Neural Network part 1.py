# | Situation                         | Common activation |
# | --------------------------------- | ----------------- |
# | Hidden layers                     | ReLU              | Rectified Linear unit
# | Binary classification output      | Sigmoid           |
# | Multi-class classification output | Softmax           |
# | Some specialized networks         | Tanh              |
# Neuron → Activation → Forward propagation → Loss → Backpropagation → Gradient descent → Epochs → Batches

# 1. What is a neuron? ← we're here
# 2. What are weights? Weight is simply how important input given to a neuron
# 3. What is bias? z = x1w1 + x2w2 + x3w3 +...+ b where x are information and w are weights and b is the bias
# 4. What is an activation function? The activation function decides how the neuron should respond to its calculated value. The famous activation function is Rectified Linear Unit (ReLU(x) = max(0,x))
# Input → Output
# -5    → 0
# -2    → 0
#  0    → 0
#  2    → 2
#  5    → 5
# 5. How does a neuron make a prediction?
# 6. What is a neural network/layers?
# 7. Forward propagation: Take the input and move it forward through the neural network until we get a prediction.
# 8. Loss function: How wrong was the model
# 10. Gradient : How should we change the weights to make it less wrong?
# 9. Backpropagation

# 10. Gradient descent: How should we change the weights to make it less wrong?
# 11. Build one with NumPy
# 12. Build one with PyTorch

# | Concept                 | Job                                     |
# | ----------------------- | --------------------------------------- |
# | **Neuron**              | Performs calculations                   |
# | **Weight**              | Controls input influence                |
# | **Bias**                | Shifts the calculation                  |
# | **Activation**          | Adds non-linearity                      |
# | **Forward propagation** | Produces prediction                     |
# | **Loss**                | Measures error                          |
# | **Gradient**            | Tells how parameters affect loss        |
# | **Backpropagation**     | Calculates gradients backward           |
# | **Gradient descent**    | Uses gradients to reduce loss           |
# | **Learning rate**       | Controls update size                    |
# | **Epoch**               | One complete pass through training data |


# TRAINING
# │
# ▼
# ┌───────────┐
# │   INPUT   │
# └─────┬─────┘
# ↓
# FORWARD
# PROPAGATION
# ↓
# PREDICTION
# ↓
# LOSS
# ↓
# BACKPROPAGATION
# ↓
# GRADIENTS
# ↓
# GRADIENT
# DESCENT
# ↓
# UPDATE
# PARAMETERS
# │
# └──────────→ REPEAT




import numpy as np

# -------------------------
# Training data
# -------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [1]
])

# -------------------------
# Initialize parameters
# -------------------------
np.random.seed(42)

W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros((1, 3))

W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros((1, 1))

learning_rate = 0.1


# -------------------------
# Activation functions
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# -------------------------
# Training
# -------------------------
for epoch in range(10000):

    # ===== Forward propagation =====

    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)  # ReLU

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    # ===== Loss =====

    loss = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))

    # ===== Backpropagation =====

    m = X.shape[0]

   # ==== Calculating the loss according to the Predicted Value ====
   # In fact calculating how wrong we are from the actual value

    dZ2 = A2 - y

    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * (Z1 > 0)

    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # ===== Update parameters =====

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# -------------------------
# Make predictions
# -------------------------
predictions = A2
print("\nPredictions:")
print(predictions)
