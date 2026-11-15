
"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 4 — NEURAL NETWORK FROM SCRATCH
================================================================================

PURPOSE
-------
Lesson 3 taught us how a simple model learns parameters using gradient descent.

We used:

    y = wx + b

and learned:

    loss
    derivative
    gradient
    gradient descent
    parameter updates

However, real computer-vision models need much more expressive functions.

A real neural network contains:

    multiple layers
    weights
    biases
    nonlinear activation functions
    forward propagation
    loss calculation
    backpropagation
    gradient descent

This lesson builds a small neural network COMPLETELY FROM SCRATCH.

We will NOT use:

    PyTorch
    TensorFlow
    Keras
    automatic differentiation
    pretrained models
    pretrained weights

Only:

    Python
    NumPy
    Matplotlib

The goal is not to build the largest neural network.

The goal is to understand exactly what happens inside one.

-------------------------------------------------------------------------------
WHAT YOU WILL LEARN
-------------------------------------------------------------------------------

1. Why a single linear layer is limited
2. Why nonlinear activation functions are necessary
3. ReLU
4. Sigmoid
5. Forward propagation
6. Hidden layers
7. Neural-network parameters
8. Binary classification
9. Binary cross-entropy
10. Backpropagation
11. Chain rule
12. Gradients through multiple layers
13. Gradient descent for neural networks
14. Numerical gradient checking
15. Training a neural network from scratch
16. Decision boundaries
17. Loss curves
18. Real-world computer-vision interpretation

-------------------------------------------------------------------------------
COURSE ROADMAP
-------------------------------------------------------------------------------

Phase 0  -> Mathematical foundations
Phase 1  -> Image and tensor representation
Phase 2  -> Neural networks from scratch
Phase 3  -> CNNs and visual recognition
Phase 4  -> Autoencoders and VAE
Phase 5  -> GANs
Phase 6  -> Attention and Transformers
Phase 7  -> Generative modeling fundamentals
Phase 8  -> Diffusion models from scratch
Phase 9  -> Latent diffusion
Phase 10 -> Conditioning and cross-attention
Phase 11 -> Own text-to-image model
Phase 12 -> Advanced diffusion
Phase 13 -> Scaling and training infrastructure

Current progression:

    Lesson 1
        image -> numerical data

    Lesson 2
        vectors -> matrices -> tensors

    Lesson 3
        loss -> gradients -> gradient descent

    Lesson 4
        multi-layer neural network

    Lesson 5+
        deeper vision networks

-------------------------------------------------------------------------------
REAL-WORLD PROBLEM
-------------------------------------------------------------------------------

Consider a robot camera observing objects.

Suppose we extract two visual features:

    x1 = object brightness
    x2 = object size

We want to classify the object:

    class 0 = object A
    class 1 = object B

A single linear model can only create a linear decision boundary.

That is insufficient for many real-world visual patterns.

A neural network solves this by combining:

    linear transformation
            +
    nonlinear activation
            +
    another linear transformation

This allows the model to approximate much more complicated functions.

-------------------------------------------------------------------------------
WHY NONLINEARITY IS NECESSARY
-------------------------------------------------------------------------------

Suppose we stack two linear layers:

    h = W1 x + b1

    y = W2 h + b2

Substitute the first equation:

    y = W2(W1x + b1) + b2

which becomes:

    y = W2W1x + W2b1 + b2

This is STILL just a linear transformation.

Therefore:

    linear layer
        +
    linear layer

does NOT create a genuinely deeper nonlinear model.

We need an activation function between layers.

For example:

    h = ReLU(W1x + b1)

    y = W2h + b2

Now the network can represent nonlinear functions.

-------------------------------------------------------------------------------
RELU
-------------------------------------------------------------------------------

ReLU means:

    Rectified Linear Unit

Definition:

    ReLU(x) = max(0, x)

Therefore:

    x < 0 -> 0
    x > 0 -> x

Derivative:

    d ReLU / dx = 0    when x < 0
                  1    when x > 0

ReLU is widely used because it is simple and computationally efficient.

-------------------------------------------------------------------------------
SIGMOID
-------------------------------------------------------------------------------

The sigmoid function is:

                    1
    sigmoid(x) = ---------
                 1 + e^(-x)

Its output is between:

    0 and 1

Therefore it is useful for binary classification.

For example:

    sigmoid(output) = 0.92

can be interpreted as approximately:

    92% probability of class 1

Its derivative can be written as:

    sigmoid'(x) = sigmoid(x)(1-sigmoid(x))

-------------------------------------------------------------------------------
NETWORK ARCHITECTURE
-------------------------------------------------------------------------------

We will build:

    Input
      |
      | 2 features
      v
    Hidden Layer
      |
      | 16 neurons
      v
    ReLU
      |
      v
    Output Layer
      |
      | 1 neuron
      v
    Sigmoid
      |
      v
    probability

Architecture:

    2 -> 16 -> 1

This is a small multilayer perceptron.

-------------------------------------------------------------------------------
FORWARD PROPAGATION
-------------------------------------------------------------------------------

First layer:

    z1 = XW1 + b1

Then:

    a1 = ReLU(z1)

Second layer:

    z2 = a1W2 + b2

Then:

    y_hat = sigmoid(z2)

The complete forward pass is:

    X
     |
     v
    Z1 = XW1+b1
     |
     v
    A1 = ReLU(Z1)
     |
     v
    Z2 = A1W2+b2
     |
     v
    Y_hat = sigmoid(Z2)

-------------------------------------------------------------------------------
BINARY CROSS-ENTROPY
-------------------------------------------------------------------------------

For binary classification we use:

    L = -[y log(y_hat)
          +
          (1-y)log(1-y_hat)]

For multiple samples:

    L = mean(
        -y log(y_hat)
        -(1-y)log(1-y_hat)
    )

The loss becomes small when the predicted probability is close to the
correct class.

Examples:

    y = 1
    y_hat = 0.99

    -> very small loss

but:

    y = 1
    y_hat = 0.01

    -> very large loss

-------------------------------------------------------------------------------
BACKPROPAGATION
-------------------------------------------------------------------------------

Backpropagation calculates:

    dL/dW1
    dL/db1
    dL/dW2
    dL/db2

It uses the chain rule.

Conceptually:

    loss
      |
      v
    output layer
      |
      v
    hidden layer
      |
      v
    input

The gradient information travels backward through the network.

-------------------------------------------------------------------------------
CHAIN RULE
-------------------------------------------------------------------------------

If:

    y = f(g(x))

then:

    dy/dx = dy/dg * dg/dx

Neural networks contain many nested functions.

Therefore backpropagation is essentially an efficient application of the
chain rule through the computational graph.

-------------------------------------------------------------------------------
OUTPUT-LAYER GRADIENT
-------------------------------------------------------------------------------

For:

    sigmoid(z2)

combined with binary cross-entropy, the derivative simplifies to:

    dL/dz2 = y_hat - y

This is one of the most useful simplifications in binary classification.

Then:

    dL/dW2 = A1^T dL/dz2

and:

    dL/db2 = sum(dL/dz2)

-------------------------------------------------------------------------------
HIDDEN-LAYER GRADIENT
-------------------------------------------------------------------------------

First:

    dL/dA1 = dL/dz2 W2^T

Then ReLU:

    dL/dZ1 = dL/dA1 * ReLU'(Z1)

Then:

    dL/dW1 = X^T dL/dZ1

and:

    dL/db1 = sum(dL/dZ1)

-------------------------------------------------------------------------------
PARAMETER UPDATE
-------------------------------------------------------------------------------

For every parameter:

    parameter <- parameter
                 - learning_rate * gradient

Therefore:

    W1 <- W1 - lr*dW1

    b1 <- b1 - lr*db1

    W2 <- W2 - lr*dW2

    b2 <- b2 - lr*db2

This is the same gradient-descent principle from Lesson 3.

The difference is that now we have many parameters and must propagate
gradients through multiple layers.

-------------------------------------------------------------------------------
REAL-WORLD CONNECTION
-------------------------------------------------------------------------------

This small network is not yet a CNN.

But the conceptual training loop is already the same:

    image
      |
      v
    neural network
      |
      v
    prediction
      |
      v
    loss
      |
      v
    backpropagation
      |
      v
    gradients
      |
      v
    parameter update

CNNs later replace the simple fully-connected representation with
convolutional operations.

-------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. ACTIVATION FUNCTIONS
# =============================================================================

def relu(x):
    """
    ReLU:

        ReLU(x) = max(0,x)
    """

    return np.maximum(
        0.0,
        x,
    )


def relu_derivative(x):
    """
    Derivative of ReLU.

    For x > 0:

        derivative = 1

    For x < 0:

        derivative = 0

    At exactly zero, we use zero for this implementation.
    """

    return (
        x > 0.0
    ).astype(np.float64)


def sigmoid(x):
    """
    Numerically stable sigmoid.

        sigmoid(x) = 1 / (1 + exp(-x))

    We implement separate positive/negative branches to reduce overflow
    problems for very large values.
    """

    result = np.empty_like(
        x,
        dtype=np.float64,
    )

    positive = x >= 0.0
    negative = ~positive

    result[positive] = (
        1.0
        / (
            1.0
            + np.exp(-x[positive])
        )
    )

    exp_x = np.exp(
        x[negative]
    )

    result[negative] = (
        exp_x
        / (
            1.0
            + exp_x
        )
    )

    return result


# =============================================================================
# 2. LOSS FUNCTION
# =============================================================================

def binary_cross_entropy(
    y_pred,
    y_true,
):
    """
    Binary cross-entropy.

        L = mean(
            -y log(p)
            -(1-y)log(1-p)
        )

    Predictions are clipped slightly away from 0 and 1 to avoid log(0).
    """

    epsilon = 1e-12

    y_pred_clipped = np.clip(
        y_pred,
        epsilon,
        1.0 - epsilon,
    )

    loss = -np.mean(
        y_true
        * np.log(y_pred_clipped)
        +
        (1.0 - y_true)
        * np.log(1.0 - y_pred_clipped)
    )

    return loss


# =============================================================================
# 3. SYNTHETIC NONLINEAR CLASSIFICATION DATA
# =============================================================================

def create_dataset(
    seed=42,
):
    """
    Create a nonlinear two-class dataset.

    We use two-dimensional points.

    Class 0:
        points inside a central region

    Class 1:
        points outside that region

    The exact pattern is intentionally nonlinear.

    A straight line cannot separate the two classes perfectly.

    This gives the neural network a real reason to use nonlinear hidden
    representations.
    """

    rng = np.random.default_rng(
        seed
    )

    n_samples = 600

    X = rng.uniform(
        -1.5,
        1.5,
        size=(n_samples, 2),
    )

    radius = np.sqrt(
        X[:, 0] ** 2
        +
        X[:, 1] ** 2
    )

    y = (
        radius > 0.75
    ).astype(
        np.float64
    ).reshape(
        -1,
        1,
    )

    return X, y


# =============================================================================
# 4. TRAIN/TEST SPLIT
# =============================================================================

def train_test_split(
    X,
    y,
    test_ratio=0.2,
    seed=123,
):
    """
    Randomly split the dataset into training and testing subsets.
    """

    rng = np.random.default_rng(
        seed
    )

    indices = np.arange(
        len(X)
    )

    rng.shuffle(
        indices
    )

    split = int(
        len(X)
        * (1.0 - test_ratio)
    )

    train_indices = (
        indices[:split]
    )

    test_indices = (
        indices[split:]
    )

    return (
        X[train_indices],
        y[train_indices],
        X[test_indices],
        y[test_indices],
    )


# =============================================================================
# 5. NETWORK INITIALIZATION
# =============================================================================

def initialize_parameters(
    input_size=2,
    hidden_size=16,
    output_size=1,
    seed=7,
):
    """
    Initialize network parameters.

    Architecture:

        2 -> 16 -> 1

    We use He-style scaling for the ReLU layer:

        std ~ sqrt(2 / fan_in)

    This is not a pretrained initialization.

    Everything starts from random values.
    """

    rng = np.random.default_rng(
        seed
    )

    W1 = (
        rng.normal(
            0.0,
            np.sqrt(
                2.0 / input_size
            ),
            size=(
                input_size,
                hidden_size,
            ),
        )
    )

    b1 = np.zeros(
        (
            1,
            hidden_size,
        )
    )

    W2 = (
        rng.normal(
            0.0,
            np.sqrt(
                2.0 / hidden_size
            ),
            size=(
                hidden_size,
                output_size,
            ),
        )
    )

    b2 = np.zeros(
        (
            1,
            output_size,
        )
    )

    return W1, b1, W2, b2


# =============================================================================
# 6. FORWARD PASS
# =============================================================================

def forward(
    X,
    W1,
    b1,
    W2,
    b2,
):
    """
    Perform forward propagation.

        Z1 = XW1 + b1
        A1 = ReLU(Z1)

        Z2 = A1W2 + b2
        Y  = sigmoid(Z2)

    Cache intermediate values because backpropagation needs them.
    """

    Z1 = (
        X @ W1
        + b1
    )

    A1 = relu(
        Z1
    )

    Z2 = (
        A1 @ W2
        + b2
    )

    Y_hat = sigmoid(
        Z2
    )

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "Y_hat": Y_hat,
    }

    return Y_hat, cache


# =============================================================================
# 7. BACKPROPAGATION
# =============================================================================

def backward(
    y_true,
    cache,
    W2,
):
    """
    Backpropagation through the network.

    Network:

        X
         |
        W1
         |
        ReLU
         |
        W2
         |
        sigmoid
         |
        loss

    The derivative of binary cross-entropy combined with sigmoid is:

        dL/dZ2 = Y_hat - Y

    Then:

        dW2 = A1^T @ dZ2

        db2 = sum(dZ2)

        dA1 = dZ2 @ W2^T

        dZ1 = dA1 * ReLU'(Z1)

        dW1 = X^T @ dZ1

        db1 = sum(dZ1)

    All gradients are averaged over the batch.
    """

    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Y_hat = cache["Y_hat"]

    batch_size = X.shape[0]

    # -------------------------------------------------------------------------
    # Output gradient
    # -------------------------------------------------------------------------

    dZ2 = (
        Y_hat
        - y_true
    )

    # -------------------------------------------------------------------------
    # Gradient of W2
    # -------------------------------------------------------------------------

    dW2 = (
        A1.T @ dZ2
    ) / batch_size

    # -------------------------------------------------------------------------
    # Gradient of b2
    # -------------------------------------------------------------------------

    db2 = np.sum(
        dZ2,
        axis=0,
        keepdims=True,
    ) / batch_size

    # -------------------------------------------------------------------------
    # Propagate gradient into hidden layer
    # -------------------------------------------------------------------------

    dA1 = (
        dZ2 @ W2.T
    )

    # -------------------------------------------------------------------------
    # ReLU derivative
    # -------------------------------------------------------------------------

    dZ1 = (
        dA1
        * relu_derivative(Z1)
    )

    # -------------------------------------------------------------------------
    # Gradient of W1
    # -------------------------------------------------------------------------

    dW1 = (
        X.T @ dZ1
    ) / batch_size

    # -------------------------------------------------------------------------
    # Gradient of b1
    # -------------------------------------------------------------------------

    db1 = np.sum(
        dZ1,
        axis=0,
        keepdims=True,
    ) / batch_size

    return (
        dW1,
        db1,
        dW2,
        db2,
    )


# =============================================================================
# 8. PARAMETER UPDATE
# =============================================================================

def update_parameters(
    W1,
    b1,
    W2,
    b2,
    dW1,
    db1,
    dW2,
    db2,
    learning_rate,
):
    """
    Gradient-descent parameter update.

        parameter_new =
            parameter_old
            - learning_rate * gradient
    """

    W1 = (
        W1
        - learning_rate * dW1
    )

    b1 = (
        b1
        - learning_rate * db1
    )

    W2 = (
        W2
        - learning_rate * dW2
    )

    b2 = (
        b2
        - learning_rate * db2
    )

    return (
        W1,
        b1,
        W2,
        b2,
    )


# =============================================================================
# 9. PREDICTION
# =============================================================================

def predict(
    X,
    W1,
    b1,
    W2,
    b2,
    threshold=0.5,
):
    """
    Convert probabilities into binary class predictions.
    """

    probabilities, _ = forward(
        X,
        W1,
        b1,
        W2,
        b2,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.float64
    )

    return (
        probabilities,
        predictions,
    )


# =============================================================================
# 10. ACCURACY
# =============================================================================

def accuracy(
    y_true,
    y_pred,
):
    """
    Classification accuracy.
    """

    return np.mean(
        y_true == y_pred
    )


# =============================================================================
# 11. NUMERICAL GRADIENT CHECK
# =============================================================================

def numerical_gradient(
    loss_function,
    parameter,
    h=1e-5,
):
    """
    Numerically estimate the gradient of a scalar loss with respect to every
    element of a parameter array.

    Central difference:

        dL/dtheta_i ≈
            [L(theta_i+h)-L(theta_i-h)]/(2h)

    This is computationally expensive.

    We use it only for gradient verification, not for training.
    """

    gradient = np.zeros_like(
        parameter
    )

    iterator = np.nditer(
        parameter,
        flags=[
            "multi_index",
            "refs_ok",
        ],
        op_flags=[
            "readwrite"
        ],
    )

    while not iterator.finished:

        index = iterator.multi_index

        original_value = (
            parameter[index]
        )

        parameter[index] = (
            original_value + h
        )

        loss_plus = (
            loss_function()
        )

        parameter[index] = (
            original_value - h
        )

        loss_minus = (
            loss_function()
        )

        parameter[index] = (
            original_value
        )

        gradient[index] = (
            loss_plus - loss_minus
        ) / (
            2.0 * h
        )

        iterator.iternext()

    return gradient


# =============================================================================
# 12. GRADIENT CHECK
# =============================================================================

def gradient_check():
    """
    Compare analytical backpropagation gradients against numerical gradients.

    We intentionally use a very small network and small dataset.

    This is a test of the implementation itself.

    A large neural network should NOT be gradient-checked this way during
    normal training because numerical differentiation is extremely expensive.
    """

    print("\n" + "=" * 80)
    print("GRADIENT CHECK")
    print("=" * 80)

    rng = np.random.default_rng(
        999
    )

    X = rng.normal(
        0.0,
        0.5,
        size=(5, 2),
    )

    y = (
        rng.random(
            (5, 1)
        )
        > 0.5
    ).astype(
        np.float64
    )

    W1, b1, W2, b2 = (
        initialize_parameters(
            input_size=2,
            hidden_size=3,
            output_size=1,
            seed=55,
        )
    )

    y_hat, cache = forward(
        X,
        W1,
        b1,
        W2,
        b2,
    )

    analytical_gradients = backward(
        y,
        cache,
        W2,
    )

    def loss_with_current_parameters():
        predictions, _ = forward(
            X,
            W1,
            b1,
            W2,
            b2,
        )

        return binary_cross_entropy(
            predictions,
            y,
        )

    numerical_W1 = numerical_gradient(
        loss_with_current_parameters,
        W1,
    )

    numerical_b1 = numerical_gradient(
        loss_with_current_parameters,
        b1,
    )

    numerical_W2 = numerical_gradient(
        loss_with_current_parameters,
        W2,
    )

    numerical_b2 = numerical_gradient(
        loss_with_current_parameters,
        b2,
    )

    analytical_W1 = (
        analytical_gradients[0]
    )

    analytical_b1 = (
        analytical_gradients[1]
    )

    analytical_W2 = (
        analytical_gradients[2]
    )

    analytical_b2 = (
        analytical_gradients[3]
    )

    differences = [
        np.max(
            np.abs(
                analytical_W1
                - numerical_W1
            )
        ),
        np.max(
            np.abs(
                analytical_b1
                - numerical_b1
            )
        ),
        np.max(
            np.abs(
                analytical_W2
                - numerical_W2
            )
        ),
        np.max(
            np.abs(
                analytical_b2
                - numerical_b2
            )
        ),
    ]

    names = [
        "W1",
        "b1",
        "W2",
        "b2",
    ]

    for name, difference in zip(
        names,
        differences,
    ):
        print(
            f"{name}: max absolute difference = "
            f"{difference:.10e}"
        )

    for difference in differences:

        assert difference < 1e-6, (
            "Gradient check failed."
        )

    print(
        "\nPASS: analytical backpropagation "
        "matches numerical gradients."
    )


# =============================================================================
# 13. TRAIN NETWORK
# =============================================================================

def train_network(
    X,
    y,
    hidden_size=16,
    learning_rate=0.1,
    epochs=3000,
):
    """
    Train the neural network from random initialization.

    Nothing is pretrained.

    The network learns entirely from the supplied dataset.
    """

    W1, b1, W2, b2 = (
        initialize_parameters(
            input_size=X.shape[1],
            hidden_size=hidden_size,
            output_size=1,
        )
    )

    loss_history = []
    accuracy_history = []

    for epoch in range(
        epochs
    ):

        # ---------------------------------------------------------------------
        # Forward propagation
        # ---------------------------------------------------------------------

        y_hat, cache = forward(
            X,
            W1,
            b1,
            W2,
            b2,
        )

        # ---------------------------------------------------------------------
        # Loss
        # ---------------------------------------------------------------------

        loss = binary_cross_entropy(
            y_hat,
            y,
        )

        # ---------------------------------------------------------------------
        # Backpropagation
        # ---------------------------------------------------------------------

        gradients = backward(
            y,
            cache,
            W2,
        )

        dW1, db1, dW2, db2 = (
            gradients
        )

        # ---------------------------------------------------------------------
        # Parameter update
        # ---------------------------------------------------------------------

        (
            W1,
            b1,
            W2,
            b2,
        ) = update_parameters(
            W1,
            b1,
            W2,
            b2,
            dW1,
            db1,
            dW2,
            db2,
            learning_rate,
        )

        # ---------------------------------------------------------------------
        # Training metrics
        # ---------------------------------------------------------------------

        probabilities, predictions = (
            predict(
                X,
                W1,
                b1,
                W2,
                b2,
            )
        )

        current_accuracy = accuracy(
            y,
            predictions,
        )

        loss_history.append(
            loss
        )

        accuracy_history.append(
            current_accuracy
        )

        if (
            epoch == 0
            or (epoch + 1) % 300 == 0
        ):
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Loss {loss:.6f} | "
                f"Accuracy {current_accuracy:.4f}"
            )

    return (
        W1,
        b1,
        W2,
        b2,
        np.array(loss_history),
        np.array(accuracy_history),
    )


# =============================================================================
# 14. VISUALIZE DATASET
# =============================================================================

def visualize_dataset(
    X,
    y,
):
    """
    Visualize the nonlinear classification problem.

    This makes it clear why a simple linear classifier is insufficient.
    """

    plt.figure(
        figsize=(8, 7)
    )

    class_0 = (
        y[:, 0] == 0
    )

    class_1 = (
        y[:, 0] == 1
    )

    plt.scatter(
        X[class_0, 0],
        X[class_0, 1],
        label="Class 0",
        alpha=0.7,
    )

    plt.scatter(
        X[class_1, 0],
        X[class_1, 1],
        label="Class 1",
        alpha=0.7,
    )

    plt.xlabel(
        "Visual feature 1"
    )

    plt.ylabel(
        "Visual feature 2"
    )

    plt.title(
        "Nonlinear Vision Classification Problem"
    )

    plt.axis(
        "equal"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# 15. VISUALIZE DECISION BOUNDARY
# =============================================================================

def visualize_decision_boundary(
    X,
    y,
    W1,
    b1,
    W2,
    b2,
):
    """
    Visualize the learned nonlinear decision boundary.

    A dense grid is passed through the neural network.

    The resulting probability surface shows how the model separates the two
    classes.
    """

    x_min = X[:, 0].min() - 0.2
    x_max = X[:, 0].max() + 0.2

    y_min = X[:, 1].min() - 0.2
    y_max = X[:, 1].max() + 0.2

    grid_x = np.linspace(
        x_min,
        x_max,
        300,
    )

    grid_y = np.linspace(
        y_min,
        y_max,
        300,
    )

    mesh_x, mesh_y = np.meshgrid(
        grid_x,
        grid_y,
    )

    grid = np.column_stack(
        [
            mesh_x.ravel(),
            mesh_y.ravel(),
        ]
    )

    probabilities, _ = predict(
        grid,
        W1,
        b1,
        W2,
        b2,
    )

    probability_map = (
        probabilities
        .reshape(
            mesh_x.shape
        )
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.contourf(
        mesh_x,
        mesh_y,
        probability_map,
        levels=30,
        alpha=0.5,
    )

    class_0 = (
        y[:, 0] == 0
    )

    class_1 = (
        y[:, 0] == 1
    )

    plt.scatter(
        X[class_0, 0],
        X[class_0, 1],
        label="Class 0",
        edgecolors="black",
    )

    plt.scatter(
        X[class_1, 0],
        X[class_1, 1],
        label="Class 1",
        edgecolors="black",
    )

    plt.contour(
        mesh_x,
        mesh_y,
        probability_map,
        levels=[0.5],
        linewidths=2,
    )

    plt.xlabel(
        "Visual feature 1"
    )

    plt.ylabel(
        "Visual feature 2"
    )

    plt.title(
        "Learned Nonlinear Decision Boundary"
    )

    plt.axis(
        "equal"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# 16. VISUALIZE TRAINING
# =============================================================================

def visualize_training(
    loss_history,
    accuracy_history,
):
    """
    Show the two most important training curves:

        loss
        accuracy
    """

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        loss_history,
        linewidth=2,
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Binary Cross-Entropy"
    )

    plt.title(
        "Neural Network Training Loss"
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        accuracy_history,
        linewidth=2,
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Neural Network Training Accuracy"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 17. FINAL TESTS
# =============================================================================

def run_tests(
    X_train,
    y_train,
    X_test,
    y_test,
    W1,
    b1,
    W2,
    b2,
    loss_history,
):
    """
    Final verification suite.

    We verify:

        tensor dimensions
        finite parameters
        loss reduction
        predictions
        accuracy
        test-set execution
    """

    print("\n" + "=" * 80)
    print("FINAL TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Shape tests
    # -------------------------------------------------------------------------

    assert X_train.ndim == 2
    assert y_train.ndim == 2

    assert X_train.shape[1] == 2
    assert y_train.shape[1] == 1

    print(
        "PASS: training tensor shapes"
    )

    # -------------------------------------------------------------------------
    # Parameter shape tests
    # -------------------------------------------------------------------------

    assert W1.shape == (
        2,
        16,
    )

    assert b1.shape == (
        1,
        16,
    )

    assert W2.shape == (
        16,
        1,
    )

    assert b2.shape == (
        1,
        1,
    )

    print(
        "PASS: network parameter shapes"
    )

    # -------------------------------------------------------------------------
    # Finite parameter test
    # -------------------------------------------------------------------------

    assert np.all(
        np.isfinite(W1)
    )

    assert np.all(
        np.isfinite(b1)
    )

    assert np.all(
        np.isfinite(W2)
    )

    assert np.all(
        np.isfinite(b2)
    )

    print(
        "PASS: all parameters finite"
    )

    # -------------------------------------------------------------------------
    # Loss test
    # -------------------------------------------------------------------------

    assert np.all(
        np.isfinite(loss_history)
    )

    assert (
        loss_history[-1]
        < loss_history[0]
    )

    print(
        "PASS: training reduced loss"
    )

    # -------------------------------------------------------------------------
    # Training prediction test
    # -------------------------------------------------------------------------

    train_probabilities, train_predictions = (
        predict(
            X_train,
            W1,
            b1,
            W2,
            b2,
        )
    )

    assert train_probabilities.shape == (
        len(X_train),
        1,
    )

    assert train_predictions.shape == (
        len(X_train),
        1,
    )

    print(
        "PASS: training predictions"
    )

    # -------------------------------------------------------------------------
    # Probability range
    # -------------------------------------------------------------------------

    assert np.all(
        train_probabilities >= 0.0
    )

    assert np.all(
        train_probabilities <= 1.0
    )

    print(
        "PASS: probability range"
    )

    # -------------------------------------------------------------------------
    # Test set execution
    # -------------------------------------------------------------------------

    test_probabilities, test_predictions = (
        predict(
            X_test,
            W1,
            b1,
            W2,
            b2,
        )
    )

    test_accuracy = accuracy(
        y_test,
        test_predictions,
    )

    assert test_probabilities.shape == (
        len(X_test),
        1,
    )

    print(
        f"PASS: test-set prediction | "
        f"accuracy={test_accuracy:.4f}"
    )

    print("\n" + "-" * 80)
    print(
        "ALL LESSON 4 TESTS PASSED"
    )
    print("-" * 80)

    return test_accuracy


# =============================================================================
# 18. REAL-WORLD APPLICATION
# =============================================================================

def print_real_world_application():
    """
    Connect the lesson to an actual computer-vision pipeline.
    """

    print("\n" + "=" * 80)
    print("REAL-WORLD APPLICATION")
    print("=" * 80)

    print(
        """
Imagine a robot camera inspecting objects on a conveyor.

The real pipeline might eventually look like:

    Camera
       |
       v
    RGB image
       |
       v
    Tensor
       |
       v
    CNN
       |
       v
    Visual features
       |
       v
    Neural network
       |
       v
    Object classification
       |
       v
    Robot action

Today's network used only:

    2 numerical features

instead of an image.

But the learning mechanism is already the same:

    input
      |
      v
    weights
      |
      v
    activation
      |
      v
    prediction
      |
      v
    loss
      |
      v
    backpropagation
      |
      v
    gradients
      |
      v
    parameter update

A CNN will later replace the simple input representation with learned
spatial features extracted directly from images.

This is the bridge from:

    mathematical neural network

to:

    computer vision neural network.
"""
    )


# =============================================================================
# 19. LESSON RESULT
# =============================================================================

def print_lesson_result(
    train_accuracy,
    test_accuracy,
):
    """
    State the concrete result achieved by the lesson.
    """

    print("\n" + "=" * 80)
    print("LESSON 4 RESULT")
    print("=" * 80)

    print(
        f"""
The neural network started from random parameters.

It then learned a nonlinear classification function using:

    forward propagation
    binary cross-entropy
    backpropagation
    gradient descent

Final training accuracy:

    {train_accuracy:.4f}

Final test accuracy:

    {test_accuracy:.4f}

The important result is that the network learned a nonlinear decision
boundary without pretrained weights.

This is the first complete learning system in the course:

    DATA
      |
      v
    MODEL
      |
      v
    PREDICTION
      |
      v
    LOSS
      |
      v
    BACKPROPAGATION
      |
      v
    GRADIENT DESCENT
      |
      v
    LEARNED MODEL
"""
    )


# =============================================================================
# 20. NEXT LESSON
# =============================================================================

def print_next_lesson():
    """
    Explain why convolution is the next major step.
    """

    print("\n" + "=" * 80)
    print("CONNECTION TO LESSON 5")
    print("=" * 80)

    print(
        """
Our neural network now works with numerical features.

But images contain spatial structure.

For example:

    pixel near pixel
        |
        v
    edges
        |
        v
    textures
        |
        v
    shapes
        |
        v
    objects

A fully-connected neural network does not naturally exploit this spatial
structure.

We therefore need:

    CONVOLUTION

Lesson 5 will introduce the mathematical idea behind convolution.

We will build a convolution operation from scratch using NumPy.

We will see how a small kernel can detect:

    edges
    horizontal patterns
    vertical patterns
    simple visual structures

The progression will become:

    Lesson 1
        image

    Lesson 2
        tensor

    Lesson 3
        gradient descent

    Lesson 4
        neural network

    Lesson 5
        convolution

    Lesson 6+
        CNN

    Later
        deep visual representation

    Later
        VAE / GAN / Transformer

    Later
        diffusion model

    Final goal
        your own generative vision system
"""
    )


# =============================================================================
# 21. MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("COMPUTER VISION ADVANCED CRASH COURSE")
    print("LESSON 4 — NEURAL NETWORK FROM SCRATCH")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: activation sanity checks
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ACTIVATION FUNCTION CHECK")
    print("=" * 80)

    test_values = np.array(
        [
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
        ]
    )

    print(
        "Input:",
        test_values,
    )

    print(
        "ReLU:",
        relu(test_values),
    )

    print(
        "Sigmoid:",
        sigmoid(test_values),
    )

    # -------------------------------------------------------------------------
    # Step 2: gradient verification
    # -------------------------------------------------------------------------

    gradient_check()

    # -------------------------------------------------------------------------
    # Step 3: create data
    # -------------------------------------------------------------------------

    X, y = create_dataset()

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = train_test_split(
        X,
        y,
    )

    print("\n" + "=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        "Total samples:",
        len(X),
    )

    print(
        "Training samples:",
        len(X_train),
    )

    print(
        "Test samples:",
        len(X_test),
    )

    print(
        "Input shape:",
        X.shape,
    )

    print(
        "Target shape:",
        y.shape,
    )

    # -------------------------------------------------------------------------
    # Step 4: visualize dataset
    # -------------------------------------------------------------------------

    visualize_dataset(
        X_train,
        y_train,
    )

    # -------------------------------------------------------------------------
    # Step 5: train
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING NEURAL NETWORK")
    print("=" * 80)

    (
        W1,
        b1,
        W2,
        b2,
        loss_history,
        accuracy_history,
    ) = train_network(
        X_train,
        y_train,
        hidden_size=16,
        learning_rate=0.1,
        epochs=3000,
    )

    # -------------------------------------------------------------------------
    # Step 6: evaluate
    # -------------------------------------------------------------------------

    _, train_predictions = predict(
        X_train,
        W1,
        b1,
        W2,
        b2,
    )

    _, test_predictions = predict(
        X_test,
        W1,
        b1,
        W2,
        b2,
    )

    train_accuracy = accuracy(
        y_train,
        train_predictions,
    )

    test_accuracy = accuracy(
        y_test,
        test_predictions,
    )

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    print(
        f"Training accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Test accuracy:     {test_accuracy:.4f}"
    )

    # -------------------------------------------------------------------------
    # Step 7: final tests
    # -------------------------------------------------------------------------

    verified_test_accuracy = run_tests(
        X_train,
        y_train,
        X_test,
        y_test,
        W1,
        b1,
        W2,
        b2,
        loss_history,
    )

    # -------------------------------------------------------------------------
    # Step 8: visualize training
    # -------------------------------------------------------------------------

    visualize_training(
        loss_history,
        accuracy_history,
    )

    # -------------------------------------------------------------------------
    # Step 9: visualize decision boundary
    # -------------------------------------------------------------------------

    visualize_decision_boundary(
        X_train,
        y_train,
        W1,
        b1,
        W2,
        b2,
    )

    # -------------------------------------------------------------------------
    # Step 10: real-world application
    # -------------------------------------------------------------------------

    print_real_world_application()

    # -------------------------------------------------------------------------
    # Step 11: lesson result
    # -------------------------------------------------------------------------

    print_lesson_result(
        train_accuracy,
        verified_test_accuracy,
    )

    # -------------------------------------------------------------------------
    # Step 12: next lesson
    # -------------------------------------------------------------------------

    print_next_lesson()

    print("\nLesson 4 completed successfully.")
