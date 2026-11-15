
"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 3 — GRADIENT DESCENT FROM SCRATCH
================================================================================

PURPOSE
-------
In Lesson 2 we learned how images, vectors, matrices, and tensors are
represented numerically.

We also introduced the neural-network equation:

    y = Wx + b

But knowing how to calculate an output is not enough.

A neural network must LEARN its parameters.

This lesson introduces the fundamental mechanism behind that learning process:

    GRADIENT DESCENT

We will build the complete idea from scratch.

No PyTorch.
No TensorFlow.
No pretrained model.
No automatic differentiation.

Only:

    Python
    NumPy
    Matplotlib

The goal is to understand what a training algorithm is actually doing.

-------------------------------------------------------------------------------
WHAT YOU WILL LEARN
-------------------------------------------------------------------------------

1. What a parameter is
2. What a loss function is
3. Why minimizing loss produces learning
4. Derivatives
5. Numerical derivatives
6. Analytical derivatives
7. Gradients
8. Gradient descent
9. Learning rate
10. Parameter updates
11. Linear regression from scratch
12. Training a model using gradient descent
13. Loss curves
14. Numerical verification
15. Real-world computer-vision interpretation

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

Current position:

    Lesson 1
        Images -> numerical data

    Lesson 2
        Vectors -> matrices -> tensors

    Lesson 3
        Parameters -> loss -> gradients -> learning

    Future:
        gradients -> neural networks -> CNNs -> generative models

-------------------------------------------------------------------------------
REAL-WORLD PROBLEM
-------------------------------------------------------------------------------

Imagine a camera looking at objects on a table.

Suppose we want a simple model that predicts:

    object size

from:

    image-derived feature

For example:

    x = image feature
    y = measured object size

We might initially choose:

    y_pred = wx + b

But we do not know:

    w
    b

We need to learn them from examples.

Given:

    x = observed feature
    y = known measurement

the model predicts:

    y_pred = wx + b

Then we calculate the error:

    loss = difference between prediction and truth

Then we adjust:

    w
    b

to reduce the loss.

This is the basic structure of machine learning.

-------------------------------------------------------------------------------
THEORY: PARAMETERS
-------------------------------------------------------------------------------

A parameter is a numerical value that the model learns.

For our simple model:

    y_pred = wx + b

the parameters are:

    w
    b

Initially, we can choose arbitrary values.

For example:

    w = 0
    b = 0

The model will probably make bad predictions.

Training means finding parameter values that produce smaller errors.

-------------------------------------------------------------------------------
THEORY: LOSS
-------------------------------------------------------------------------------

A loss function measures how wrong a prediction is.

For one sample:

    error = y_pred - y

A common squared error is:

    L = (y_pred - y)^2

For multiple samples, we often use mean squared error:

                 1
    MSE = ---------------- sum (y_pred - y)^2
                 N

where:

    N = number of training samples

The goal of training is:

    minimize MSE

-------------------------------------------------------------------------------
WHY SQUARED ERROR?
-------------------------------------------------------------------------------

Suppose:

    error = +2

or:

    error = -2

Both should represent an error magnitude of 2.

If we simply used:

    error

positive and negative errors could cancel.

Squaring gives:

    (+2)^2 = 4
    (-2)^2 = 4

Therefore the loss is always non-negative.

-------------------------------------------------------------------------------
DERIVATIVE
-------------------------------------------------------------------------------

The derivative tells us how a function changes when its input changes.

Consider:

    f(x) = x^2

The analytical derivative is:

    df/dx = 2x

At:

    x = 3

the derivative is:

    6

This tells us that increasing x slightly will increase f(x).

The derivative therefore gives us a DIRECTION OF CHANGE.

-------------------------------------------------------------------------------
NUMERICAL DERIVATIVE
-------------------------------------------------------------------------------

We can estimate a derivative without knowing its analytical formula.

The central-difference approximation is:

                         f(x+h) - f(x-h)
    f'(x) ≈             -----------------
                                2h

where h is a very small number.

This is extremely useful for verifying analytical derivatives.

-------------------------------------------------------------------------------
GRADIENT
-------------------------------------------------------------------------------

For multiple parameters, we have multiple derivatives.

For example:

    L(w,b)

The gradient is:

             [ dL/dw ]
    grad L = [ dL/db ]

The gradient points in the direction of greatest increase of the loss.

Therefore, to reduce the loss, we move in the opposite direction:

    parameter_new =
        parameter_old - learning_rate * gradient

This is gradient descent.

-------------------------------------------------------------------------------
GRADIENT DESCENT
-------------------------------------------------------------------------------

The fundamental update rule is:

    theta_new = theta_old - alpha * grad(L)

where:

    theta = model parameters
    alpha = learning rate
    grad(L) = gradient of the loss

Example:

    theta = 5
    gradient = 2
    learning_rate = 0.1

Then:

    theta_new = 5 - 0.1*2
              = 4.8

The parameter moves toward lower loss.

-------------------------------------------------------------------------------
LEARNING RATE
-------------------------------------------------------------------------------

The learning rate controls how large each update is.

Too small:

    training is extremely slow

Too large:

    training may overshoot
    training may oscillate
    training may diverge

A useful mental model:

    gradient = direction
    learning rate = step size

-------------------------------------------------------------------------------
ANALYTICAL GRADIENT FOR LINEAR REGRESSION
-------------------------------------------------------------------------------

Our model is:

    y_hat = wx + b

For one sample:

    error = y_hat - y

Loss:

    L = (y_hat - y)^2

Because:

    y_hat = wx + b

we have:

    L = (wx + b - y)^2

Differentiate with respect to w:

    dL/dw = 2(wx + b - y)x

Differentiate with respect to b:

    dL/db = 2(wx + b - y)

For N samples, the mean squared error is:

             1
    MSE =    - sum(error^2)
             N

Therefore:

             2
    dMSE/dw = - sum(error*x)
             N

and:

             2
    dMSE/db = - sum(error)
             N

These are the gradients we will implement.

-------------------------------------------------------------------------------
IMPLEMENTATION STRATEGY
-------------------------------------------------------------------------------

We will perform the following pipeline:

    Generate synthetic data
            |
            v
    Initialize w and b
            |
            v
    Predict
            |
            v
    Calculate MSE
            |
            v
    Calculate gradients
            |
            v
    Update w and b
            |
            v
    Repeat
            |
            v
    Learned model

This is the core training loop.

-------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. BASIC FUNCTION
# =============================================================================

def square_function(x):
    """
    Simple function:

        f(x) = x^2

    This is used to introduce derivatives before applying them to machine
    learning.
    """

    return x ** 2


# =============================================================================
# 2. ANALYTICAL DERIVATIVE
# =============================================================================

def square_derivative(x):
    """
    Analytical derivative of:

        f(x) = x^2

    is:

        f'(x) = 2x
    """

    return 2.0 * x


# =============================================================================
# 3. NUMERICAL DERIVATIVE
# =============================================================================

def numerical_derivative(function, x, h=1e-5):
    """
    Estimate the derivative using the central-difference approximation:

                f(x+h) - f(x-h)
        f'(x) ≈ -----------------
                     2h

    This method is intentionally implemented manually.

    Later, when we use automatic differentiation frameworks, understanding
    this calculation will make the automatic system much less mysterious.
    """

    return (
        function(x + h) - function(x - h)
    ) / (2.0 * h)


# =============================================================================
# 4. VERIFY DERIVATIVE
# =============================================================================

def verify_derivative():
    """
    Compare numerical and analytical derivatives.

    This is an important engineering habit:

        analytical result
                vs
        numerical approximation

    If they disagree significantly, something is probably wrong.
    """

    print("\n" + "=" * 80)
    print("1. DERIVATIVE VERIFICATION")
    print("=" * 80)

    test_points = [
        -3.0,
        -1.0,
        0.0,
        1.0,
        3.0,
    ]

    for x in test_points:

        analytical = square_derivative(x)

        numerical = numerical_derivative(
            square_function,
            x,
        )

        print(
            f"x={x:6.2f} | "
            f"analytical={analytical:10.6f} | "
            f"numerical={numerical:10.6f}"
        )


# =============================================================================
# 5. SYNTHETIC REAL-WORLD DATA
# =============================================================================

def create_training_data():
    """
    Create a synthetic dataset representing a simplified computer-vision
    measurement problem.

    Imagine a vision system extracts one numerical feature from an image.

        x = visual feature

    The target is:

        y = physical object measurement

    We construct:

        y = 3x + 2

    and add small noise.

    In a real system, the relationship would not necessarily be this simple.

    Here the purpose is to isolate the learning mechanism.
    """

    rng = np.random.default_rng(42)

    x = np.linspace(
        0.0,
        10.0,
        50,
    )

    true_w = 3.0
    true_b = 2.0

    noise = rng.normal(
        loc=0.0,
        scale=0.5,
        size=x.shape,
    )

    y = (
        true_w * x
        + true_b
        + noise
    )

    return x, y, true_w, true_b


# =============================================================================
# 6. MODEL
# =============================================================================

def predict(x, w, b):
    """
    Linear model:

        y_hat = wx + b

    This is the simplest possible learnable model.

    Later:

        linear layer
            ->
        nonlinear neural network
            ->
        CNN
            ->
        U-Net
            ->
        diffusion model
    """

    return w * x + b


# =============================================================================
# 7. MEAN SQUARED ERROR
# =============================================================================

def mean_squared_error(y_pred, y_true):
    """
    Mean Squared Error:

                 1
        MSE =    - sum((y_pred-y_true)^2)
                 N
    """

    error = y_pred - y_true

    return np.mean(
        error ** 2
    )


# =============================================================================
# 8. ANALYTICAL GRADIENT
# =============================================================================

def linear_regression_gradients(
    x,
    y,
    w,
    b,
):
    """
    Calculate analytical gradients for:

        y_hat = wx + b

    Loss:

        MSE = mean((y_hat-y)^2)

    Gradient with respect to w:

             2
        dL/dw = mean(error*x)

    Gradient with respect to b:

             2
        dL/db = mean(error)

    More precisely:

        dL/dw = (2/N) * sum(error*x)

        dL/db = (2/N) * sum(error)

    where:

        error = y_hat-y
    """

    y_pred = predict(
        x,
        w,
        b,
    )

    error = y_pred - y

    grad_w = (
        2.0
        * np.mean(
            error * x
        )
    )

    grad_b = (
        2.0
        * np.mean(error)
    )

    return grad_w, grad_b


# =============================================================================
# 9. NUMERICAL GRADIENT FOR W
# =============================================================================

def numerical_gradient_w(
    x,
    y,
    w,
    b,
    h=1e-5,
):
    """
    Numerically estimate:

        dL/dw

    using:

        [L(w+h,b) - L(w-h,b)]
        ---------------------
                 2h
    """

    loss_plus = mean_squared_error(
        predict(x, w + h, b),
        y,
    )

    loss_minus = mean_squared_error(
        predict(x, w - h, b),
        y,
    )

    return (
        loss_plus - loss_minus
    ) / (2.0 * h)


# =============================================================================
# 10. NUMERICAL GRADIENT FOR B
# =============================================================================

def numerical_gradient_b(
    x,
    y,
    w,
    b,
    h=1e-5,
):
    """
    Numerically estimate:

        dL/db

    using:

        [L(w,b+h) - L(w,b-h)]
        ---------------------
                 2h
    """

    loss_plus = mean_squared_error(
        predict(x, w, b + h),
        y,
    )

    loss_minus = mean_squared_error(
        predict(x, w, b - h),
        y,
    )

    return (
        loss_plus - loss_minus
    ) / (2.0 * h)


# =============================================================================
# 11. VERIFY LINEAR REGRESSION GRADIENTS
# =============================================================================

def verify_linear_gradients(
    x,
    y,
):
    """
    Compare analytical and numerical gradients.

    This is one of the most important tests in the lesson.

    Before trusting a training algorithm, verify that its gradients are
    mathematically correct.
    """

    print("\n" + "=" * 80)
    print("2. LINEAR REGRESSION GRADIENT VERIFICATION")
    print("=" * 80)

    w = 0.7
    b = -1.3

    analytical_w, analytical_b = (
        linear_regression_gradients(
            x,
            y,
            w,
            b,
        )
    )

    numerical_w = numerical_gradient_w(
        x,
        y,
        w,
        b,
    )

    numerical_b = numerical_gradient_b(
        x,
        y,
        w,
        b,
    )

    print(
        f"Analytical dL/dw = {analytical_w:.10f}"
    )

    print(
        f"Numerical   dL/dw = {numerical_w:.10f}"
    )

    print()

    print(
        f"Analytical dL/db = {analytical_b:.10f}"
    )

    print(
        f"Numerical   dL/db = {numerical_b:.10f}"
    )

    assert np.allclose(
        analytical_w,
        numerical_w,
        rtol=1e-5,
        atol=1e-7,
    ), (
        "Gradient verification failed for w.\n"
        f"Analytical: {analytical_w}\n"
        f"Numerical:  {numerical_w}"
    )

    assert np.allclose(
        analytical_b,
        numerical_b,
        rtol=1e-5,
        atol=1e-7,
    ), (
        "Gradient verification failed for b.\n"
        f"Analytical: {analytical_b}\n"
        f"Numerical:  {numerical_b}"
    )

    print("\nPASS: analytical and numerical gradients agree.")


# =============================================================================
# 12. GRADIENT DESCENT TRAINING
# =============================================================================

def train_linear_model(
    x,
    y,
    learning_rate=0.01,
    epochs=1000,
):
    """
    Train:

        y_hat = wx + b

    using gradient descent.

    Training loop:

        1. predict
        2. calculate loss
        3. calculate gradient
        4. update parameters
        5. repeat

    Update equations:

        w <- w - alpha * dL/dw

        b <- b - alpha * dL/db

    where alpha is the learning rate.
    """

    # Initial parameters.
    w = 0.0
    b = 0.0

    loss_history = []

    for epoch in range(epochs):

        # -------------------------------------------------------------
        # Forward pass
        # -------------------------------------------------------------

        y_pred = predict(
            x,
            w,
            b,
        )

        # -------------------------------------------------------------
        # Calculate loss
        # -------------------------------------------------------------

        loss = mean_squared_error(
            y_pred,
            y,
        )

        loss_history.append(
            loss
        )

        # -------------------------------------------------------------
        # Calculate gradients
        # -------------------------------------------------------------

        grad_w, grad_b = (
            linear_regression_gradients(
                x,
                y,
                w,
                b,
            )
        )

        # -------------------------------------------------------------
        # Gradient descent update
        # -------------------------------------------------------------

        w = (
            w
            - learning_rate * grad_w
        )

        b = (
            b
            - learning_rate * grad_b
        )

        # -------------------------------------------------------------
        # Print progress
        # -------------------------------------------------------------

        if (
            epoch == 0
            or (epoch + 1) % 100 == 0
        ):
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Loss {loss:10.6f} | "
                f"w {w:8.4f} | "
                f"b {b:8.4f}"
            )

    return (
        w,
        b,
        np.array(loss_history),
    )


# =============================================================================
# 13. VERIFY THAT TRAINING REDUCES LOSS
# =============================================================================

def verify_training(
    loss_history,
):
    """
    Verify the most important property of gradient descent:

        final loss < initial loss

    The model should improve.
    """

    initial_loss = loss_history[0]
    final_loss = loss_history[-1]

    print("\n" + "=" * 80)
    print("3. TRAINING VERIFICATION")
    print("=" * 80)

    print(
        f"Initial loss: {initial_loss:.6f}"
    )

    print(
        f"Final loss:   {final_loss:.6f}"
    )

    assert final_loss < initial_loss, (
        "Training failed to reduce the loss."
    )

    print(
        "\nPASS: gradient descent reduced the loss."
    )


# =============================================================================
# 14. VISUALIZE DATA AND LEARNED MODEL
# =============================================================================

def visualize_model(
    x,
    y,
    w,
    b,
    true_w,
    true_b,
):
    """
    Show:

        1. Training data
        2. Learned line
        3. Ground-truth line

    This makes the result of optimization visually understandable.
    """

    x_plot = np.linspace(
        x.min(),
        x.max(),
        200,
    )

    learned_y = predict(
        x_plot,
        w,
        b,
    )

    true_y = predict(
        x_plot,
        true_w,
        true_b,
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        x,
        y,
        label="Training data",
    )

    plt.plot(
        x_plot,
        learned_y,
        linewidth=3,
        label="Learned model",
    )

    plt.plot(
        x_plot,
        true_y,
        linestyle="--",
        linewidth=2,
        label="Underlying relationship",
    )

    plt.xlabel(
        "Visual feature"
    )

    plt.ylabel(
        "Object measurement"
    )

    plt.title(
        "Learning a Vision-Related Measurement"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# 15. VISUALIZE LOSS
# =============================================================================

def visualize_loss(
    loss_history,
):
    """
    Visualize the optimization process.

    The x-axis is training time.

    The y-axis is loss.

    A successful training run should generally show decreasing loss.
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
        "Mean Squared Error"
    )

    plt.title(
        "Gradient Descent Loss Curve"
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 16. LEARNING-RATE EXPERIMENT
# =============================================================================

def learning_rate_experiment(
    x,
    y,
):
    """
    Demonstrate the effect of learning rate.

    We train three models:

        small learning rate
        reasonable learning rate
        large learning rate

    The experiment demonstrates why learning rate is a critical training
    hyperparameter.
    """

    print("\n" + "=" * 80)
    print("4. LEARNING-RATE EXPERIMENT")
    print("=" * 80)

    learning_rates = [
        0.001,
        0.01,
        0.1,
    ]

    histories = []

    for lr in learning_rates:

        w = 0.0
        b = 0.0

        history = []

        for _ in range(300):

            y_pred = predict(
                x,
                w,
                b,
            )

            loss = mean_squared_error(
                y_pred,
                y,
            )

            history.append(
                loss
            )

            grad_w, grad_b = (
                linear_regression_gradients(
                    x,
                    y,
                    w,
                    b,
                )
            )

            w -= lr * grad_w
            b -= lr * grad_b

        histories.append(
            (
                lr,
                np.array(history),
            )
        )

        print(
            f"Learning rate={lr:.3f} | "
            f"final loss={history[-1]:.6f}"
        )

    plt.figure(
        figsize=(9, 6)
    )

    for lr, history in histories:

        plt.plot(
            history,
            label=f"learning rate = {lr}",
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "MSE Loss"
    )

    plt.title(
        "Effect of Learning Rate"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# 17. REAL-WORLD APPLICATION
# =============================================================================

def print_real_world_application():
    """
    Explain how this simple example maps to a real computer-vision system.
    """

    print("\n" + "=" * 80)
    print("REAL-WORLD APPLICATION")
    print("=" * 80)

    print(
        """
Imagine a warehouse robot.

The robot has:

    RGB camera
        |
        v
    image
        |
        v
    visual feature extraction
        |
        v
    numerical feature x
        |
        v
    learned model
        |
        v
    predicted object measurement
        |
        v
    robot decision

For today's simplified example:

    x = one visual feature

    y = object measurement

    model:

        y_hat = wx + b

The model learns w and b from examples.

In a real vision system, x would not normally be just one manually selected
number.

Instead, a neural network might transform:

    image
      |
      v
    tensor
      |
      v
    convolution layers
      |
      v
    feature maps
      |
      v
    learned representation
      |
      v
    prediction

But the fundamental idea remains:

    PARAMETERS
        |
        v
    PREDICTION
        |
        v
    LOSS
        |
        v
    GRADIENT
        |
        v
    PARAMETER UPDATE
        |
        +------------------+
                           |
                           v
                       repeat

This is the core learning loop.
"""
    )


# =============================================================================
# 18. TEST ALL CORE COMPONENTS
# =============================================================================

def run_all_tests(
    x,
    y,
    loss_history,
):
    """
    Final integrated verification.

    The lesson should not be considered complete unless all important
    mathematical and training checks pass.
    """

    print("\n" + "=" * 80)
    print("FINAL TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test 1: numerical derivative
    # -------------------------------------------------------------------------

    x_test = 2.5

    numerical = numerical_derivative(
        square_function,
        x_test,
    )

    analytical = square_derivative(
        x_test
    )

    assert np.allclose(
        numerical,
        analytical,
        rtol=1e-5,
        atol=1e-7,
    ), (
        "Square-function derivative test failed."
    )

    print(
        "PASS: numerical derivative"
    )


    # -------------------------------------------------------------------------
    # Test 2: data dimensions
    # -------------------------------------------------------------------------

    assert x.ndim == 1
    assert y.ndim == 1
    assert x.shape == y.shape

    print(
        "PASS: training data dimensions"
    )


    # -------------------------------------------------------------------------
    # Test 3: loss history
    # -------------------------------------------------------------------------

    assert loss_history.ndim == 1
    assert len(loss_history) > 1

    print(
        "PASS: loss history"
    )


    # -------------------------------------------------------------------------
    # Test 4: loss reduction
    # -------------------------------------------------------------------------

    assert (
        loss_history[-1]
        < loss_history[0]
    )

    print(
        "PASS: loss reduction"
    )


    # -------------------------------------------------------------------------
    # Test 5: finite values
    # -------------------------------------------------------------------------

    assert np.all(
        np.isfinite(loss_history)
    )

    print(
        "PASS: all training values finite"
    )


    print("\n" + "-" * 80)
    print("ALL LESSON 3 TESTS PASSED")
    print("-" * 80)


# =============================================================================
# 19. LESSON SUMMARY
# =============================================================================

def print_lesson_summary(
    learned_w,
    learned_b,
    true_w,
    true_b,
):
    """
    Print the final learning result.
    """

    print("\n" + "=" * 80)
    print("LESSON 3 RESULT")
    print("=" * 80)

    print(
        f"""
Underlying relationship:

    y = {true_w:.2f}x + {true_b:.2f}

Learned relationship:

    y = {learned_w:.4f}x + {learned_b:.4f}

The model started with:

    w = 0
    b = 0

and learned parameters that approximate the data.

The important achievement is NOT simply learning a straight line.

The important achievement is understanding this loop:

    data
      |
      v
    prediction
      |
      v
    loss
      |
      v
    gradient
      |
      v
    parameter update
      |
      v
    improved model

This is the basic learning mechanism used throughout modern machine learning.
"""
    )


# =============================================================================
# 20. CONNECTION TO NEXT LESSON
# =============================================================================

def print_next_lesson():
    """
    Explain why Lesson 4 follows naturally.
    """

    print("\n" + "=" * 80)
    print("CONNECTION TO LESSON 4")
    print("=" * 80)

    print(
        """
Lesson 3 taught us how parameters can be learned.

We trained:

    y = wx + b

But real images are much more complicated.

A useful vision model needs:

    multiple parameters
        +
    nonlinear functions
        +
    multiple layers

For example:

    image
      |
      v
    linear layer
      |
      v
    activation
      |
      v
    linear layer
      |
      v
    prediction

This introduces the concept of a:

    NEURAL NETWORK

Lesson 4 will build a small neural network from scratch.

We will introduce:

    parameters
    layers
    activations
    forward propagation
    loss
    backpropagation
    gradients
    parameter updates

The important transition will be:

    Lesson 3
        one simple model

             |

             v

    Lesson 4
        multi-layer neural network

             |

             v

    Later
        CNN

             |

             v

        Vision Transformer

             |

             v

        U-Net

             |

             v

        Diffusion model

             |

             v

        Your own generative vision model
"""
    )


# =============================================================================
# 21. MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("COMPUTER VISION ADVANCED CRASH COURSE")
    print("LESSON 3 — GRADIENT DESCENT FROM SCRATCH")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1
    # -------------------------------------------------------------------------

    verify_derivative()

    # -------------------------------------------------------------------------
    # Step 2
    # -------------------------------------------------------------------------

    x_train, y_train, true_w, true_b = (
        create_training_data()
    )

    print("\n" + "=" * 80)
    print("TRAINING DATA")
    print("=" * 80)

    print(
        "Number of samples:",
        len(x_train),
    )

    print(
        "First 5 x values:",
        x_train[:5],
    )

    print(
        "First 5 y values:",
        y_train[:5],
    )

    # -------------------------------------------------------------------------
    # Step 3
    # -------------------------------------------------------------------------

    verify_linear_gradients(
        x_train,
        y_train,
    )

    # -------------------------------------------------------------------------
    # Step 4
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("GRADIENT DESCENT TRAINING")
    print("=" * 80)

    learned_w, learned_b, loss_history = (
        train_linear_model(
            x_train,
            y_train,
            learning_rate=0.01,
            epochs=1000,
        )
    )

    # -------------------------------------------------------------------------
    # Step 5
    # -------------------------------------------------------------------------

    verify_training(
        loss_history
    )

    # -------------------------------------------------------------------------
    # Step 6
    # -------------------------------------------------------------------------

    run_all_tests(
        x_train,
        y_train,
        loss_history,
    )

    # -------------------------------------------------------------------------
    # Step 7
    # -------------------------------------------------------------------------

    print_real_world_application()

    # -------------------------------------------------------------------------
    # Step 8
    # -------------------------------------------------------------------------

    learning_rate_experiment(
        x_train,
        y_train,
    )

    # -------------------------------------------------------------------------
    # Step 9
    # -------------------------------------------------------------------------

    visualize_model(
        x_train,
        y_train,
        learned_w,
        learned_b,
        true_w,
        true_b,
    )

    # -------------------------------------------------------------------------
    # Step 10
    # -------------------------------------------------------------------------

    visualize_loss(
        loss_history
    )

    # -------------------------------------------------------------------------
    # Step 11
    # -------------------------------------------------------------------------

    print_lesson_summary(
        learned_w,
        learned_b,
        true_w,
        true_b,
    )

    # -------------------------------------------------------------------------
    # Step 12
    # -------------------------------------------------------------------------

    print_next_lesson()

    print("\nLesson 3 completed successfully.")
