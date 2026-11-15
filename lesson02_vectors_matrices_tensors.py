
"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 2 — VECTORS, MATRICES, AND TENSORS
================================================================================

PURPOSE
-------
This lesson builds the mathematical representation used by almost every modern
computer-vision and generative-AI system.

A neural network does not directly "see" an image.

It receives numerical tensors.

Before learning CNNs, Transformers, VAEs, GANs, or Diffusion Models, we need
to understand:

    vector
    matrix
    tensor
    shape
    element-wise operations
    dot product
    matrix-vector multiplication
    matrix-matrix multiplication
    broadcasting
    coordinate transformations
    linear layers

The goal is to make these operations intuitive rather than treating them as
black-box NumPy operations.

-------------------------------------------------------------------------------
WHAT YOU WILL LEARN
-------------------------------------------------------------------------------

1. What a vector is
2. What a matrix is
3. What a tensor is
4. Tensor shapes
5. Element-wise operations
6. Dot products
7. Matrix-vector multiplication
8. Matrix-matrix multiplication
9. Broadcasting
10. RGB image tensors
11. Batch tensors
12. 2-D geometric transformations
13. Rotation
14. Scaling
15. Combined transformations
16. The mathematical structure of a neural-network linear layer
17. Numerical verification
18. Visualization
19. Connection to robotics and computer vision

-------------------------------------------------------------------------------
TECHNICAL ROADMAP
-------------------------------------------------------------------------------

The overall course is:

Phase 0  -> Mathematics
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
Phase 13 -> Scaling and production training

This lesson is the bridge between:

    IMAGE
      |
      v
    TENSOR
      |
      v
    MATHEMATICAL OPERATIONS
      |
      v
    NEURAL NETWORK
      |
      v
    LEARNED VISION SYSTEM

-------------------------------------------------------------------------------
REAL-WORLD PROBLEM
-------------------------------------------------------------------------------

Imagine a robot looking at a table using an RGB camera.

The camera produces:

    image = H x W x 3

For example:

    480 x 640 x 3

Every pixel contains three values:

    R
    G
    B

A robot-learning system may then process this image as:

    Camera
       |
       v
    RGB tensor
       |
       v
    Tensor transformations
       |
       v
    Neural network
       |
       v
    Object representation
       |
       v
    Robot decision

The mathematical operations in this lesson are the foundations of that entire
pipeline.

-------------------------------------------------------------------------------
THEORY
-------------------------------------------------------------------------------

VECTOR
------

A vector is an ordered collection of numbers.

Example:

    x = [1, 2, 3]

In Python:

    x = np.array([1, 2, 3])

A vector can represent:

    position
    velocity
    color
    feature values
    robot joint states
    neural-network activations

For example, a 2-D image coordinate:

    p = [x, y]

A robot joint state:

    q = [q1, q2, q3, q4, q5, q6]

-------------------------------------------------------------------------------
MATRIX
------

A matrix is a rectangular grid of numbers.

Example:

        [1 2 3]
    A = [4 5 6]

Its shape is:

    (2, 3)

meaning:

    2 rows
    3 columns

Matrices are extremely important because they can transform vectors.

-------------------------------------------------------------------------------
TENSOR
------

A tensor generalizes vectors and matrices to multiple dimensions.

Examples:

    scalar:
        shape = ()

    vector:
        shape = (3,)

    matrix:
        shape = (3, 4)

    RGB image:
        shape = (H, W, 3)

    batch of RGB images:
        shape = (N, H, W, 3)

For example:

    8 images
    64 x 64 pixels
    3 color channels

has shape:

    (8, 64, 64, 3)

Modern neural networks are fundamentally tensor-processing systems.

-------------------------------------------------------------------------------
MATHEMATICS
-------------------------------------------------------------------------------

VECTOR ADDITION
---------------

For:

    a = [a1, a2, a3]
    b = [b1, b2, b3]

we have:

    a + b =
    [a1+b1, a2+b2, a3+b3]

-------------------------------------------------------------------------------
ELEMENT-WISE MULTIPLICATION
-------------------------------------------------------------------------------

    a * b =
    [a1*b1, a2*b2, a3*b3]

This is different from matrix multiplication.

-------------------------------------------------------------------------------
DOT PRODUCT
-----------

For:

    a = [a1, a2, a3]
    b = [b1, b2, b3]

the dot product is:

    a · b = a1*b1 + a2*b2 + a3*b3

The result is a scalar.

The dot product appears everywhere in machine learning.

For example:

    neuron output = w · x + b

-------------------------------------------------------------------------------
MATRIX-VECTOR MULTIPLICATION
----------------------------

Suppose:

        [w11 w12 w13]
    W = [w21 w22 w23]

and:

        [x1]
    x = [x2]
        [x3]

Then:

        [w11*x1 + w12*x2 + w13*x3]
    Wx =
        [w21*x1 + w22*x2 + w23*x3]

This is the basic operation of a linear neural-network layer.

-------------------------------------------------------------------------------
MATRIX-MATRIX MULTIPLICATION
----------------------------

If:

    A has shape (m, n)
    B has shape (n, p)

then:

    A @ B

has shape:

    (m, p)

The inner dimensions must match.

Example:

    (2,3) @ (3,2) -> (2,2)

-------------------------------------------------------------------------------
BROADCASTING
------------

NumPy allows operations between arrays with compatible shapes.

For example:

    [1,2,3] + 10

becomes:

    [11,12,13]

Broadcasting is heavily used in neural-network implementations.

For example:

    y = x + bias

where a bias vector is applied to many neurons or many pixels.

-------------------------------------------------------------------------------
IMAGE TENSOR
------------

An RGB image can be represented as:

    image.shape = (H, W, 3)

For example:

    (64, 64, 3)

The last dimension represents:

    channel 0 = Red
    channel 1 = Green
    channel 2 = Blue

-------------------------------------------------------------------------------
BATCH TENSOR
------------

A batch of images can be represented as:

    (N, H, W, C)

where:

    N = number of images
    H = height
    W = width
    C = channels

Example:

    (8, 64, 64, 3)

means:

    8 images
    64 pixels high
    64 pixels wide
    3 channels

This becomes important when training neural networks.

-------------------------------------------------------------------------------
2-D COORDINATE TRANSFORMATION
-----------------------------

Suppose a camera detects an object at:

    p = [x, y]

A scaling transformation is:

        [sx  0]
    S = [0  sy]

Then:

    p' = S p

A rotation transformation is:

             [cos(theta)  -sin(theta)]
    R(theta)=[sin(theta)   cos(theta)]

Then:

    p' = R p

Multiple transformations can be combined using matrix multiplication.

For example:

    T = R @ S

and:

    p' = T @ p

-------------------------------------------------------------------------------
NEURAL NETWORK CONNECTION
-------------------------

A simple neural-network layer is:

    y = Wx + b

where:

    x = input vector
    W = learned weight matrix
    b = learned bias vector
    y = output vector

During training, the neural network learns W and b.

This extremely simple equation is the mathematical ancestor of much more
complicated vision models.

CNNs, Transformers, U-Nets, VAEs, and diffusion networks all ultimately
perform large numbers of tensor operations.

-------------------------------------------------------------------------------
IMPLEMENTATION
-------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. VECTOR OPERATIONS
# =============================================================================

print("\n" + "=" * 80)
print("1. VECTOR OPERATIONS")
print("=" * 80)

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

vector_sum = a + b
elementwise_product = a * b
dot_product = np.dot(a, b)

print("a =", a)
print("b =", b)
print("a + b =", vector_sum)
print("a * b =", elementwise_product)
print("a · b =", dot_product)


# =============================================================================
# 2. MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("2. MATRIX")
print("=" * 80)

A = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

print("A =")
print(A)

print("A.shape =", A.shape)


# =============================================================================
# 3. MATRIX-VECTOR MULTIPLICATION
# =============================================================================

print("\n" + "=" * 80)
print("3. MATRIX-VECTOR MULTIPLICATION")
print("=" * 80)

W = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

x = np.array([10.0, 20.0, 30.0])

matrix_vector_result = W @ x

print("W =")
print(W)

print("x =", x)

print("W @ x =", matrix_vector_result)

print(
    "\nManual calculation:"
    "\nfirst output  = 1*10 + 2*20 + 3*30 = 140"
    "\nsecond output = 4*10 + 5*20 + 6*30 = 320"
)


# =============================================================================
# 4. MATRIX-MATRIX MULTIPLICATION
# =============================================================================

print("\n" + "=" * 80)
print("4. MATRIX-MATRIX MULTIPLICATION")
print("=" * 80)

B = np.array(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
)

matrix_matrix_result = A @ B

print("A =")
print(A)

print("\nB =")
print(B)

print("\nA @ B =")
print(matrix_matrix_result)

print("\nShape:")
print("A.shape =", A.shape)
print("B.shape =", B.shape)
print("(A @ B).shape =", matrix_matrix_result.shape)


# =============================================================================
# 5. BROADCASTING
# =============================================================================

print("\n" + "=" * 80)
print("5. BROADCASTING")
print("=" * 80)

broadcast_result = a + 10.0

print("a =", a)
print("a + 10 =", broadcast_result)


# =============================================================================
# 6. RGB IMAGE TENSOR
# =============================================================================

print("\n" + "=" * 80)
print("6. RGB IMAGE TENSOR")
print("=" * 80)

H = 64
W_img = 64
C = 3

image = np.zeros((H, W_img, C), dtype=np.float32)

print("Image shape:", image.shape)
print("Image dimensions:")
print("Height :", H)
print("Width  :", W_img)
print("Channels:", C)

print("\nChannel meaning:")
print("channel 0 = Red")
print("channel 1 = Green")
print("channel 2 = Blue")


# =============================================================================
# 7. BATCH OF IMAGES
# =============================================================================

print("\n" + "=" * 80)
print("7. BATCH OF IMAGES")
print("=" * 80)

N = 8

batch = np.zeros(
    (N, H, W_img, C),
    dtype=np.float32,
)

print("Batch shape:", batch.shape)

print(
    "\nInterpretation:"
    "\nN =", N,
    "\nH =", H,
    "\nW =", W_img,
    "\nC =", C,
)


# =============================================================================
# 8. ROBOT/CAMERA COORDINATE TRANSFORMATION
# =============================================================================

print("\n" + "=" * 80)
print("8. ROBOT/CAMERA COORDINATE TRANSFORMATION")
print("=" * 80)

# Imagine a camera detects an object at this pixel coordinate.
point = np.array([100.0, 50.0])

print("Original camera coordinate:", point)


# -----------------------------------------------------------------------------
# Scaling
# -----------------------------------------------------------------------------

scale_matrix = np.array(
    [
        [1.5, 0.0],
        [0.0, 2.0],
    ]
)

scaled_point = scale_matrix @ point

print("\nScale matrix:")
print(scale_matrix)

print("Scaled point:", scaled_point)


# -----------------------------------------------------------------------------
# Rotation
# -----------------------------------------------------------------------------

theta = np.deg2rad(45.0)

rotation_matrix = np.array(
    [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
)

rotated_point = rotation_matrix @ point

print("\nRotation angle: 45 degrees")

print("Rotation matrix:")
print(rotation_matrix)

print("Rotated point:", rotated_point)


# -----------------------------------------------------------------------------
# Combined transformation
# -----------------------------------------------------------------------------

combined_transform = rotation_matrix @ scale_matrix

transformed_point = combined_transform @ point

print("\nCombined transformation:")
print(combined_transform)

print("Transformed point:", transformed_point)


# =============================================================================
# 9. NEURAL NETWORK LINEAR LAYER
# =============================================================================

print("\n" + "=" * 80)
print("9. NEURAL NETWORK LINEAR LAYER")
print("=" * 80)

# A simple neural-network layer:
#
#     y = Wx + b
#
# Input:
#     x has 3 features
#
# Output:
#     y has 2 features
#
# Therefore:
#
#     W -> shape (2,3)
#     x -> shape (3,)
#     b -> shape (2,)
#     y -> shape (2,)

x_features = np.array(
    [
        0.5,
        1.0,
        -0.5,
    ],
    dtype=np.float64,
)

W_layer = np.array(
    [
        [1.0, 2.0, 0.5],
        [-1.0, 0.5, 2.0],
    ],
    dtype=np.float64,
)

b_layer = np.array(
    [
        0.1,
        -0.2,
    ],
    dtype=np.float64,
)

linear_output = W_layer @ x_features + b_layer

print("Input x:")
print(x_features)

print("\nWeight matrix W:")
print(W_layer)

print("\nBias b:")
print(b_layer)

print("\nOutput y = W @ x + b:")
print(linear_output)

print(
    "\nManual calculation:"
    "\n"
    "y[0] = 1.0*0.5 + 2.0*1.0 + 0.5*(-0.5) + 0.1"
    "\n     = 0.5 + 2.0 - 0.25 + 0.1"
    "\n     = 2.35"
)

print(
    "\n"
    "y[1] = -1.0*0.5 + 0.5*1.0 + 2.0*(-0.5) - 0.2"
    "\n     = -0.5 + 0.5 - 1.0 - 0.2"
    "\n     = -1.2"
)


# =============================================================================
# 10. CREATE A SYNTHETIC IMAGE FOR VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("10. SYNTHETIC IMAGE VISUALIZATION")
print("=" * 80)

visual_image = np.zeros(
    (128, 128, 3),
    dtype=np.float32,
)

# Create a red rectangular object.
visual_image[35:90, 45:100, 0] = 1.0

# Create a green rectangular object.
visual_image[15:45, 15:45, 1] = 1.0

# Create a blue rectangular object.
visual_image[80:115, 15:55, 2] = 1.0

print("Synthetic image shape:", visual_image.shape)


# =============================================================================
# 11. SIMPLE OBJECT CENTER CALCULATION
# =============================================================================

print("\n" + "=" * 80)
print("11. OBJECT CENTER CALCULATION")
print("=" * 80)

red_mask = visual_image[:, :, 0] > 0.5

red_positions = np.argwhere(red_mask)

red_y_min, red_x_min = red_positions.min(axis=0)
red_y_max, red_x_max = red_positions.max(axis=0)

red_center_x = (red_x_min + red_x_max) / 2.0
red_center_y = (red_y_min + red_y_max) / 2.0

print("Red object bounding box:")
print("x_min =", red_x_min)
print("x_max =", red_x_max)
print("y_min =", red_y_min)
print("y_max =", red_y_max)

print(
    "\nRed object center:",
    np.array([red_center_x, red_center_y]),
)


# =============================================================================
# 12. VISUALIZE IMAGE AND COORDINATE TRANSFORMATION
# =============================================================================

def visualize_results():
    """
    Visualize the concepts developed in this lesson.

    The first panel shows the synthetic RGB image.

    The second panel shows the original and transformed camera coordinate.

    The visualization is not merely decorative. It connects the numerical
    representation to a physical computer-vision problem.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # -------------------------------------------------------------------------
    # Image
    # -------------------------------------------------------------------------

    axes[0].imshow(visual_image)
    axes[0].set_title("Synthetic RGB Image")
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Y pixel")

    axes[0].scatter(
        red_center_x,
        red_center_y,
        s=100,
        marker="x",
        linewidths=3,
    )

    # -------------------------------------------------------------------------
    # Coordinate transformation
    # -------------------------------------------------------------------------

    axes[1].scatter(
        point[0],
        point[1],
        s=100,
        label="Original",
    )

    axes[1].scatter(
        transformed_point[0],
        transformed_point[1],
        s=100,
        label="Transformed",
    )

    axes[1].plot(
        [point[0], transformed_point[0]],
        [point[1], transformed_point[1]],
        linestyle="--",
    )

    axes[1].set_title(
        "2-D Matrix Transformation"
    )

    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    axes[1].grid(True)
    axes[1].legend()

    axes[1].axis("equal")

    plt.tight_layout()
    plt.show()


# =============================================================================
# 13. NUMERICAL VERIFICATION / TESTS
# =============================================================================

def run_tests():
    """
    Numerical verification is essential.

    We do not want to merely print numbers and assume that they are correct.

    Each fundamental mathematical operation is independently verified.

    This is especially important because these operations will later become
    components of neural-network implementations.
    """

    print("\n" + "=" * 80)
    print("RUNNING NUMERICAL TESTS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test 1: vector addition
    # -------------------------------------------------------------------------

    expected_vector_sum = np.array(
        [5.0, 7.0, 9.0]
    )

    assert np.allclose(
        vector_sum,
        expected_vector_sum,
    ), (
        f"Vector addition failed.\n"
        f"Expected: {expected_vector_sum}\n"
        f"Actual:   {vector_sum}"
    )

    print("PASS: vector addition")


    # -------------------------------------------------------------------------
    # Test 2: element-wise multiplication
    # -------------------------------------------------------------------------

    expected_elementwise = np.array(
        [4.0, 10.0, 18.0]
    )

    assert np.allclose(
        elementwise_product,
        expected_elementwise,
    ), (
        f"Element-wise multiplication failed.\n"
        f"Expected: {expected_elementwise}\n"
        f"Actual:   {elementwise_product}"
    )

    print("PASS: element-wise multiplication")


    # -------------------------------------------------------------------------
    # Test 3: dot product
    # -------------------------------------------------------------------------

    expected_dot = 32.0

    assert np.isclose(
        dot_product,
        expected_dot,
    ), (
        f"Dot product failed.\n"
        f"Expected: {expected_dot}\n"
        f"Actual:   {dot_product}"
    )

    print("PASS: dot product")


    # -------------------------------------------------------------------------
    # Test 4: matrix shape
    # -------------------------------------------------------------------------

    assert A.shape == (2, 3), (
        f"Matrix shape failed.\n"
        f"Expected: (2, 3)\n"
        f"Actual:   {A.shape}"
    )

    print("PASS: matrix shape")


    # -------------------------------------------------------------------------
    # Test 5: matrix-vector multiplication
    # -------------------------------------------------------------------------

    expected_matrix_vector = np.array(
        [140.0, 320.0]
    )

    assert np.allclose(
        matrix_vector_result,
        expected_matrix_vector,
    ), (
        f"Matrix-vector multiplication failed.\n"
        f"Expected: {expected_matrix_vector}\n"
        f"Actual:   {matrix_vector_result}"
    )

    print("PASS: matrix-vector multiplication")


    # -------------------------------------------------------------------------
    # Test 6: matrix-matrix multiplication
    # -------------------------------------------------------------------------

    expected_matrix_matrix = np.array(
        [
            [22.0, 28.0],
            [49.0, 64.0],
        ]
    )

    assert np.allclose(
        matrix_matrix_result,
        expected_matrix_matrix,
    ), (
        f"Matrix-matrix multiplication failed.\n"
        f"Expected:\n{expected_matrix_matrix}\n"
        f"Actual:\n{matrix_matrix_result}"
    )

    print("PASS: matrix-matrix multiplication")


    # -------------------------------------------------------------------------
    # Test 7: broadcasting
    # -------------------------------------------------------------------------

    expected_broadcast = np.array(
        [11.0, 12.0, 13.0]
    )

    assert np.allclose(
        broadcast_result,
        expected_broadcast,
    ), (
        f"Broadcasting failed.\n"
        f"Expected: {expected_broadcast}\n"
        f"Actual:   {broadcast_result}"
    )

    print("PASS: broadcasting")


    # -------------------------------------------------------------------------
    # Test 8: image tensor shape
    # -------------------------------------------------------------------------

    assert image.shape == (64, 64, 3), (
        f"Image tensor shape failed.\n"
        f"Expected: (64, 64, 3)\n"
        f"Actual:   {image.shape}"
    )

    print("PASS: RGB image tensor shape")


    # -------------------------------------------------------------------------
    # Test 9: batch tensor shape
    # -------------------------------------------------------------------------

    assert batch.shape == (8, 64, 64, 3), (
        f"Batch tensor shape failed.\n"
        f"Expected: (8, 64, 64, 3)\n"
        f"Actual:   {batch.shape}"
    )

    print("PASS: batch tensor shape")


    # -------------------------------------------------------------------------
    # Test 10: scale transformation
    # -------------------------------------------------------------------------

    expected_scaled_point = np.array(
        [150.0, 100.0]
    )

    assert np.allclose(
        scaled_point,
        expected_scaled_point,
    ), (
        f"Scaling transformation failed.\n"
        f"Expected: {expected_scaled_point}\n"
        f"Actual:   {scaled_point}"
    )

    print("PASS: scaling transformation")


    # -------------------------------------------------------------------------
    # Test 11: rotation
    # -------------------------------------------------------------------------

    expected_rotated_point = np.array(
        [
            100.0 * np.cos(theta) - 50.0 * np.sin(theta),
            100.0 * np.sin(theta) + 50.0 * np.cos(theta),
        ]
    )

    assert np.allclose(
        rotated_point,
        expected_rotated_point,
    ), (
        f"Rotation transformation failed.\n"
        f"Expected: {expected_rotated_point}\n"
        f"Actual:   {rotated_point}"
    )

    print("PASS: rotation transformation")


    # -------------------------------------------------------------------------
    # Test 12: combined transformation
    # -------------------------------------------------------------------------

    expected_combined_point = (
        rotation_matrix
        @ scale_matrix
        @ point
    )

    assert np.allclose(
        transformed_point,
        expected_combined_point,
    ), (
        f"Combined transformation failed.\n"
        f"Expected: {expected_combined_point}\n"
        f"Actual:   {transformed_point}"
    )

    print("PASS: combined transformation")


    # -------------------------------------------------------------------------
    # Test 13: neural-network linear layer
    # -------------------------------------------------------------------------

    #
    # IMPORTANT:
    #
    # Compute the expected answer explicitly from the mathematical equation.
    #
    #     y = Wx + b
    #
    # First output:
    #
    #     1.0*0.5 + 2.0*1.0 + 0.5*(-0.5) + 0.1
    #
    #     = 0.5 + 2.0 - 0.25 + 0.1
    #
    #     = 2.35
    #
    # Second output:
    #
    #     -1.0*0.5 + 0.5*1.0 + 2.0*(-0.5) - 0.2
    #
    #     = -0.5 + 0.5 - 1.0 - 0.2
    #
    #     = -1.2
    #
    # Therefore:
    #
    #     y = [2.35, -1.2]
    #

    expected_linear_output = np.array(
        [2.35, -1.2]
    )

    assert np.allclose(
        linear_output,
        expected_linear_output,
    ), (
        f"Linear layer failed.\n"
        f"Expected: {expected_linear_output}\n"
        f"Actual:   {linear_output}"
    )

    print("PASS: neural-network linear layer")


    # -------------------------------------------------------------------------
    # Test 14: red object exists
    # -------------------------------------------------------------------------

    assert red_positions.shape[0] > 0, (
        "Red object detection failed."
    )

    print("PASS: red object detected")


    # -------------------------------------------------------------------------
    # Test 15: red object center
    # -------------------------------------------------------------------------

    expected_red_center = np.array(
        [
            (45 + 99) / 2.0,
            (35 + 89) / 2.0,
        ]
    )

    actual_red_center = np.array(
        [
            red_center_x,
            red_center_y,
        ]
    )

    assert np.allclose(
        actual_red_center,
        expected_red_center,
    ), (
        f"Red object center failed.\n"
        f"Expected: {expected_red_center}\n"
        f"Actual:   {actual_red_center}"
    )

    print("PASS: red object center")


    # -------------------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("ALL LESSON 2 TESTS PASSED")
    print("-" * 80)


# =============================================================================
# 14. REAL-WORLD APPLICATION
# =============================================================================

def print_real_world_application():
    """
    Explain how today's mathematics maps to an actual computer-vision system.
    """

    print("\n" + "=" * 80)
    print("REAL-WORLD APPLICATION")
    print("=" * 80)

    print(
        """
Example: Robot Camera Object Localization

A robot camera produces:

    RGB camera
        |
        v
    image tensor
        |
        | shape = H x W x 3
        v
    neural network
        |
        v
    object features
        |
        v
    object position
        |
        v
    robot coordinate transformation
        |
        v
    robot motion

Today's lesson provides the mathematical foundation for this pipeline.

The important connection is:

    image
       -> tensor
       -> matrix/tensor operations
       -> neural-network layer
       -> learned representation

The same basic numerical operations will later appear inside:

    CNN
    ResNet
    Autoencoder
    VAE
    GAN
    Transformer
    U-Net
    Diffusion Model
    Vision-Language Model
    Vision-Action Model
"""
    )


# =============================================================================
# 15. EXPERIMENT
# =============================================================================

def run_experiment():
    """
    Small experiment demonstrating that matrix transformations change a
    coordinate according to the mathematical parameters.

    Change the scale and rotation values and run the program again.

    This is intentionally simple because the goal is to build intuition before
    introducing neural-network learning.
    """

    print("\n" + "=" * 80)
    print("EXPERIMENT")
    print("=" * 80)

    experiment_point = np.array(
        [40.0, 20.0]
    )

    experiment_scale = np.array(
        [
            [2.0, 0.0],
            [0.0, 1.5],
        ]
    )

    experiment_angle = np.deg2rad(30.0)

    experiment_rotation = np.array(
        [
            [
                np.cos(experiment_angle),
                -np.sin(experiment_angle),
            ],
            [
                np.sin(experiment_angle),
                np.cos(experiment_angle),
            ],
        ]
    )

    experiment_transform = (
        experiment_rotation
        @ experiment_scale
    )

    experiment_result = (
        experiment_transform
        @ experiment_point
    )

    print("Original point:")
    print(experiment_point)

    print("\nScale:")
    print(experiment_scale)

    print("\nRotation: 30 degrees")

    print("\nFinal transformed point:")
    print(experiment_result)


# =============================================================================
# 16. LESSON SUMMARY
# =============================================================================

def print_lesson_summary():
    print("\n" + "=" * 80)
    print("LESSON 2 SUMMARY")
    print("=" * 80)

    print(
        """
You should now understand:

    [1] Vector
        An ordered collection of numbers.

    [2] Matrix
        A rectangular numerical structure.

    [3] Tensor
        A multidimensional numerical structure.

    [4] Shape
        The dimensions of a tensor.

    [5] Dot product
        A fundamental operation used by neurons.

    [6] Matrix multiplication
        A fundamental operation used by neural-network layers.

    [7] Broadcasting
        A mechanism for applying compatible operations across dimensions.

    [8] Image tensor
        H x W x C

    [9] Batch tensor
        N x H x W x C

    [10] Coordinate transformation
        Matrix multiplication can transform spatial coordinates.

    [11] Neural-network linear layer

        y = Wx + b

This equation is one of the most important equations in the entire course.
"""
    )


# =============================================================================
# 17. CONNECTION TO NEXT LESSON
# =============================================================================

def print_next_lesson():
    print("\n" + "=" * 80)
    print("CONNECTION TO LESSON 3")
    print("=" * 80)

    print(
        """
LESSON 2:
    We learned how numerical data is represented and transformed.

The important neural-network equation was:

    y = Wx + b

But one major question remains:

    HOW DO WE FIND W AND b?

We do not normally know the correct values beforehand.

A neural network learns them from data.

That leads directly to:

    LOSS
      |
      v
    GRADIENT
      |
      v
    PARAMETER UPDATE
      |
      v
    LOWER LOSS

Lesson 3 will build:

    GRADIENT DESCENT FROM SCRATCH

We will calculate derivatives numerically and analytically, visualize a loss
function, and train parameters using gradient descent.

That will be the first real step from:

    "performing mathematical operations"

to:

    "learning mathematical parameters from data."

This is the foundation of neural-network training.
"""
    )


# =============================================================================
# MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("COMPUTER VISION ADVANCED CRASH COURSE")
    print("LESSON 2 — VECTORS, MATRICES, AND TENSORS")
    print("=" * 80)

    print_real_world_application()

    run_experiment()

    run_tests()

    print_lesson_summary()

    print_next_lesson()

    print("\nStarting visualization...")

    visualize_results()

    print("\nLesson 2 completed successfully.")
    print("Next: Lesson 3 — Gradient Descent From Scratch.")
