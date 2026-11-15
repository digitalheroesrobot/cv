"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 5 — CONVOLUTION FROM SCRATCH
================================================================================

PURPOSE
-------
Understand exactly how convolution-based feature extraction works in a
computer-vision neural network.

In the previous lessons we learned:

    Image
      ↓
    Numerical tensor
      ↓
    Vectors / matrices / tensors
      ↓
    Gradient descent
      ↓
    Neural network
      ↓
    Nonlinear classification

Now we need to answer an important question:

    "How should a neural network process an image?"

A fully connected network treats every pixel as an unrelated input feature.
That is inefficient because images have strong spatial structure.

CNNs solve this using local receptive fields and learnable filters.

This lesson builds the core convolution operation FROM SCRATCH.

No:
    PyTorch
    TensorFlow
    JAX
    OpenCV
    pretrained models
    pretrained weights
    autodiff

Only:
    NumPy
    Matplotlib

The goal is understanding, not hiding the operation inside a framework.


================================================================================
WHAT YOU WILL LEARN
================================================================================

1. Why fully connected layers are inefficient for images
2. Local receptive fields
3. Kernels / filters
4. Feature maps
5. Convolution vs cross-correlation
6. Stride
7. Padding
8. Edge detection
9. Horizontal and vertical edge filters
10. Multi-channel image convolution
11. Multiple filters
12. Numerical verification
13. Visualization of learned-style feature extraction
14. Real-world computer-vision applications
15. Why convolution leads directly to CNNs


================================================================================
TECHNICAL ROADMAP
================================================================================

CURRENT COURSE POSITION

    Phase 0
    Mathematics
        ↓
    Phase 1
    Image / Tensor Representation
        ↓
    Phase 2
    Neural Networks From Scratch
        ↓
    >>> LESSON 5: CONVOLUTION <<<
        ↓
    Lesson 6
    Complete CNN From Scratch
        ↓
    CNN Training
        ↓
    ResNet-style architectures
        ↓
    Autoencoders / VAEs
        ↓
    GANs
        ↓
    Attention / Transformers
        ↓
    Diffusion Models
        ↓
    Your Own Generative Vision Model


================================================================================
REAL-WORLD PROBLEM
================================================================================

Consider a robot camera looking at a table.

The camera may see:

    +--------------------------------+
    |                                |
    |          ______                |
    |         /      \               |
    |        |  CUP   |              |
    |         \______/               |
    |                                |
    +--------------------------------+

The raw RGB pixels do not directly say:

    "This is a cup."

However, the image contains useful local structures:

    edges
    corners
    contours
    textures
    shapes

For example, the boundary of the cup creates strong local intensity changes.

A convolution filter can detect those local structures.

This is the fundamental idea behind CNNs.


================================================================================
1. WHY FULLY CONNECTED NETWORKS ARE NOT IDEAL FOR IMAGES
================================================================================

Suppose an RGB camera produces:

    height = 224
    width  = 224
    channels = 3

Number of input values:

    224 * 224 * 3 = 150,528

A fully connected layer with 1,000 neurons would require approximately:

    150,528 * 1,000
    = 150 million weights

That is enormous.

More importantly, a fully connected layer does not naturally understand
that nearby pixels are spatially related.

A pixel at:

    (100, 100)

is treated similarly to a pixel at:

    (10, 10)

in terms of connectivity.

CNNs instead exploit:

    LOCALITY

and

    WEIGHT SHARING


================================================================================
2. LOCAL RECEPTIVE FIELD
================================================================================

Instead of looking at the entire image at once, a convolution filter looks
at a small local region.

For example:

    3 x 3

window:

        +---+---+---+
        | x | x | x |
        +---+---+---+
        | x | x | x |
        +---+---+---+
        | x | x | x |
        +---+---+---+

The filter moves across the image.

At each location it computes a weighted sum.

This local region is called the:

    RECEPTIVE FIELD


================================================================================
3. KERNEL / FILTER
================================================================================

A kernel is a small matrix of weights.

Example:

    vertical edge detector

        -1   0   +1
        -1   0   +1
        -1   0   +1

If the left side of a local region is dark and the right side is bright,
the filter produces a strong response.

This creates a feature map.


================================================================================
4. CONVOLUTION VS CROSS-CORRELATION
================================================================================

There is an important terminology detail.

Mathematically, convolution flips the kernel before applying it.

For a 2D kernel K:

    K_flipped = flip vertically + horizontally

However, most CNN implementations use the operation called:

    CROSS-CORRELATION

without flipping the kernel.

When people say:

    "CNN convolution"

they usually mean this cross-correlation operation.

In this lesson we implement the CNN-style operation.

That means:

    output[i,j]
        =
    sum(
        image[i:i+k, j:j+k] * kernel
    )

This distinction is important when comparing our implementation against
mathematical signal-processing definitions.


================================================================================
5. BASIC EQUATION
================================================================================

For an image X and kernel K:

    Y[i,j]
        =
    Σ_m Σ_n
        X[i+m, j+n] K[m,n]

where:

    i = output row
    j = output column
    m,n = kernel coordinates

This is simply:

    local weighted sum


================================================================================
6. OUTPUT SIZE
================================================================================

For a 2D input:

    H x W

with:

    kernel size = K

padding = P

stride = S

the output dimensions are:

    H_out = floor((H + 2P - K) / S) + 1

    W_out = floor((W + 2P - K) / S) + 1

For example:

    input = 7 x 7
    kernel = 3 x 3
    padding = 0
    stride = 1

Then:

    H_out = (7 - 3) / 1 + 1
          = 5

So:

    output = 5 x 5


================================================================================
7. STRIDE
================================================================================

Stride controls how far the filter moves.

Stride = 1:

    move one pixel at a time

Stride = 2:

    move two pixels at a time

Larger stride:

    smaller output
    lower spatial resolution
    more aggressive downsampling


================================================================================
8. PADDING
================================================================================

Without padding, the spatial dimensions shrink.

For example:

    7 x 7
       ↓
    3 x 3 kernel
       ↓
    5 x 5

Padding adds values around the border.

Usually:

    zero padding

For:

    P = 1

a 7 x 7 image becomes conceptually:

    9 x 9

before applying a 3 x 3 kernel.

Then:

    (9 - 3) + 1 = 7

So:

    7 x 7 → 7 x 7

This is commonly called:

    SAME padding

when stride = 1 and kernel size is odd.


================================================================================
9. EDGE DETECTION
================================================================================

Edges are locations where intensity changes rapidly.

Example:

    dark dark dark | bright bright bright

The transition between dark and bright is an edge.

A simple vertical-edge detector is:

        -1   0   +1
        -1   0   +1
        -1   0   +1

A horizontal-edge detector is:

        -1  -1  -1
         0   0   0
        +1  +1  +1

These filters are hand-designed.

In a CNN, the important idea is:

    WE DO NOT HAVE TO DESIGN THESE FILTERS MANUALLY.

The network can learn them from data.

For example, an early CNN layer may learn filters resembling:

    edge detectors
    corner detectors
    texture detectors

Later layers combine these simpler features into more complex structures.


================================================================================
10. MULTI-CHANNEL IMAGES
================================================================================

Real camera images have multiple channels.

RGB:

    H x W x 3

A convolution filter must therefore also have channel dimensions.

For example:

    3 x 3 x 3

means:

    height  = 3
    width   = 3
    channels = 3

The output is one feature map.

Mathematically:

    Y[i,j]
        =
    Σ_c Σ_m Σ_n
        X[i+m,j+n,c] K[m,n,c]

where:

    c = channel index


================================================================================
11. MULTIPLE FILTERS
================================================================================

Suppose the input is:

    H x W x 3

and we use:

    16 filters

Each filter has shape:

    3 x 3 x 3

Then the output has:

    16 feature maps

Therefore:

    H x W x 3
         ↓
    convolution
         ↓
    H x W x 16

This is the basic operation of a convolutional layer.


================================================================================
12. WEIGHT SHARING
================================================================================

A major advantage of convolution is weight sharing.

The same kernel is used at every spatial position.

Suppose a 3 x 3 kernel is used.

It does NOT require a different set of 9 weights for every image position.

Instead:

    one kernel
        ↓
    reused everywhere

This dramatically reduces the number of parameters.

It also provides a useful property:

    if a pattern appears in different locations,
    the same filter can detect it.


================================================================================
13. TODAY'S IMPLEMENTATION
================================================================================

We will implement:

    convolution_2d()

    convolution_2d_with_padding()

    convolution_multi_channel()

    convolution_multiple_filters()

We will then:

    1. Generate a synthetic image
    2. Apply edge detectors
    3. Verify output dimensions
    4. Compare two implementations
    5. Apply filters to an RGB image
    6. Visualize feature maps
    7. Demonstrate stride
    8. Demonstrate padding
    9. Demonstrate multiple filters
    10. Connect this to a real camera/robot problem


================================================================================
IMPLEMENTATION
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1 — BASIC 2D CROSS-CORRELATION
# ==============================================================================

def convolution_2d(image, kernel, stride=1):
    """
    CNN-style 2D cross-correlation.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Single-channel image.

    kernel : ndarray, shape (K, K)
        Convolution kernel.

    stride : int
        Number of pixels to move the kernel each step.

    Returns
    -------
    output : ndarray
        Feature map.
    """

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if kernel.ndim != 2:
        raise ValueError("kernel must be a 2D array")

    if stride <= 0:
        raise ValueError("stride must be positive")

    H, W = image.shape
    KH, KW = kernel.shape

    if KH > H or KW > W:
        raise ValueError("kernel cannot be larger than image")

    output_height = (H - KH) // stride + 1
    output_width = (W - KW) // stride + 1

    output = np.zeros(
        (output_height, output_width),
        dtype=np.float64
    )

    for i in range(output_height):
        for j in range(output_width):

            row_start = i * stride
            col_start = j * stride

            local_region = image[
                row_start:row_start + KH,
                col_start:col_start + KW
            ]

            output[i, j] = np.sum(local_region * kernel)

    return output


# ==============================================================================
# SECTION 2 — PADDING
# ==============================================================================

def zero_pad_2d(image, padding):
    """
    Add zero padding around a 2D image.
    """

    if padding < 0:
        raise ValueError("padding must be >= 0")

    if padding == 0:
        return image.copy()

    return np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )


def convolution_2d_with_padding(
    image,
    kernel,
    stride=1,
    padding=0
):
    """
    2D CNN-style cross-correlation with zero padding.
    """

    padded_image = zero_pad_2d(image, padding)

    return convolution_2d(
        padded_image,
        kernel,
        stride=stride
    )


# ==============================================================================
# SECTION 3 — MULTI-CHANNEL CONVOLUTION
# ==============================================================================

def convolution_multi_channel(
    image,
    kernel,
    stride=1,
    padding=0
):
    """
    Convolution for a multi-channel image.

    image:
        H x W x C

    kernel:
        KH x KW x C

    output:
        H_out x W_out

    One kernel produces one feature map.
    """

    if image.ndim != 3:
        raise ValueError("image must have shape (H, W, C)")

    if kernel.ndim != 3:
        raise ValueError("kernel must have shape (KH, KW, C)")

    H, W, C = image.shape
    KH, KW, KC = kernel.shape

    if C != KC:
        raise ValueError(
            "Image channels and kernel channels must match"
        )

    if stride <= 0:
        raise ValueError("stride must be positive")

    padded_image = np.pad(
        image,
        (
            (padding, padding),
            (padding, padding),
            (0, 0)
        ),
        mode="constant",
        constant_values=0
    )

    padded_H, padded_W, _ = padded_image.shape

    output_height = (
        (padded_H - KH) // stride
    ) + 1

    output_width = (
        (padded_W - KW) // stride
    ) + 1

    output = np.zeros(
        (output_height, output_width),
        dtype=np.float64
    )

    for i in range(output_height):
        for j in range(output_width):

            row_start = i * stride
            col_start = j * stride

            local_region = padded_image[
                row_start:row_start + KH,
                col_start:col_start + KW,
                :
            ]

            output[i, j] = np.sum(
                local_region * kernel
            )

    return output


# ==============================================================================
# SECTION 4 — MULTIPLE FILTERS
# ==============================================================================

def convolution_multiple_filters(
    image,
    kernels,
    stride=1,
    padding=0
):
    """
    Apply multiple multi-channel kernels.

    image:
        H x W x C

    kernels:
        N_filters x KH x KW x C

    output:
        H_out x W_out x N_filters
    """

    if image.ndim != 3:
        raise ValueError("image must have shape (H, W, C)")

    if kernels.ndim != 4:
        raise ValueError(
            "kernels must have shape "
            "(N_filters, KH, KW, C)"
        )

    number_of_filters = kernels.shape[0]

    feature_maps = []

    for filter_index in range(number_of_filters):

        feature_map = convolution_multi_channel(
            image,
            kernels[filter_index],
            stride=stride,
            padding=padding
        )

        feature_maps.append(feature_map)

    return np.stack(
        feature_maps,
        axis=-1
    )


# ==============================================================================
# SECTION 5 — RELU
# ==============================================================================

def relu(x):
    """
    ReLU activation.

        ReLU(x) = max(0, x)

    CNNs normally combine convolution with nonlinear activation.

    Convolution:
        extracts a feature response

    ReLU:
        introduces nonlinearity
    """

    return np.maximum(0, x)


# ==============================================================================
# SECTION 6 — SYNTHETIC IMAGE
# ==============================================================================

def create_test_image(size=64):
    """
    Create a synthetic image containing geometric structures.

    This gives us a controlled environment for testing edge detection.
    """

    image = np.zeros(
        (size, size),
        dtype=np.float64
    )

    # Large bright rectangle.
    image[15:48, 20:44] = 1.0

    # Smaller bright square.
    image[25:35, 8:18] = 0.8

    # Diagonal structure.
    for i in range(10, 30):
        j = i + 30

        if 0 <= j < size:
            image[i, j] = 1.0

    return image


# ==============================================================================
# SECTION 7 — EDGE DETECTION KERNELS
# ==============================================================================

vertical_edge_kernel = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ]
)

horizontal_edge_kernel = np.array(
    [
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)

# A simple sharpening/high-frequency filter.
sharpen_kernel = np.array(
    [
        [0.0, -1.0, 0.0],
        [-1.0, 5.0, -1.0],
        [0.0, -1.0, 0.0],
    ]
)

# Simple blur/average filter.
blur_kernel = np.ones(
    (3, 3),
    dtype=np.float64
) / 9.0


# ==============================================================================
# SECTION 8 — REFERENCE IMPLEMENTATION FOR VERIFICATION
# ==============================================================================

def reference_convolution_2d(image, kernel, stride=1):
    """
    Very explicit reference implementation.

    This intentionally uses direct scalar multiplication inside
    the innermost loop.

    It is slower, but it makes the mathematics obvious.

    We will compare it against convolution_2d().
    """

    H, W = image.shape
    KH, KW = kernel.shape

    output_height = (H - KH) // stride + 1
    output_width = (W - KW) // stride + 1

    output = np.zeros(
        (output_height, output_width),
        dtype=np.float64
    )

    for i in range(output_height):

        for j in range(output_width):

            total = 0.0

            for m in range(KH):

                for n in range(KW):

                    image_value = image[
                        i * stride + m,
                        j * stride + n
                    ]

                    kernel_value = kernel[m, n]

                    total += (
                        image_value *
                        kernel_value
                    )

            output[i, j] = total

    return output


# ==============================================================================
# SECTION 9 — NUMERICAL VERIFICATION
# ==============================================================================

def run_basic_tests():
    """
    Verify the fundamental convolution implementation.

    Tests include:

        shape correctness
        known numerical result
        reference implementation agreement
        padding behavior
        stride behavior
        multi-channel behavior
        multiple-filter behavior
    """

    print("\n" + "=" * 78)
    print("RUNNING LESSON 5 TESTS")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # Test 1 — Known simple example
    # --------------------------------------------------------------------------

    image = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        dtype=np.float64
    )

    kernel = np.array(
        [
            [1, 0],
            [0, 1]
        ],
        dtype=np.float64
    )

    result = convolution_2d(
        image,
        kernel
    )

    expected = np.array(
        [
            [6, 8],
            [12, 14]
        ],
        dtype=np.float64
    )

    assert result.shape == (2, 2)

    assert np.allclose(
        result,
        expected
    )

    print("PASS: basic numerical convolution")

    # --------------------------------------------------------------------------
    # Test 2 — Reference implementation
    # --------------------------------------------------------------------------

    rng = np.random.default_rng(123)

    random_image = rng.normal(
        size=(11, 13)
    )

    random_kernel = rng.normal(
        size=(3, 3)
    )

    output_fast_style = convolution_2d(
        random_image,
        random_kernel,
        stride=1
    )

    output_reference = reference_convolution_2d(
        random_image,
        random_kernel,
        stride=1
    )

    assert np.allclose(
        output_fast_style,
        output_reference,
        atol=1e-12
    )

    print(
        "PASS: implementation matches "
        "reference implementation"
    )

    # --------------------------------------------------------------------------
    # Test 3 — Stride
    # --------------------------------------------------------------------------

    stride_output = convolution_2d(
        random_image,
        random_kernel,
        stride=2
    )

    expected_height = (
        (11 - 3) // 2
    ) + 1

    expected_width = (
        (13 - 3) // 2
    ) + 1

    assert stride_output.shape == (
        expected_height,
        expected_width
    )

    print("PASS: stride output dimensions")

    # --------------------------------------------------------------------------
    # Test 4 — Padding
    # --------------------------------------------------------------------------

    padded_output = convolution_2d_with_padding(
        random_image,
        random_kernel,
        stride=1,
        padding=1
    )

    assert padded_output.shape == (
        11,
        13
    )

    print("PASS: padding output dimensions")

    # --------------------------------------------------------------------------
    # Test 5 — Multi-channel
    # --------------------------------------------------------------------------

    rgb_image = rng.normal(
        size=(8, 9, 3)
    )

    rgb_kernel = rng.normal(
        size=(3, 3, 3)
    )

    rgb_output = convolution_multi_channel(
        rgb_image,
        rgb_kernel,
        padding=1
    )

    assert rgb_output.shape == (
        8,
        9
    )

    print("PASS: multi-channel convolution")

    # --------------------------------------------------------------------------
    # Test 6 — Multiple filters
    # --------------------------------------------------------------------------

    filters = rng.normal(
        size=(5, 3, 3, 3)
    )

    feature_maps = convolution_multiple_filters(
        rgb_image,
        filters,
        padding=1
    )

    assert feature_maps.shape == (
        8,
        9,
        5
    )

    print("PASS: multiple filters")

    # --------------------------------------------------------------------------
    # Test 7 — Finite values
    # --------------------------------------------------------------------------

    assert np.all(
        np.isfinite(feature_maps)
    )

    print("PASS: outputs contain finite values")

    print("=" * 78)
    print("ALL BASIC TESTS PASSED")
    print("=" * 78)


# ==============================================================================
# SECTION 10 — VISUALIZATION
# ==============================================================================

def visualize_edge_detection(
    image,
    vertical_edges,
    horizontal_edges,
    sharpened
):
    """
    Visualize the effect of several filters.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10)
    )

    axes[0, 0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0, 0].set_title(
        "Original Synthetic Image"
    )

    axes[0, 1].imshow(
        vertical_edges,
        cmap="gray"
    )

    axes[0, 1].set_title(
        "Vertical Edge Response"
    )

    axes[1, 0].imshow(
        horizontal_edges,
        cmap="gray"
    )

    axes[1, 0].set_title(
        "Horizontal Edge Response"
    )

    axes[1, 1].imshow(
        sharpened,
        cmap="gray"
    )

    axes[1, 1].set_title(
        "Sharpening / High-Frequency Response"
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_stride_and_padding(
    image,
    normal_output,
    stride_output,
    padded_output
):
    """
    Show how stride and padding affect spatial dimensions.
    """

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes[0].imshow(
        image,
        cmap="gray"
    )

    axes[0].set_title(
        f"Input\n{image.shape}"
    )

    axes[1].imshow(
        normal_output,
        cmap="gray"
    )

    axes[1].set_title(
        f"Stride=1, P=0\n{normal_output.shape}"
    )

    axes[2].imshow(
        stride_output,
        cmap="gray"
    )

    axes[2].set_title(
        f"Stride=2, P=0\n{stride_output.shape}"
    )

    axes[3].imshow(
        padded_output,
        cmap="gray"
    )

    axes[3].set_title(
        f"Stride=1, P=1\n{padded_output.shape}"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_rgb_feature_maps(
    rgb_image,
    feature_maps
):
    """
    Visualize an RGB image and feature maps produced by multiple filters.
    """

    number_of_filters = feature_maps.shape[-1]

    rows = 2
    columns = 4

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 7)
    )

    axes[0, 0].imshow(
        rgb_image
    )

    axes[0, 0].set_title(
        "RGB Input"
    )

    axes[0, 0].axis("off")

    for filter_index in range(
        min(number_of_filters, 7)
    ):

        row = (
            (filter_index + 1) // columns
        )

        col = (
            (filter_index + 1) % columns
        )

        axes[row, col].imshow(
            feature_maps[:, :, filter_index],
            cmap="gray"
        )

        axes[row, col].set_title(
            f"Feature Map {filter_index}"
        )

        axes[row, col].axis("off")

    # Turn off unused axes.
    for index in range(
        number_of_filters + 1,
        rows * columns
    ):

        row = index // columns
        col = index % columns

        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# SECTION 11 — RGB SYNTHETIC CAMERA IMAGE
# ==============================================================================

def create_rgb_camera_scene(
    height=96,
    width=128
):
    """
    Create a synthetic RGB camera frame.

    Scene:

        dark background
        bright rectangular object
        colored object
        floor-like region

    This approximates the type of structured image that a robot camera
    might provide.
    """

    image = np.zeros(
        (height, width, 3),
        dtype=np.float64
    )

    # Background.
    image[:, :, :] = 0.08

    # Floor region.
    image[
        int(height * 0.65):,
        :,
        :
    ] = np.array(
        [0.20, 0.20, 0.20]
    )

    # Large object.
    image[
        int(height * 0.25):int(height * 0.62),
        int(width * 0.30):int(width * 0.55),
        :
    ] = np.array(
        [0.75, 0.75, 0.75]
    )

    # Red object.
    image[
        int(height * 0.40):int(height * 0.58),
        int(width * 0.68):int(width * 0.83),
        :
    ] = np.array(
        [0.90, 0.10, 0.10]
    )

    # Blue object.
    image[
        int(height * 0.68):int(height * 0.88),
        int(width * 0.15):int(width * 0.32),
        :
    ] = np.array(
        [0.10, 0.20, 0.90]
    )

    return image


# ==============================================================================
# SECTION 12 — RGB FILTER EXPERIMENT
# ==============================================================================

def create_rgb_edge_filters():
    """
    Create several multi-channel filters.

    Each filter has:

        3 x 3 x 3

dimensions.

    We intentionally construct filters that emphasize different
    image characteristics.

    In a real CNN these values would normally be learned.
    """

    filters = []

    # --------------------------------------------------------------------------
    # Filter 1
    # Simple luminance-like vertical edge.
    # --------------------------------------------------------------------------

    base_vertical = vertical_edge_kernel

    filter_1 = np.stack(
        [
            base_vertical,
            base_vertical,
            base_vertical
        ],
        axis=-1
    )

    filters.append(filter_1)

    # --------------------------------------------------------------------------
    # Filter 2
    # Simple luminance-like horizontal edge.
    # --------------------------------------------------------------------------

    base_horizontal = horizontal_edge_kernel

    filter_2 = np.stack(
        [
            base_horizontal,
            base_horizontal,
            base_horizontal
        ],
        axis=-1
    )

    filters.append(filter_2)

    # --------------------------------------------------------------------------
    # Filter 3
    # Red-channel emphasis.
    # --------------------------------------------------------------------------

    filter_3 = np.zeros(
        (3, 3, 3),
        dtype=np.float64
    )

    filter_3[:, :, 0] = (
        np.ones((3, 3)) / 9.0
    )

    filter_3[:, :, 1] = (
        -0.5 * np.ones((3, 3)) / 9.0
    )

    filter_3[:, :, 2] = (
        -0.5 * np.ones((3, 3)) / 9.0
    )

    filters.append(filter_3)

    # --------------------------------------------------------------------------
    # Filter 4
    # Blue-channel emphasis.
    # --------------------------------------------------------------------------

    filter_4 = np.zeros(
        (3, 3, 3),
        dtype=np.float64
    )

    filter_4[:, :, 0] = (
        -0.5 * np.ones((3, 3)) / 9.0
    )

    filter_4[:, :, 1] = (
        -0.5 * np.ones((3, 3)) / 9.0
    )

    filter_4[:, :, 2] = (
        np.ones((3, 3)) / 9.0
    )

    filters.append(filter_4)

    return np.stack(
        filters,
        axis=0
    )


# ==============================================================================
# SECTION 13 — REAL-WORLD APPLICATION EXPERIMENT
# ==============================================================================

def run_real_world_robot_example():
    """
    Demonstrate how convolution can be used in a robot-camera pipeline.

    Conceptual pipeline:

        Robot camera
             |
             v
        RGB image
             |
             v
        convolution filters
             |
             v
        edge / texture features
             |
             v
        CNN feature extraction
             |
             v
        object recognition / localization
             |
             v
        robot action

    We simulate the first part of that pipeline here.
    """

    print("\n" + "=" * 78)
    print("REAL-WORLD ROBOT CAMERA EXPERIMENT")
    print("=" * 78)

    rgb_image = create_rgb_camera_scene()

    kernels = create_rgb_edge_filters()

    feature_maps = convolution_multiple_filters(
        rgb_image,
        kernels,
        stride=1,
        padding=1
    )

    activated_feature_maps = relu(
        feature_maps
    )

    print(
        "Camera image shape:",
        rgb_image.shape
    )

    print(
        "Number of convolution filters:",
        kernels.shape[0]
    )

    print(
        "Kernel shape:",
        kernels.shape[1:]
    )

    print(
        "Feature map tensor shape:",
        activated_feature_maps.shape
    )

    # The feature map tensor is:

    # H x W x N_filters

    # This means the original RGB image has been transformed into a
    # learned-style feature representation.

    print(
        "\nExample interpretation:"
    )

    print(
        "  Feature map 0 -> vertical structure response"
    )

    print(
        "  Feature map 1 -> horizontal structure response"
    )

    print(
        "  Feature map 2 -> red-vs-other color response"
    )

    print(
        "  Feature map 3 -> blue-vs-other color response"
    )

    print(
        "\nThis is the beginning of a CNN perception pipeline."
    )

    return (
        rgb_image,
        activated_feature_maps
    )


# ==============================================================================
# SECTION 14 — PARAMETER COUNT COMPARISON
# ==============================================================================

def compare_parameter_efficiency():
    """
    Compare the number of parameters in a fully connected layer
    versus a convolution layer.

    This demonstrates one of the central reasons CNNs are useful.
    """

    print("\n" + "=" * 78)
    print("PARAMETER EFFICIENCY COMPARISON")
    print("=" * 78)

    H = 224
    W = 224
    C = 3

    dense_neurons = 64

    dense_parameters = (
        H * W * C * dense_neurons
        + dense_neurons
    )

    kernel_height = 3
    kernel_width = 3
    filters = 64

    convolution_parameters = (
        kernel_height *
        kernel_width *
        C *
        filters
        + filters
    )

    print(
        f"Image size: {H} x {W} x {C}"
    )

    print(
        f"Dense layer neurons: {dense_neurons}"
    )

    print(
        f"Dense parameters: {dense_parameters:,}"
    )

    print(
        f"Convolution filters: {filters}"
    )

    print(
        f"Convolution parameters: "
        f"{convolution_parameters:,}"
    )

    reduction = (
        dense_parameters /
        convolution_parameters
    )

    print(
        f"\nParameter-count ratio: "
        f"{reduction:.1f}x"
    )

    print(
        "\nImportant:"
    )

    print(
        "The convolution layer also preserves spatial structure"
    )

    print(
        "and reuses the same filter across the image."
    )


# ==============================================================================
# SECTION 15 — MAIN LESSON
# ==============================================================================

def main():

    print("\n")
    print("=" * 78)
    print("COMPUTER VISION ADVANCED CRASH COURSE")
    print("LESSON 5 — CONVOLUTION FROM SCRATCH")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # Step 1
    # Basic tests
    # --------------------------------------------------------------------------

    run_basic_tests()

    # --------------------------------------------------------------------------
    # Step 2
    # Create synthetic grayscale image
    # --------------------------------------------------------------------------

    image = create_test_image()

    print("\nSynthetic grayscale image:")
    print("Shape:", image.shape)

    # --------------------------------------------------------------------------
    # Step 3
    # Apply edge filters
    # --------------------------------------------------------------------------

    vertical_edges = convolution_2d(
        image,
        vertical_edge_kernel
    )

    horizontal_edges = convolution_2d(
        image,
        horizontal_edge_kernel
    )

    sharpened = convolution_2d(
        image,
        sharpen_kernel,
        stride=1
    )

    # --------------------------------------------------------------------------
    # Step 4
    # Visualize edge extraction
    # --------------------------------------------------------------------------

    visualize_edge_detection(
        image,
        vertical_edges,
        horizontal_edges,
        sharpened
    )

    # --------------------------------------------------------------------------
    # Step 5
    # Stride and padding
    # --------------------------------------------------------------------------

    normal_output = convolution_2d(
        image,
        vertical_edge_kernel,
        stride=1
    )

    stride_output = convolution_2d(
        image,
        vertical_edge_kernel,
        stride=2
    )

    padded_output = convolution_2d_with_padding(
        image,
        vertical_edge_kernel,
        stride=1,
        padding=1
    )

    print("\nStride / padding experiment:")

    print(
        "Input shape:",
        image.shape
    )

    print(
        "Stride=1, padding=0:",
        normal_output.shape
    )

    print(
        "Stride=2, padding=0:",
        stride_output.shape
    )

    print(
        "Stride=1, padding=1:",
        padded_output.shape
    )

    visualize_stride_and_padding(
        image,
        normal_output,
        stride_output,
        padded_output
    )

    # --------------------------------------------------------------------------
    # Step 6
    # RGB convolution
    # --------------------------------------------------------------------------

    rgb_image = create_rgb_camera_scene()

    kernels = create_rgb_edge_filters()

    feature_maps = convolution_multiple_filters(
        rgb_image,
        kernels,
        stride=1,
        padding=1
    )

    feature_maps_relu = relu(
        feature_maps
    )

    print("\nRGB convolution experiment:")

    print(
        "RGB image:",
        rgb_image.shape
    )

    print(
        "Filters:",
        kernels.shape
    )

    print(
        "Feature maps:",
        feature_maps.shape
    )

    print(
        "After ReLU:",
        feature_maps_relu.shape
    )

    visualize_rgb_feature_maps(
        rgb_image,
        feature_maps_relu
    )

    # --------------------------------------------------------------------------
    # Step 7
    # Real-world robot application
    # --------------------------------------------------------------------------

    robot_rgb, robot_features = (
        run_real_world_robot_example()
    )

    # --------------------------------------------------------------------------
    # Step 8
    # Parameter efficiency
    # --------------------------------------------------------------------------

    compare_parameter_efficiency()

    # --------------------------------------------------------------------------
    # Step 9
    # Final verification
    # --------------------------------------------------------------------------

    assert image.ndim == 2

    assert vertical_edges.ndim == 2

    assert horizontal_edges.ndim == 2

    assert robot_rgb.shape[-1] == 3

    assert robot_features.ndim == 3

    assert robot_features.shape[-1] == 4

    assert np.all(
        np.isfinite(robot_features)
    )

    assert np.all(
        robot_features >= 0
    )

    print("\n" + "=" * 78)
    print("FINAL LESSON VERIFICATION")
    print("=" * 78)

    print("PASS: grayscale convolution")
    print("PASS: edge detection")
    print("PASS: stride")
    print("PASS: padding")
    print("PASS: multi-channel convolution")
    print("PASS: multiple filters")
    print("PASS: ReLU feature extraction")
    print("PASS: numerical verification")
    print("PASS: real-world RGB camera experiment")

    print("=" * 78)
    print("LESSON 5 COMPLETED SUCCESSFULLY")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # LESSON RESULT
    # --------------------------------------------------------------------------

    print(
        "\nLESSON RESULT:"
    )

    print(
        "We can now transform an image into spatial feature maps "
        "using convolution."
    )

    print(
        "We understand how kernels detect local patterns."
    )

    print(
        "We understand stride and padding."
    )

    print(
        "We understand multi-channel convolution."
    )

    print(
        "We understand how multiple filters produce multiple feature maps."
    )

    print(
        "We understand why CNNs are much more suitable for images "
        "than naive fully connected networks."
    )

    print(
        "\nREAL-WORLD RESULT:"
    )

    print(
        "A robot camera image can be converted into spatial feature maps "
        "that emphasize edges, structures, and color patterns."
    )

    print(
        "\nNEXT LESSON:"
    )

    print(
        "Lesson 6 — Build a Complete CNN From Scratch"
    )

    print(
        "Next we will combine:"
    )

    print(
        "    Convolution"
        " -> ReLU"
        " -> Pooling"
        " -> Flatten"
        " -> Fully Connected Layer"
        " -> Classification"
    )

    print(
        "\nThis will turn today's individual convolution operation "
        "into a complete trainable vision model."
    )


if __name__ == "__main__":
    main()