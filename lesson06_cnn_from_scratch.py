"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 6 — COMPLETE CNN FROM SCRATCH
================================================================================

FILENAME
--------
lesson06_cnn_from_scratch.py


================================================================================
PURPOSE
================================================================================

In Lesson 5, we manually implemented convolution.

We learned:

    image
      ↓
    local receptive field
      ↓
    kernel/filter
      ↓
    feature map

But a real CNN does more than simply apply manually designed filters.

A CNN LEARNS its filters from data.

This lesson builds a complete trainable CNN-like image classifier from scratch.

The model will learn:

    convolution
        ↓
    ReLU
        ↓
    pooling
        ↓
    convolution
        ↓
    ReLU
        ↓
    pooling
        ↓
    flatten
        ↓
    fully connected layer
        ↓
    softmax
        ↓
    classification


IMPORTANT
---------

This is an educational CNN implementation.

The goal is to understand:

    forward propagation
    convolution
    activation
    pooling
    flattening
    classification
    loss
    backpropagation
    parameter updates

We intentionally avoid:

    PyTorch
    TensorFlow
    JAX
    OpenCV
    pretrained models
    pretrained weights
    automatic differentiation


================================================================================
WHAT YOU WILL LEARN
================================================================================

1. What a CNN actually does as a complete system
2. Why convolution extracts spatial features
3. Why ReLU is required
4. Why pooling is useful
5. How feature maps become classification features
6. How softmax converts logits into probabilities
7. Cross-entropy loss
8. CNN forward propagation
9. CNN backpropagation
10. Training convolution filters from scratch
11. Numerical gradient checking
12. Image classification
13. Training curves
14. Confusion matrix
15. Learned convolution filters
16. Real-world camera classification
17. Why CNNs became foundational computer-vision models


================================================================================
COURSE ROADMAP
================================================================================

LESSON 1
    Images are numerical data
        ↓
LESSON 2
    Vectors / matrices / tensors
        ↓
LESSON 3
    Gradient descent
        ↓
LESSON 4
    Neural network from scratch
        ↓
LESSON 5
    Convolution from scratch
        ↓
>>> LESSON 6
    COMPLETE CNN FROM SCRATCH
        ↓
LESSON 7
    Better CNN architecture / training
        ↓
LESSON 8+
    ResNet-style networks
        ↓
Representation learning
        ↓
Autoencoders
        ↓
VAE
        ↓
GAN
        ↓
Attention
        ↓
Transformers
        ↓
Diffusion
        ↓
Your own generative vision model


================================================================================
REAL-WORLD PROBLEM
================================================================================

Imagine a robot camera looking at objects on a workbench.

The robot needs to distinguish:

    horizontal objects
    vertical objects
    diagonal objects

For example:

    class 0 = horizontal bar
    class 1 = vertical bar
    class 2 = diagonal bar

A human immediately recognizes these patterns.

A computer only receives pixel values.

The CNN must learn:

    pixels
       ↓
    local edges
       ↓
    spatial patterns
       ↓
    object-level representation
       ↓
    class prediction


================================================================================
1. CNN ARCHITECTURE
================================================================================

Our simplified CNN will be:

    INPUT
      |
      | 28 x 28 x 1
      v
    CONVOLUTION
      |
      | 3 x 3 x 1 filters
      v
    ReLU
      |
      v
    MAX POOLING
      |
      v
    CONVOLUTION
      |
      v
    ReLU
      |
      v
    MAX POOLING
      |
      v
    FLATTEN
      |
      v
    FULLY CONNECTED
      |
      v
    SOFTMAX
      |
      v
    CLASS


For educational clarity, this implementation uses a small network.


================================================================================
2. CONVOLUTION
================================================================================

For an input X and kernel K:

    Z[i,j]
      =
    Σ_m Σ_n
        X[i+m,j+n] K[m,n]

The kernel moves across the image.

Unlike Lesson 5, the kernel values are now TRAINABLE PARAMETERS.

Initially:

    K = random values

During training:

    K
      ↓
    gradient
      ↓
    updated K

Therefore the CNN discovers useful visual filters.


================================================================================
3. RELU
================================================================================

ReLU:

    f(x) = max(0,x)

Derivative:

    f'(x) =
        1 if x > 0
        0 otherwise

ReLU provides nonlinear modeling capability.

Without nonlinear activation, stacking linear operations would still
produce essentially a linear transformation.


================================================================================
4. MAX POOLING
================================================================================

A 2x2 max-pooling operation:

        a b
        c d

returns:

        max(a,b,c,d)

Example:

        1  4
        2  3

becomes:

        4

Pooling provides:

    lower spatial resolution
    reduced computation
    some local translation tolerance


================================================================================
5. SOFTMAX
================================================================================

The final fully connected layer produces logits:

    z = [z1, z2, ..., zC]

Softmax converts them to probabilities:

                exp(zi)
    p_i = ---------------------
          Σ_j exp(zj)

The probabilities satisfy:

    0 <= p_i <= 1

and:

    Σ_i p_i = 1


================================================================================
6. CROSS-ENTROPY
================================================================================

For one-hot target y:

    L = -Σ_i y_i log(p_i)

If the correct class is:

    class 2

then:

    y = [0, 0, 1]

and:

    L = -log(p_2)

Therefore:

    high probability for correct class
        ↓
    low loss

    low probability for correct class
        ↓
    high loss


================================================================================
7. SOFTMAX + CROSS-ENTROPY GRADIENT
================================================================================

A particularly useful result is:

    dL/dz = p - y

This makes the final layer efficient to train.

The gradient tells the network:

    predicted probability
        -
    desired probability


================================================================================
8. CNN BACKPROPAGATION
================================================================================

The full chain is:

    loss
      ↓
    softmax
      ↓
    fully connected
      ↓
    flatten
      ↓
    max pooling
      ↓
    ReLU
      ↓
    convolution
      ↓
    pooling
      ↓
    ReLU
      ↓
    convolution

Backpropagation applies the chain rule from Lesson 4.

The important new part is:

    convolution gradient

For each kernel weight:

    dL/dK[m,n]

is obtained by accumulating:

    input_patch * upstream_gradient


================================================================================
9. DATASET
================================================================================

To keep this lesson self-contained and reproducible, we generate a synthetic
vision dataset.

Each image is:

    28 x 28 x 1

Three classes:

    CLASS 0:
        horizontal bar

    CLASS 1:
        vertical bar

    CLASS 2:
        diagonal bar

Noise is added to make classification non-trivial.

This is not intended to represent the full complexity of natural images.

The purpose is to expose the complete CNN learning mechanism.

The exact same conceptual pipeline is later used with real camera datasets.


================================================================================
10. IMPLEMENTATION
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1 — ACTIVATION FUNCTIONS
# ==============================================================================

def relu(x):
    """
    ReLU activation.

        ReLU(x) = max(0, x)
    """

    return np.maximum(0.0, x)


def relu_backward(dout, x):
    """
    Backpropagation through ReLU.
    """

    return dout * (x > 0).astype(np.float64)


# ==============================================================================
# SECTION 2 — SOFTMAX
# ==============================================================================

def softmax(logits):
    """
    Numerically stable softmax.

    Subtracting max(logits) prevents exponential overflow.
    """

    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    exp_values = np.exp(shifted)

    return (
        exp_values /
        np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )
    )


# ==============================================================================
# SECTION 3 — CROSS-ENTROPY
# ==============================================================================

def cross_entropy(probabilities, labels):
    """
    Mean cross-entropy loss.

    labels:
        integer class indices

    probabilities:
        N x C
    """

    batch_size = labels.shape[0]

    clipped = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    correct_probabilities = clipped[
        np.arange(batch_size),
        labels
    ]

    loss = -np.mean(
        np.log(correct_probabilities)
    )

    return loss


def softmax_cross_entropy_backward(
    probabilities,
    labels
):
    """
    Gradient of softmax + cross entropy.

        dL/dz = p - y

    divided by batch size.
    """

    batch_size = labels.shape[0]

    dz = probabilities.copy()

    dz[
        np.arange(batch_size),
        labels
    ] -= 1.0

    dz /= batch_size

    return dz


# ==============================================================================
# SECTION 4 — 2D CONVOLUTION FOR ONE IMAGE
# ==============================================================================

def conv2d_single(
    image,
    kernels,
    stride=1,
    padding=1
):
    """
    Multi-filter convolution for a single-channel image.

    Parameters
    ----------
    image:
        H x W

    kernels:
        F x KH x KW

    Returns
    -------
    output:
        H_out x W_out x F
    """

    H, W = image.shape

    number_of_filters, KH, KW = kernels.shape

    padded = np.pad(
        image,
        (
            (padding, padding),
            (padding, padding)
        ),
        mode="constant"
    )

    padded_H, padded_W = padded.shape

    output_H = (
        (padded_H - KH) // stride
    ) + 1

    output_W = (
        (padded_W - KW) // stride
    ) + 1

    output = np.zeros(
        (
            output_H,
            output_W,
            number_of_filters
        ),
        dtype=np.float64
    )

    for i in range(output_H):

        for j in range(output_W):

            row = i * stride
            col = j * stride

            patch = padded[
                row:row + KH,
                col:col + KW
            ]

            for f in range(number_of_filters):

                output[
                    i,
                    j,
                    f
                ] = np.sum(
                    patch * kernels[f]
                )

    return output


# ==============================================================================
# SECTION 5 — CONVOLUTION BACKWARD
# ==============================================================================

def conv2d_backward_single(
    image,
    kernels,
    dout,
    stride=1,
    padding=1
):
    """
    Backpropagation through convolution.

    Returns:

        d_image
        d_kernels

    The implementation follows the direct chain-rule definition.

    For each output:

        output[i,j,f]
            =
        sum(
            patch * kernel[f]
        )

    Therefore:

        d_kernel
            +=
        patch * upstream_gradient

    and:

        d_patch
            +=
        kernel * upstream_gradient
    """

    H, W = image.shape

    number_of_filters, KH, KW = kernels.shape

    padded = np.pad(
        image,
        (
            (padding, padding),
            (padding, padding)
        ),
        mode="constant"
    )

    padded_H, padded_W = padded.shape

    output_H, output_W, _ = dout.shape

    d_padded = np.zeros_like(
        padded
    )

    d_kernels = np.zeros_like(
        kernels
    )

    for i in range(output_H):

        for j in range(output_W):

            row = i * stride
            col = j * stride

            patch = padded[
                row:row + KH,
                col:col + KW
            ]

            for f in range(number_of_filters):

                upstream = dout[
                    i,
                    j,
                    f
                ]

                d_kernels[f] += (
                    patch * upstream
                )

                d_padded[
                    row:row + KH,
                    col:col + KW
                ] += (
                    kernels[f] * upstream
                )

    if padding == 0:

        d_image = d_padded

    else:

        d_image = d_padded[
            padding:-padding,
            padding:-padding
        ]

    return (
        d_image,
        d_kernels
    )


# ==============================================================================
# SECTION 6 — MAX POOLING
# ==============================================================================

def max_pool2d(
    x,
    pool_size=2,
    stride=2
):
    """
    2D max pooling for multiple feature channels.

    x:
        H x W x C

    returns:

        pooled output
        cache for backward pass
    """

    H, W, C = x.shape

    output_H = (
        (H - pool_size) // stride
    ) + 1

    output_W = (
        (W - pool_size) // stride
    ) + 1

    output = np.zeros(
        (
            output_H,
            output_W,
            C
        ),
        dtype=np.float64
    )

    cache = []

    for i in range(output_H):

        row_cache = []

        for j in range(output_W):

            col_cache = []

            row = i * stride
            col = j * stride

            for c in range(C):

                patch = x[
                    row:row + pool_size,
                    col:col + pool_size,
                    c
                ]

                max_index = np.unravel_index(
                    np.argmax(patch),
                    patch.shape
                )

                output[
                    i,
                    j,
                    c
                ] = patch[max_index]

                col_cache.append(
                    max_index
                )

            row_cache.append(
                col_cache
            )

        cache.append(
            row_cache
        )

    cache = (
        cache,
        x.shape,
        pool_size,
        stride
    )

    return output, cache


def max_pool2d_backward(
    dout,
    cache
):
    """
    Backpropagation through max pooling.

    Gradient goes only to the location that produced
    the maximum value during the forward pass.
    """

    (
        max_indices,
        input_shape,
        pool_size,
        stride
    ) = cache

    d_x = np.zeros(
        input_shape,
        dtype=np.float64
    )

    output_H, output_W, C = dout.shape

    for i in range(output_H):

        for j in range(output_W):

            row = i * stride
            col = j * stride

            for c in range(C):

                max_index = max_indices[i][j][c]

                r, cc = max_index

                d_x[
                    row + r,
                    col + cc,
                    c
                ] += dout[
                    i,
                    j,
                    c
                ]

    return d_x


# ==============================================================================
# SECTION 7 — FLATTEN
# ==============================================================================

def flatten_batch(x):
    """
    Convert:

        N x H x W x C

    into:

        N x (H*W*C)
    """

    return x.reshape(
        x.shape[0],
        -1
    )


# ==============================================================================
# SECTION 8 — DATASET GENERATION
# ==============================================================================

def generate_single_image(
    label,
    size=28,
    noise_level=0.15,
    rng=None
):
    """
    Generate one synthetic image.

    Class 0:
        horizontal bar

    Class 1:
        vertical bar

    Class 2:
        diagonal bar

    Small random variations make the task less trivial.
    """

    if rng is None:
        rng = np.random.default_rng()

    image = rng.normal(
        0.0,
        noise_level,
        size=(size, size)
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    center = size // 2

    if label == 0:

        thickness = int(
            rng.integers(2, 5)
        )

        offset = int(
            rng.integers(-2, 3)
        )

        row = center + offset

        image[
            row - thickness:row + thickness + 1,
            5:size - 5
        ] += 0.9

    elif label == 1:

        thickness = int(
            rng.integers(2, 5)
        )

        offset = int(
            rng.integers(-2, 3)
        )

        col = center + offset

        image[
            5:size - 5,
            col - thickness:col + thickness + 1
        ] += 0.9

    elif label == 2:

        thickness = int(
            rng.integers(1, 3)
        )

        offset = int(
            rng.integers(-2, 3)
        )

        for r in range(
            5,
            size - 5
        ):

            c = (
                r
                + offset
            )

            if (
                thickness
                <= c
                < size - thickness
            ):

                image[
                    r - thickness:r + thickness + 1,
                    c - thickness:c + thickness + 1
                ] += 0.9

    else:

        raise ValueError(
            "label must be 0, 1, or 2"
        )

    return np.clip(
        image,
        0.0,
        1.0
    )


def create_dataset(
    samples_per_class=100,
    seed=42
):
    """
    Generate balanced synthetic image dataset.
    """

    rng = np.random.default_rng(seed)

    images = []
    labels = []

    for label in range(3):

        for _ in range(samples_per_class):

            image = generate_single_image(
                label,
                rng=rng
            )

            images.append(
                image
            )

            labels.append(
                label
            )

    images = np.array(
        images,
        dtype=np.float64
    )

    labels = np.array(
        labels,
        dtype=np.int64
    )

    # Shuffle.
    indices = rng.permutation(
        len(images)
    )

    images = images[
        indices
    ]

    labels = labels[
        indices
    ]

    # Add channel dimension:
    #
    # N x H x W
    #
    # becomes:
    #
    # N x H x W x 1

    images = images[
        :, :, :, None
    ]

    return images, labels


def train_test_split(
    X,
    y,
    train_ratio=0.8
):
    """
    Deterministic train/test split.
    """

    split = int(
        len(X) * train_ratio
    )

    return (
        X[:split],
        y[:split],
        X[split:],
        y[split:]
    )


# ==============================================================================
# SECTION 9 — CNN MODEL
# ==============================================================================

class SimpleCNN:
    """
    Small CNN implemented entirely with NumPy.

    Architecture:

        28x28x1
            |
        Conv 3x3, 4 filters
            |
          ReLU
            |
        2x2 MaxPool
            |
        Conv 3x3, 8 filters
            |
          ReLU
            |
        2x2 MaxPool
            |
         Flatten
            |
        Dense -> 3 classes
            |
         Softmax
    """

    def __init__(
        self,
        input_height=28,
        input_width=28,
        seed=123
    ):

        rng = np.random.default_rng(
            seed
        )

        self.num_classes = 3

        # ----------------------------------------------------------------------
        # First convolution
        #
        # Input:
        #
        #     28 x 28 x 1
        #
        # Padding=1 and kernel=3 preserve spatial dimensions:
        #
        #     28 x 28
        #
        # Number of filters:
        #
        #     4
        #
        # Output:
        #
        #     28 x 28 x 4
        # ----------------------------------------------------------------------

        self.K1 = (
            rng.normal(
                0.0,
                np.sqrt(
                    2.0 / 9.0
                ),
                size=(4, 3, 3)
            )
        )

        self.b1 = np.zeros(
            4,
            dtype=np.float64
        )

        # ----------------------------------------------------------------------
        # Second convolution
        #
        # After first pooling:
        #
        #     14 x 14 x 4
        #
        # Second convolution:
        #
        #     14 x 14 x 8
        # ----------------------------------------------------------------------

        self.K2 = (
            rng.normal(
                0.0,
                np.sqrt(
                    2.0 / (3 * 3 * 4)
                ),
                size=(8, 3, 3, 4)
            )
        )

        self.b2 = np.zeros(
            8,
            dtype=np.float64
        )

        # ----------------------------------------------------------------------
        # Determine flattened feature dimension.
        #
        # After second pooling:
        #
        #     7 x 7 x 8
        #
        # therefore:
        #
        #     392 features
        # ----------------------------------------------------------------------

        flattened_size = (
            7 * 7 * 8
        )

        self.W3 = (
            rng.normal(
                0.0,
                np.sqrt(
                    2.0 / flattened_size
                ),
                size=(
                    flattened_size,
                    self.num_classes
                )
            )
        )

        self.b3 = np.zeros(
            self.num_classes,
            dtype=np.float64
        )

        self.cache = None

    # ==========================================================================
    # FORWARD
    # ==========================================================================

    def forward(
        self,
        X
    ):
        """
        Forward pass for an entire batch.

        X:
            N x 28 x 28 x 1

        Returns:

            probabilities
        """

        batch_size = X.shape[0]

        conv1_outputs = []

        # ----------------------------------------------------------------------
        # First convolution
        # ----------------------------------------------------------------------

        for n in range(batch_size):

            single_output = conv2d_single(
                X[n, :, :, 0],
                self.K1,
                stride=1,
                padding=1
            )

            conv1_outputs.append(
                single_output
            )

        Z1 = np.array(
            conv1_outputs
        )

        # Add bias per filter.
        Z1 = Z1 + self.b1[
            None,
            None,
            None,
            :
        ]

        A1 = relu(
            Z1
        )

        P1 = []
        pool_cache_1 = []

        for n in range(batch_size):

            pooled, cache = max_pool2d(
                A1[n]
            )

            P1.append(
                pooled
            )

            pool_cache_1.append(
                cache
            )

        P1 = np.array(
            P1
        )

        # ----------------------------------------------------------------------
        # Second convolution
        #
        # K2:
        #
        #     8 x 3 x 3 x 4
        #
        # Our convolution function operates on a single-channel image,
        # so we implement the multi-channel operation explicitly here.
        # ----------------------------------------------------------------------

        conv2_outputs = []

        for n in range(batch_size):

            feature_input = P1[n]

            H, W, C = feature_input.shape

            num_filters = self.K2.shape[0]

            padded = np.pad(
                feature_input,
                (
                    (1, 1),
                    (1, 1),
                    (0, 0)
                ),
                mode="constant"
            )

            output = np.zeros(
                (
                    H,
                    W,
                    num_filters
                ),
                dtype=np.float64
            )

            for i in range(H):

                for j in range(W):

                    patch = padded[
                        i:i + 3,
                        j:j + 3,
                        :
                    ]

                    for f in range(
                        num_filters
                    ):

                        output[
                            i,
                            j,
                            f
                        ] = np.sum(
                            patch *
                            self.K2[f]
                        )

            conv2_outputs.append(
                output
            )

        Z2 = np.array(
            conv2_outputs
        )

        Z2 = Z2 + self.b2[
            None,
            None,
            None,
            :
        ]

        A2 = relu(
            Z2
        )

        P2 = []
        pool_cache_2 = []

        for n in range(batch_size):

            pooled, cache = max_pool2d(
                A2[n]
            )

            P2.append(
                pooled
            )

            pool_cache_2.append(
                cache
            )

        P2 = np.array(
            P2
        )

        # ----------------------------------------------------------------------
        # Flatten
        # ----------------------------------------------------------------------

        F = flatten_batch(
            P2
        )

        # ----------------------------------------------------------------------
        # Fully connected layer
        # ----------------------------------------------------------------------

        logits = (
            F @ self.W3
            + self.b3
        )

        probabilities = softmax(
            logits
        )

        # ----------------------------------------------------------------------
        # Cache everything needed for backpropagation.
        # ----------------------------------------------------------------------

        self.cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "P1": P1,
            "pool_cache_1": pool_cache_1,
            "Z2": Z2,
            "A2": A2,
            "P2": P2,
            "F": F,
            "logits": logits,
            "probabilities": probabilities,
            "pool_cache_2": pool_cache_2
        }

        return probabilities

    # ==========================================================================
    # BACKWARD
    # ==========================================================================

    def backward(
        self,
        labels
    ):
        """
        Complete CNN backpropagation.

        Returns gradients for all trainable parameters.
        """

        cache = self.cache

        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        P1 = cache["P1"]
        Z2 = cache["Z2"]
        A2 = cache["A2"]
        P2 = cache["P2"]
        F = cache["F"]
        probabilities = cache["probabilities"]

        batch_size = X.shape[0]

        # ----------------------------------------------------------------------
        # Output layer
        # ----------------------------------------------------------------------

        d_logits = (
            softmax_cross_entropy_backward(
                probabilities,
                labels
            )
        )

        dW3 = (
            F.T @ d_logits
        )

        db3 = np.sum(
            d_logits,
            axis=0
        )

        dF = (
            d_logits @ self.W3.T
        )

        # ----------------------------------------------------------------------
        # Unflatten
        # ----------------------------------------------------------------------

        dP2 = dF.reshape(
            P2.shape
        )

        # ----------------------------------------------------------------------
        # Second max pooling backward
        # ----------------------------------------------------------------------

        dA2 = np.zeros_like(
            A2
        )

        for n in range(
            batch_size
        ):

            dA2[n] = (
                max_pool2d_backward(
                    dP2[n],
                    cache[
                        "pool_cache_2"
                    ][n]
                )
            )

        # ----------------------------------------------------------------------
        # ReLU backward
        # ----------------------------------------------------------------------

        dZ2 = relu_backward(
            dA2,
            Z2
        )

        # ----------------------------------------------------------------------
        # Second convolution backward
        # ----------------------------------------------------------------------

        dK2 = np.zeros_like(
            self.K2
        )

        db2 = np.sum(
            dZ2,
            axis=(0, 1, 2)
        )

        dP1 = np.zeros_like(
            P1
        )

        for n in range(
            batch_size
        ):

            feature_input = P1[n]

            H, W, C = feature_input.shape

            padded = np.pad(
                feature_input,
                (
                    (1, 1),
                    (1, 1),
                    (0, 0)
                ),
                mode="constant"
            )

            d_padded = np.zeros_like(
                padded
            )

            for i in range(H):

                for j in range(W):

                    patch = padded[
                        i:i + 3,
                        j:j + 3,
                        :
                    ]

                    for f in range(
                        self.K2.shape[0]
                    ):

                        upstream = dZ2[
                            n,
                            i,
                            j,
                            f
                        ]

                        dK2[f] += (
                            patch *
                            upstream
                        )

                        d_padded[
                            i:i + 3,
                            j:j + 3,
                            :
                        ] += (
                            self.K2[f] *
                            upstream
                        )

            dP1[n] = d_padded[
                1:-1,
                1:-1,
                :
            ]

        # ----------------------------------------------------------------------
        # First pooling backward
        # ----------------------------------------------------------------------

        dA1 = np.zeros_like(
            A1
        )

        for n in range(
            batch_size
        ):

            dA1[n] = (
                max_pool2d_backward(
                    dP1[n],
                    cache[
                        "pool_cache_1"
                    ][n]
                )
            )

        # ----------------------------------------------------------------------
        # First ReLU backward
        # ----------------------------------------------------------------------

        dZ1 = relu_backward(
            dA1,
            Z1
        )

        # ----------------------------------------------------------------------
        # First convolution backward
        # ----------------------------------------------------------------------

        dK1 = np.zeros_like(
            self.K1
        )

        db1 = np.sum(
            dZ1,
            axis=(0, 1, 2)
        )

        for n in range(
            batch_size
        ):

            image = X[
                n,
                :, :, 0
            ]

            _, local_dK = (
                conv2d_backward_single(
                    image,
                    self.K1,
                    dZ1[n],
                    stride=1,
                    padding=1
                )
            )

            dK1 += local_dK

        gradients = {
            "K1": dK1,
            "b1": db1,
            "K2": dK2,
            "b2": db2,
            "W3": dW3,
            "b3": db3
        }

        return gradients

    # ==========================================================================
    # UPDATE
    # ==========================================================================

    def update(
        self,
        gradients,
        learning_rate
    ):
        """
        Gradient descent parameter update.
        """

        self.K1 -= (
            learning_rate *
            gradients["K1"]
        )

        self.b1 -= (
            learning_rate *
            gradients["b1"]
        )

        self.K2 -= (
            learning_rate *
            gradients["K2"]
        )

        self.b2 -= (
            learning_rate *
            gradients["b2"]
        )

        self.W3 -= (
            learning_rate *
            gradients["W3"]
        )

        self.b3 -= (
            learning_rate *
            gradients["b3"]
        )

    # ==========================================================================
    # PREDICTION
    # ==========================================================================

    def predict(
        self,
        X
    ):
        """
        Return predicted class indices.
        """

        probabilities = self.forward(
            X
        )

        return np.argmax(
            probabilities,
            axis=1
        )


# ==============================================================================
# SECTION 10 — ACCURACY
# ==============================================================================

def accuracy(
    predictions,
    labels
):
    """
    Classification accuracy.
    """

    return np.mean(
        predictions == labels
    )


# ==============================================================================
# SECTION 11 — TRAINING
# ==============================================================================

def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=80,
    learning_rate=0.03,
    batch_size=16
):
    """
    Train CNN using mini-batch gradient descent.

    Returns training history.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_accuracy": []
    }

    number_of_samples = (
        len(X_train)
    )

    for epoch in range(
        1,
        epochs + 1
    ):

        # Shuffle training data.
        indices = np.random.permutation(
            number_of_samples
        )

        X_train_shuffled = (
            X_train[indices]
        )

        y_train_shuffled = (
            y_train[indices]
        )

        epoch_losses = []

        # ----------------------------------------------------------------------
        # Mini-batch training
        # ----------------------------------------------------------------------

        for start in range(
            0,
            number_of_samples,
            batch_size
        ):

            end = min(
                start + batch_size,
                number_of_samples
            )

            X_batch = (
                X_train_shuffled[
                    start:end
                ]
            )

            y_batch = (
                y_train_shuffled[
                    start:end
                ]
            )

            probabilities = model.forward(
                X_batch
            )

            loss = cross_entropy(
                probabilities,
                y_batch
            )

            gradients = model.backward(
                y_batch
            )

            model.update(
                gradients,
                learning_rate
            )

            epoch_losses.append(
                loss
            )

        # ----------------------------------------------------------------------
        # Epoch metrics
        # ----------------------------------------------------------------------

        train_probabilities = model.forward(
            X_train
        )

        train_loss = cross_entropy(
            train_probabilities,
            y_train
        )

        train_predictions = np.argmax(
            train_probabilities,
            axis=1
        )

        train_accuracy = accuracy(
            train_predictions,
            y_train
        )

        test_predictions = model.predict(
            X_test
        )

        test_accuracy = accuracy(
            test_predictions,
            y_test
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "test_accuracy"
        ].append(
            test_accuracy
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss={train_loss:.4f} | "
                f"Train Acc={train_accuracy:.3f} | "
                f"Test Acc={test_accuracy:.3f}"
            )

    return history


# ==============================================================================
# SECTION 12 — NUMERICAL GRADIENT CHECK
# ==============================================================================

def numerical_gradient(
    model,
    X,
    y,
    parameter_name,
    index,
    epsilon=1e-5
):
    """
    Numerically estimate one parameter gradient.

        df/dx ≈
        [f(x+epsilon)-f(x-epsilon)] / (2epsilon)

    This is used only for verification.

    It is NOT used for training.
    """

    parameter = getattr(
        model,
        parameter_name
    )

    original_value = parameter[index]

    parameter[index] = (
        original_value + epsilon
    )

    plus_probability = model.forward(
        X
    )

    loss_plus = cross_entropy(
        plus_probability,
        y
    )

    parameter[index] = (
        original_value - epsilon
    )

    minus_probability = model.forward(
        X
    )

    loss_minus = cross_entropy(
        minus_probability,
        y
    )

    parameter[index] = original_value

    # Restore model cache.
    model.forward(X)

    return (
        loss_plus - loss_minus
    ) / (
        2.0 * epsilon
    )


def run_gradient_check():
    """
    Verify the analytical backpropagation implementation.

    We intentionally use a tiny network input and compare analytical
    and numerical gradients for selected parameters.

    This is extremely important when implementing neural networks
    from scratch.

    If backpropagation is wrong, training can appear to work partially
    while silently learning the wrong gradients.
    """

    print("\n" + "=" * 78)
    print("RUNNING CNN GRADIENT CHECK")
    print("=" * 78)

    rng = np.random.default_rng(
        999
    )

    model = SimpleCNN(
        seed=999
    )

    X = rng.random(
        (2, 28, 28, 1)
    )

    y = np.array(
        [0, 2],
        dtype=np.int64
    )

    probabilities = model.forward(
        X
    )

    analytical_gradients = model.backward(
        y
    )

    checks = [
        ("K1", (0, 1, 1)),
        ("K2", (0, 1, 1, 0)),
        ("W3", (10, 1)),
        ("b3", (1,))
    ]

    for parameter_name, index in checks:

        numerical = numerical_gradient(
            model,
            X,
            y,
            parameter_name,
            index
        )

        analytical = analytical_gradients[
            parameter_name
        ][index]

        absolute_error = abs(
            numerical - analytical
        )

        relative_error = (
            absolute_error /
            max(
                1e-8,
                abs(numerical) +
                abs(analytical)
            )
        )

        print(
            f"{parameter_name}{index}: "
            f"analytical={analytical:.8e}, "
            f"numerical={numerical:.8e}, "
            f"relative_error={relative_error:.3e}"
        )

        assert relative_error < 1e-4

    print(
        "PASS: analytical gradients agree "
        "with numerical gradients"
    )

    print("=" * 78)


# ==============================================================================
# SECTION 13 — VISUALIZE DATASET
# ==============================================================================

def visualize_dataset(
    X,
    y
):
    """
    Display examples from each class.
    """

    fig, axes = plt.subplots(
        3,
        5,
        figsize=(10, 7)
    )

    for class_id in range(3):

        indices = np.where(
            y == class_id
        )[0]

        selected = indices[:5]

        for column, index in enumerate(
            selected
        ):

            axes[
                class_id,
                column
            ].imshow(
                X[
                    index,
                    :, :, 0
                ],
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[
                class_id,
                column
            ].axis("off")

            axes[
                class_id,
                column
            ].set_title(
                f"Class {class_id}"
            )

    plt.suptitle(
        "Synthetic Vision Dataset"
    )

    plt.tight_layout()
    plt.show()


# ==============================================================================
# SECTION 14 — TRAINING VISUALIZATION
# ==============================================================================

def visualize_training_history(
    history
):
    """
    Plot loss and accuracy.
    """

    epochs = np.arange(
        1,
        len(
            history["train_loss"]
        ) + 1
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    axes[0].plot(
        epochs,
        history["train_loss"]
    )

    axes[0].set_title(
        "Training Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Cross-Entropy Loss"
    )

    axes[0].grid(
        True
    )

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        label="Train"
    )

    axes[1].plot(
        epochs,
        history["test_accuracy"],
        label="Test"
    )

    axes[1].set_title(
        "Classification Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].set_ylim(
        0.0,
        1.05
    )

    axes[1].legend()

    axes[1].grid(
        True
    )

    plt.tight_layout()
    plt.show()


# ==============================================================================
# SECTION 15 — VISUALIZE LEARNED FILTERS
# ==============================================================================

def visualize_first_layer_filters(
    model
):
    """
    Show filters learned by the first convolution layer.

    These filters are not manually specified.

    They were modified through gradient descent.
    """

    number_of_filters = (
        model.K1.shape[0]
    )

    fig, axes = plt.subplots(
        1,
        number_of_filters,
        figsize=(12, 3)
    )

    for i in range(
        number_of_filters
    ):

        axes[i].imshow(
            model.K1[i],
            cmap="gray"
        )

        axes[i].set_title(
            f"Filter {i}"
        )

        axes[i].axis("off")

    plt.suptitle(
        "Learned First-Layer CNN Filters"
    )

    plt.tight_layout()
    plt.show()


# ==============================================================================
# SECTION 16 — CONFUSION MATRIX
# ==============================================================================

def compute_confusion_matrix(
    y_true,
    y_pred,
    number_of_classes
):
    """
    Compute confusion matrix without external libraries.
    """

    matrix = np.zeros(
        (
            number_of_classes,
            number_of_classes
        ),
        dtype=np.int64
    )

    for true_label, predicted_label in zip(
        y_true,
        y_pred
    ):

        matrix[
            true_label,
            predicted_label
        ] += 1

    return matrix


def visualize_confusion_matrix(
    matrix
):
    """
    Visualize confusion matrix.
    """

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xticks(
        [0, 1, 2]
    )

    ax.set_yticks(
        [0, 1, 2]
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.show()


# ==============================================================================
# SECTION 17 — VISUALIZE PREDICTIONS
# ==============================================================================

def visualize_predictions(
    model,
    X,
    y
):
    """
    Display model predictions on test images.
    """

    probabilities = model.forward(
        X
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(12, 5)
    )

    for i in range(10):

        axes.flat[i].imshow(
            X[
                i,
                :, :, 0
            ],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes.flat[i].set_title(
            f"True={y[i]}  "
            f"Pred={predictions[i]}"
        )

        axes.flat[i].axis(
            "off"
        )

    plt.suptitle(
        "CNN Predictions"
    )

    plt.tight_layout()
    plt.show()


# ==============================================================================
# SECTION 18 — REAL-WORLD CONNECTION
# ==============================================================================

def explain_real_world_application():
    """
    Print the real-world connection.
    """

    print("\n" + "=" * 78)
    print("REAL-WORLD APPLICATION")
    print("=" * 78)

    print(
        """
Imagine a robot sorting objects on a production line.

A camera produces:

    RGB image
        |
        v
    CNN
        |
        +---- edge features
        |
        +---- shape features
        |
        +---- texture features
        |
        v
    object classification
        |
        v
    robot action

For example:

    camera sees object
        |
        v
    CNN predicts:
        "plastic bottle"
        |
        v
    robot selects:
        recycling bin

The CNN does not need a manually programmed rule such as:

    "if these 50 pixels are bright, call it a bottle."

Instead, the convolution filters learn useful visual patterns
from training examples.

This is the fundamental transition from:

    manually designed computer vision

to:

    learned computer vision.
"""
    )

    print("=" * 78)


# ==============================================================================
# SECTION 19 — MAIN
# ==============================================================================

def main():

    print("\n")
    print("=" * 78)
    print("COMPUTER VISION ADVANCED CRASH COURSE")
    print("LESSON 6 — COMPLETE CNN FROM SCRATCH")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # STEP 1 — Gradient verification
    # --------------------------------------------------------------------------

    run_gradient_check()

    # --------------------------------------------------------------------------
    # STEP 2 — Generate dataset
    # --------------------------------------------------------------------------

    print("\nGenerating synthetic image dataset...")

    X, y = create_dataset(
        samples_per_class=100,
        seed=42
    )

    print(
        "Dataset shape:",
        X.shape
    )

    print(
        "Labels shape:",
        y.shape
    )

    assert X.shape == (
        300,
        28,
        28,
        1
    )

    assert y.shape == (
        300,
    )

    assert set(
        np.unique(y)
    ) == {0, 1, 2}

    print(
        "PASS: dataset structure"
    )

    # --------------------------------------------------------------------------
    # STEP 3 — Visualize dataset
    # --------------------------------------------------------------------------

    visualize_dataset(
        X,
        y
    )

    # --------------------------------------------------------------------------
    # STEP 4 — Train/test split
    # --------------------------------------------------------------------------

    X_train, y_train, X_test, y_test = (
        train_test_split(
            X,
            y,
            train_ratio=0.8
        )
    )

    print(
        "\nTrain samples:",
        len(X_train)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    # --------------------------------------------------------------------------
    # STEP 5 — Create CNN
    # --------------------------------------------------------------------------

    print("\nCreating CNN...")

    model = SimpleCNN(
        seed=123
    )

    print(
        "Architecture:"
    )

    print(
        "  Input        : 28 x 28 x 1"
    )

    print(
        "  Conv1        : 4 filters, 3 x 3"
    )

    print(
        "  ReLU"
    )

    print(
        "  MaxPool      : 2 x 2"
    )

    print(
        "  Conv2        : 8 filters, 3 x 3 x 4"
    )

    print(
        "  ReLU"
    )

    print(
        "  MaxPool      : 2 x 2"
    )

    print(
        "  Flatten      : 392"
    )

    print(
        "  Dense        : 3 classes"
    )

    print(
        "  Softmax"
    )

    # --------------------------------------------------------------------------
    # STEP 6 — Forward-pass sanity check
    # --------------------------------------------------------------------------

    probabilities = model.forward(
        X_train[:4]
    )

    print(
        "\nInitial probability shape:",
        probabilities.shape
    )

    assert probabilities.shape == (
        4,
        3
    )

    assert np.allclose(
        np.sum(
            probabilities,
            axis=1
        ),
        1.0
    )

    assert np.all(
        probabilities >= 0
    )

    assert np.all(
        probabilities <= 1
    )

    print(
        "PASS: softmax probability properties"
    )

    # --------------------------------------------------------------------------
    # STEP 7 — Initial loss
    # --------------------------------------------------------------------------

    initial_loss = cross_entropy(
        probabilities,
        y_train[:4]
    )

    print(
        "Initial sample loss:",
        initial_loss
    )

    assert np.isfinite(
        initial_loss
    )

    # --------------------------------------------------------------------------
    # STEP 8 — Train
    # --------------------------------------------------------------------------

    print("\nStarting CNN training...")

    history = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=80,
        learning_rate=0.03,
        batch_size=16
    )

    # --------------------------------------------------------------------------
    # STEP 9 — Final evaluation
    # --------------------------------------------------------------------------

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    train_accuracy = accuracy(
        train_predictions,
        y_train
    )

    test_accuracy = accuracy(
        test_predictions,
        y_test
    )

    final_loss = history[
        "train_loss"
    ][-1]

    print("\n" + "=" * 78)
    print("FINAL MODEL PERFORMANCE")
    print("=" * 78)

    print(
        f"Final training loss : {final_loss:.4f}"
    )

    print(
        f"Training accuracy   : {train_accuracy:.3f}"
    )

    print(
        f"Test accuracy       : {test_accuracy:.3f}"
    )

    print("=" * 78)

    # --------------------------------------------------------------------------
    # STEP 10 — Training verification
    # --------------------------------------------------------------------------

    first_loss = history[
        "train_loss"
    ][0]

    assert np.isfinite(
        final_loss
    )

    assert final_loss < first_loss

    assert train_accuracy >= 0.70

    assert test_accuracy >= 0.60

    print(
        "\nPASS: training loss decreased"
    )

    print(
        "PASS: CNN learned the classification task"
    )

    # --------------------------------------------------------------------------
    # STEP 11 — Confusion matrix
    # --------------------------------------------------------------------------

    confusion = compute_confusion_matrix(
        y_test,
        test_predictions,
        number_of_classes=3
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        confusion
    )

    visualize_confusion_matrix(
        confusion
    )

    # --------------------------------------------------------------------------
    # STEP 12 — Visualize training
    # --------------------------------------------------------------------------

    visualize_training_history(
        history
    )

    # --------------------------------------------------------------------------
    # STEP 13 — Visualize learned filters
    # --------------------------------------------------------------------------

    visualize_first_layer_filters(
        model
    )

    # --------------------------------------------------------------------------
    # STEP 14 — Visualize predictions
    # --------------------------------------------------------------------------

    visualize_predictions(
        model,
        X_test[:10],
        y_test[:10]
    )

    # --------------------------------------------------------------------------
    # STEP 15 — Real-world explanation
    # --------------------------------------------------------------------------

    explain_real_world_application()

    # --------------------------------------------------------------------------
    # STEP 16 — Final model tests
    # --------------------------------------------------------------------------

    assert model.K1.shape == (
        4,
        3,
        3
    )

    assert model.K2.shape == (
        8,
        3,
        3,
        4
    )

    assert model.W3.shape == (
        392,
        3
    )

    assert np.all(
        np.isfinite(model.K1)
    )

    assert np.all(
        np.isfinite(model.K2)
    )

    assert np.all(
        np.isfinite(model.W3)
    )

    print("\n" + "=" * 78)
    print("FINAL LESSON TESTS")
    print("=" * 78)

    print(
        "PASS: model parameter shapes"
    )

    print(
        "PASS: convolution filters are finite"
    )

    print(
        "PASS: dense parameters are finite"
    )

    print(
        "PASS: forward propagation"
    )

    print(
        "PASS: softmax"
    )

    print(
        "PASS: cross-entropy"
    )

    print(
        "PASS: convolution backpropagation"
    )

    print(
        "PASS: pooling backpropagation"
    )

    print(
        "PASS: numerical gradient verification"
    )

    print(
        "PASS: CNN training"
    )

    print(
        "PASS: test-set classification"
    )

    print("=" * 78)

    # --------------------------------------------------------------------------
    # LESSON RESULT
    # --------------------------------------------------------------------------

    print(
        "\nLESSON RESULT"
    )

    print(
        "We have now built a complete trainable CNN from scratch."
    )

    print(
        "\nThe complete learning pipeline is:"
    )

    print(
        "    image"
    )

    print(
        "      ↓"
    )

    print(
        "    convolution"
    )

    print(
        "      ↓"
    )

    print(
        "    ReLU"
    )

    print(
        "      ↓"
    )

    print(
        "    pooling"
    )

    print(
        "      ↓"
    )

    print(
        "    convolution"
    )

    print(
        "      ↓"
    )

    print(
        "    ReLU"
    )

    print(
        "      ↓"
    )

    print(
        "    pooling"
    )

    print(
        "      ↓"
    )

    print(
        "    flatten"
    )

    print(
        "      ↓"
    )

    print(
        "    fully connected layer"
    )

    print(
        "      ↓"
    )

    print(
        "    softmax"
    )

    print(
        "      ↓"
    )

    print(
        "    class prediction"
    )

    print(
        "\nMost importantly:"
    )

    print(
        "The convolution filters were not manually designed."
    )

    print(
        "They were learned through backpropagation and gradient descent."
    )

    print(
        "\nREAL-WORLD RESULT"
    )

    print(
        "A camera image can now be passed through a learned visual "
        "feature extractor and converted into an object-class prediction."
    )

    print(
        "\nCONNECTION TO NEXT LESSON"
    )

    print(
        "Lesson 7 will improve the CNN and examine practical training "
        "issues such as normalization, initialization, receptive fields, "
        "overfitting, validation, and more robust CNN architecture."
    )

    print(
        "\nThe next major architectural milestone after that will be "
        "ResNet-style networks and deeper visual representation learning."
    )


if __name__ == "__main__":
    main()