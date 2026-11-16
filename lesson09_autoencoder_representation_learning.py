"""
Filename: lesson09_autoencoder_representation_learning.py

================================================================================
LESSON 9 — AUTOENCODERS AND REPRESENTATION LEARNING
================================================================================

PURPOSE
-------
In Lessons 1-8, we mainly learned how to analyze images and classify them.

Now we make an important transition:

    DISCRIMINATIVE VISION
        |
        | "What is this image?"
        v
    CLASSIFICATION

toward:

    REPRESENTATION LEARNING
        |
        | "What useful information can be extracted from this image?"
        v
    LATENT REPRESENTATION

This lesson builds an AUTOENCODER from scratch using PyTorch.

An autoencoder learns:

        image
          |
          v
       ENCODER
          |
          v
    latent representation
          |
          v
       DECODER
          |
          v
    reconstructed image

The network is trained without class labels.

This is our first major step toward generative modeling.

IMPORTANT
---------
No pretrained model or pretrained checkpoint is used.

================================================================================
WHAT YOU WILL LEARN
================================================================================

1. What representation learning means.
2. What an encoder does.
3. What a latent vector is.
4. What a decoder does.
5. What an autoencoder is.
6. Reconstruction loss.
7. Bottleneck representations.
8. Undercomplete autoencoders.
9. Why dimensionality reduction can be learned.
10. How latent spaces can be visualized.
11. How similar images can have similar latent representations.
12. How reconstruction quality changes with latent dimension.
13. Why autoencoders are related to generative modeling.
14. Why an ordinary autoencoder is NOT yet a good generative model.

================================================================================
TECHNICAL / COURSE ROADMAP
================================================================================

LESSON 1
    Images as numerical data

LESSON 2
    Vectors, matrices, tensors

LESSON 3
    Gradient descent

LESSON 4
    Neural networks

LESSON 5
    Convolution

LESSON 6
    CNN from scratch

LESSON 7
    Practical PyTorch CNN

LESSON 8
    Deep CNN + residual learning

LESSON 9  <-- CURRENT
    Autoencoder + representation learning

LESSON 10
    Variational Autoencoder (VAE)

LESSON 11
    GAN fundamentals

LESSON 12+
    Attention
    Transformers
    Generative modeling
    Diffusion
    Latent diffusion
    Text conditioning
    Cross-attention
    Text-to-image generation

================================================================================
REAL-WORLD PROBLEM
================================================================================

Imagine a robot camera observing thousands of images.

Suppose each image is:

    64 x 64 x 3

That means:

    12,288 pixel values

But many pixels may contain redundant information.

A useful representation might contain only:

    32
    64
    128

latent values.

For example:

        camera image
             |
             v
       CNN encoder
             |
             v
       64-dimensional
       latent vector
             |
             +------> classification
             |
             +------> anomaly detection
             |
             +------> retrieval
             |
             +------> downstream policy
             |
             +------> generation

This is extremely useful in robotics.

A robot does not necessarily need to pass the complete raw image to
every downstream component.

Instead:

        RGB image
            |
            v
       visual encoder
            |
            v
       compact representation
            |
            v
       manipulation system

================================================================================
THEORY
================================================================================

An autoencoder consists of two functions.

ENCODER:

    z = f_theta(x)

DECODER:

    x_hat = g_phi(z)

Therefore:

    x_hat = g_phi(f_theta(x))

where:

    x       = input image
    z       = latent representation
    x_hat   = reconstructed image

The model tries to make:

    x_hat ≈ x

Therefore training minimizes reconstruction error.

--------------------------------------------------------------------------------
BOTTLENECK
--------------------------------------------------------------------------------

Suppose the input contains:

    784 values

and the latent representation contains:

    32 values

Then:

    784 -> 32 -> 784

The network cannot simply copy every input value independently.

It must learn a compressed representation.

This creates the bottleneck.

--------------------------------------------------------------------------------
UNDERCOMPLETE AUTOENCODER
--------------------------------------------------------------------------------

An undercomplete autoencoder satisfies:

    dim(z) < dim(x)

Example:

    x ∈ R^784

    z ∈ R^32

The network must discover a useful compressed representation.

--------------------------------------------------------------------------------
RECONSTRUCTION
--------------------------------------------------------------------------------

The decoder produces:

    x_hat

The reconstruction error measures how different:

    x

and:

    x_hat

are.

For normalized pixel values:

    x ∈ [0,1]

a common loss is mean squared error:

    MSE = (1/N) Σ (x_i - x_hat_i)^2

For image reconstruction, another possibility is binary cross entropy,
especially when the pixels are interpreted as probabilities.

In this lesson we use MSE because it provides a simple and transparent
reconstruction objective.

--------------------------------------------------------------------------------
WHY NO LABELS?
--------------------------------------------------------------------------------

Classification requires:

    image -> class label

Autoencoder training only requires:

    image -> same image

Therefore the target is:

    target = input

This is an example of self-supervised / unsupervised-style representation
learning.

--------------------------------------------------------------------------------
LATENT SPACE
--------------------------------------------------------------------------------

Suppose:

    z ∈ R^2

Then every image becomes a point:

        z2
         ^
         |
         |      x x
         |    x
         |              o o
         |            o
         +--------------------> z1

Images with similar visual structure may become close together.

This creates a learned latent space.

This idea becomes much more powerful with a VAE.

================================================================================
MATHEMATICS
================================================================================

Encoder:

    z = f_theta(x)

Decoder:

    x_hat = g_phi(z)

Complete autoencoder:

    x_hat = g_phi(f_theta(x))

Reconstruction loss:

    L(theta, phi)
        =
    (1/N) Σ ||x_i - x_hat_i||²

Training:

    theta <- theta - alpha * dL/dtheta

    phi <- phi - alpha * dL/dphi

where:

    theta = encoder parameters
    phi   = decoder parameters
    alpha = learning rate

--------------------------------------------------------------------------------
LATENT DISTANCE
--------------------------------------------------------------------------------

Given two images:

    x_a
    x_b

their latent representations are:

    z_a = f(x_a)
    z_b = f(x_b)

Euclidean latent distance:

    d(z_a,z_b)
        =
    sqrt(Σ(z_a_i-z_b_i)²)

If the learned representation is useful, visually related samples often
have smaller latent distances than unrelated samples.

This is not guaranteed mathematically for every autoencoder.

It is something we investigate experimentally.

================================================================================
IMPLEMENTATION
================================================================================

Input:

    32 x 32 grayscale image

Flattened input:

    1024 values

Encoder:

    1024
      |
    512
      |
    128
      |
    latent_dim

Decoder:

    latent_dim
      |
    128
      |
    512
      |
    1024

The decoder uses sigmoid at the output because the image values are
normalized to [0,1].

Architecture:

    Image
      |
    Flatten
      |
    Linear
      |
    ReLU
      |
    Linear
      |
    ReLU
      |
    Latent vector
      |
    Linear
      |
    ReLU
      |
    Linear
      |
    ReLU
      |
    Linear
      |
    Sigmoid
      |
    Reconstructed image

================================================================================
EXPERIMENTS
================================================================================

We perform three important experiments.

EXPERIMENT 1
------------
Train an autoencoder with:

    latent_dim = 16

EXPERIMENT 2
------------
Train another autoencoder with:

    latent_dim = 64

We compare reconstruction quality.

EXPECTED BEHAVIOR
-----------------
A larger latent representation generally has more capacity.

Therefore:

    latent_dim = 64

should normally reconstruct the images better than:

    latent_dim = 16

for this task.

EXPERIMENT 3
------------
Analyze latent representations.

We visualize:

    - latent coordinates
    - nearest neighbors
    - reconstruction examples

================================================================================
REAL-WORLD APPLICATION
================================================================================

Possible real-world applications include:

1. IMAGE COMPRESSION

    camera image
        |
        v
      encoder
        |
        v
    latent vector
        |
        v
      decoder
        |
        v
    reconstructed image

2. ANOMALY DETECTION

Train mostly on normal images.

At inference:

    normal image
        -> low reconstruction error

    abnormal image
        -> high reconstruction error

This can be used for industrial inspection.

3. ROBOT VISION

    camera
      |
      v
    encoder
      |
      v
    compact visual representation
      |
      +----> manipulation policy
      |
      +----> object classifier
      |
      +----> anomaly detector

4. DATASET ANALYSIS

Latent vectors can be used to:

    - cluster images
    - search similar images
    - visualize datasets
    - detect unusual samples

================================================================================
IMPORTANT LIMITATION
================================================================================

A normal autoencoder is NOT automatically a good image generator.

Why?

Because the encoder only learns latent vectors for the training data.

The latent space may contain:

    holes
    irregular regions
    disconnected regions
    poorly structured areas

Suppose we randomly sample:

    z_random

There is no guarantee that:

    decoder(z_random)

will produce a realistic image.

This is the major motivation for the next step:

    VARIATIONAL AUTOENCODER

A VAE explicitly regularizes the latent space.

That creates a much more useful continuous probabilistic latent space.

================================================================================
LESSON RESULT
================================================================================

After this lesson you should understand:

    [x] encoder
    [x] decoder
    [x] latent vector
    [x] bottleneck
    [x] reconstruction loss
    [x] dimensionality reduction
    [x] representation learning
    [x] latent-space visualization
    [x] reconstruction-based anomaly detection
    [x] limitations of ordinary autoencoders

The conceptual transition is:

    CNN classification
          |
          v
    visual representation
          |
          v
    latent space
          |
          v
    generative modeling

================================================================================
NEXT LESSON
================================================================================

LESSON 10:

    VARIATIONAL AUTOENCODER

We will change:

    deterministic latent vector

into:

    probability distribution

Instead of:

    z = encoder(x)

we learn:

    mu(x)
    log_variance(x)

and sample:

    z = mu + sigma * epsilon

with:

    epsilon ~ N(0, I)

This is the reparameterization trick.

That will be our first serious introduction to probabilistic generative
modeling.

================================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

import random
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 32
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE

TRAIN_SAMPLES = 2400
VAL_SAMPLES = 600
TEST_SAMPLES = 600

NUM_CLASSES = 4

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 1e-3

LATENT_DIM_SMALL = 16
LATENT_DIM_LARGE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    """
    Make the experiment reproducible as far as practical.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# ENVIRONMENT
# =============================================================================

def print_environment():
    """
    Display software and hardware information.
    """

    print("=" * 80)
    print("LESSON 9 — AUTOENCODER REPRESENTATION LEARNING")
    print("=" * 80)

    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 80)


# =============================================================================
# SYNTHETIC IMAGE GENERATION
# =============================================================================

def create_shape_image(
    label,
    size=IMAGE_SIZE,
    noise_level=0.08
):
    """
    Generate one synthetic image.

    Classes:

        0 = horizontal bar
        1 = vertical bar
        2 = diagonal
        3 = cross

    We intentionally use multiple visual structures.

    The autoencoder does NOT receive the label.

    The label is used only for analysis and visualization.
    """

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    thickness = np.random.randint(2, 5)

    if label == 0:

        center = np.random.randint(
            size // 4,
            3 * size // 4
        )

        y0 = max(
            0,
            center - thickness
        )

        y1 = min(
            size,
            center + thickness
        )

        image[y0:y1, :] = 1.0

    elif label == 1:

        center = np.random.randint(
            size // 4,
            3 * size // 4
        )

        x0 = max(
            0,
            center - thickness
        )

        x1 = min(
            size,
            center + thickness
        )

        image[:, x0:x1] = 1.0

    elif label == 2:

        for i in range(size):

            for offset in range(
                -thickness,
                thickness + 1
            ):

                j = i + offset

                if 0 <= j < size:
                    image[i, j] = 1.0

    elif label == 3:

        center_y = size // 2
        center_x = size // 2

        image[
            center_y - thickness:
            center_y + thickness,
            :
        ] = 1.0

        image[
            :,
            center_x - thickness:
            center_x + thickness
        ] = 1.0

    else:
        raise ValueError(
            "Unknown image class."
        )

    # -------------------------------------------------------------------------
    # RANDOM TRANSLATION
    # -------------------------------------------------------------------------

    shift_y = np.random.randint(
        -3,
        4
    )

    shift_x = np.random.randint(
        -3,
        4
    )

    translated = np.zeros_like(image)

    src_y0 = max(
        0,
        -shift_y
    )

    src_y1 = min(
        size,
        size - shift_y
    )

    src_x0 = max(
        0,
        -shift_x
    )

    src_x1 = min(
        size,
        size - shift_x
    )

    dst_y0 = max(
        0,
        shift_y
    )

    dst_y1 = dst_y0 + (
        src_y1 - src_y0
    )

    dst_x0 = max(
        0,
        shift_x
    )

    dst_x1 = dst_x0 + (
        src_x1 - src_x0
    )

    translated[
        dst_y0:dst_y1,
        dst_x0:dst_x1
    ] = image[
        src_y0:src_y1,
        src_x0:src_x1
    ]

    image = translated

    # -------------------------------------------------------------------------
    # GAUSSIAN NOISE
    # -------------------------------------------------------------------------

    noise = np.random.normal(
        0.0,
        noise_level,
        image.shape
    ).astype(np.float32)

    image = image + noise

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image.astype(
        np.float32
    )


def create_dataset(
    num_samples
):
    """
    Generate a complete dataset.

    Returns:

        images: [N, H, W]
        labels: [N]

    Labels are NOT used during autoencoder training.
    They are retained for analysis.
    """

    images = []
    labels = []

    for index in range(
        num_samples
    ):

        label = index % NUM_CLASSES

        image = create_shape_image(
            label
        )

        images.append(image)
        labels.append(label)

    images = np.stack(
        images
    )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    indices = np.random.permutation(
        len(images)
    )

    images = images[indices]
    labels = labels[indices]

    return images, labels


# =============================================================================
# DATASET
# =============================================================================

class ImageDataset(Dataset):
    """
    Dataset for autoencoder training.

    The important point:

        input = target

    No class label is required for reconstruction training.
    """

    def __init__(
        self,
        images,
        labels=None
    ):

        self.images = torch.from_numpy(
            images
        ).float()

        self.labels = None

        if labels is not None:
            self.labels = torch.from_numpy(
                labels
            ).long()

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        index
    ):

        image = self.images[index]

        if self.labels is None:
            return image

        return (
            image,
            self.labels[index]
        )


# =============================================================================
# AUTOENCODER
# =============================================================================

class Autoencoder(nn.Module):
    """
    Fully-connected autoencoder.

    Input:

        [B, 32, 32]

    Flatten:

        [B, 1024]

    Encoder:

        1024 -> 512 -> 128 -> latent_dim

    Decoder:

        latent_dim -> 128 -> 512 -> 1024

    Output:

        [B, 32, 32]
    """

    def __init__(
        self,
        latent_dim
    ):

        super().__init__()

        # ---------------------------------------------------------------------
        # ENCODER
        # ---------------------------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                INPUT_DIM,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                latent_dim
            )
        )

        # ---------------------------------------------------------------------
        # DECODER
        # ---------------------------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                INPUT_DIM
            ),

            nn.Sigmoid()
        )

        self.latent_dim = latent_dim

    def encode(
        self,
        x
    ):
        """
        Convert image into latent vector.
        """

        return self.encoder(x)

    def decode(
        self,
        z
    ):
        """
        Convert latent vector into reconstructed image.
        """

        x_hat = self.decoder(z)

        return x_hat.view(
            -1,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

    def forward(
        self,
        x
    ):

        z = self.encode(x)

        x_hat = self.decode(z)

        return x_hat, z


# =============================================================================
# MODEL INFORMATION
# =============================================================================

def count_parameters(
    model
):
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# =============================================================================
# DATASET VISUALIZATION
# =============================================================================

def visualize_dataset(
    images,
    labels
):
    """
    Display examples from each class.
    """

    fig, axes = plt.subplots(
        1,
        NUM_CLASSES,
        figsize=(10, 3)
    )

    for class_id in range(
        NUM_CLASSES
    ):

        index = np.where(
            labels == class_id
        )[0][0]

        axes[class_id].imshow(
            images[index],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[class_id].set_title(
            f"Class {class_id}"
        )

        axes[class_id].axis(
            "off"
        )

    plt.suptitle(
        "Training Images"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):
    """
    Train one autoencoder epoch.

    Target is the input image itself.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for images in loader:

        images = images.to(
            DEVICE
        )

        optimizer.zero_grad()

        reconstructed, _ = model(
            images
        )

        loss = criterion(
            reconstructed,
            images
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):
    """
    Evaluate reconstruction loss.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for images in loader:

        images = images.to(
            DEVICE
        )

        reconstructed, _ = model(
            images
        )

        loss = criterion(
            reconstructed,
            images
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


# =============================================================================
# COMPLETE TRAINING
# =============================================================================

def train_autoencoder(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE
):
    """
    Train an autoencoder.

    Returns:

        history
        best_model_state
    """

    model = model.to(
        DEVICE
    )

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float(
        "inf"
    )

    best_state = None

    for epoch in range(
        1,
        epochs + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train MSE={train_loss:.6f} | "
            f"Val MSE={val_loss:.6f}"
        )

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return (
        history,
        best_state
    )


# =============================================================================
# RECONSTRUCTION VISUALIZATION
# =============================================================================

@torch.no_grad()
def visualize_reconstructions(
    model,
    dataset,
    title,
    num_examples=8
):
    """
    Display original and reconstructed images.

    Top row:
        original

    Bottom row:
        reconstruction
    """

    model.eval()

    indices = np.random.choice(
        len(dataset),
        size=num_examples,
        replace=False
    )

    fig, axes = plt.subplots(
        2,
        num_examples,
        figsize=(16, 4)
    )

    for column, index in enumerate(
        indices
    ):

        image = dataset[index]

        x = image.unsqueeze(
            0
        ).to(DEVICE)

        reconstructed, _ = model(
            x
        )

        reconstructed = (
            reconstructed[0]
            .cpu()
            .numpy()
        )

        axes[0, column].imshow(
            image.numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0, column].axis(
            "off"
        )

        axes[1, column].imshow(
            reconstructed,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1, column].axis(
            "off"
        )

    axes[0, 0].set_ylabel(
        "Original"
    )

    axes[1, 0].set_ylabel(
        "Reconstructed"
    )

    plt.suptitle(
        title
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# LOSS CURVES
# =============================================================================

def plot_loss_comparison(
    history_small,
    history_large
):
    """
    Compare reconstruction loss for two latent dimensions.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history_small["train_loss"],
        label=f"Train latent={LATENT_DIM_SMALL}"
    )

    plt.plot(
        history_small["val_loss"],
        linestyle="--",
        label=f"Val latent={LATENT_DIM_SMALL}"
    )

    plt.plot(
        history_large["train_loss"],
        label=f"Train latent={LATENT_DIM_LARGE}"
    )

    plt.plot(
        history_large["val_loss"],
        linestyle="--",
        label=f"Val latent={LATENT_DIM_LARGE}"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "MSE Reconstruction Loss"
    )

    plt.title(
        "Autoencoder Reconstruction Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# LATENT REPRESENTATION EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_latents(
    model,
    images,
    labels
):
    """
    Encode a dataset into latent vectors.
    """

    model.eval()

    x = torch.from_numpy(
        images
    ).float().to(DEVICE)

    z = model.encode(
        x
    )

    return (
        z.cpu().numpy(),
        labels
    )


# =============================================================================
# LATENT SPACE VISUALIZATION
# =============================================================================

def visualize_latent_space(
    latent_vectors,
    labels,
    title
):
    """
    Visualize latent space.

    If latent dimension > 2, we simply display the first two coordinates.

    IMPORTANT:
    This is only a visualization, not a complete dimensionality-reduction
    method such as PCA or t-SNE.
    """

    if latent_vectors.shape[1] < 2:

        raise ValueError(
            "Latent dimension must be at least 2."
        )

    plt.figure(
        figsize=(7, 6)
    )

    for class_id in range(
        NUM_CLASSES
    ):

        mask = (
            labels == class_id
        )

        plt.scatter(
            latent_vectors[
                mask,
                0
            ],
            latent_vectors[
                mask,
                1
            ],
            s=12,
            alpha=0.6,
            label=f"Class {class_id}"
        )

    plt.xlabel(
        "Latent dimension 1"
    )

    plt.ylabel(
        "Latent dimension 2"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# LATENT DISTANCE EXPERIMENT
# =============================================================================

def latent_distance(
    z1,
    z2
):
    """
    Euclidean distance between latent vectors.
    """

    return float(
        np.linalg.norm(
            z1 - z2
        )
    )


def run_latent_distance_experiment(
    model,
    images,
    labels
):
    """
    Compare latent distances for same-class and different-class images.

    We do not claim that this will always separate classes perfectly.

    The experiment demonstrates how the learned representation can encode
    visual similarity.
    """

    model.eval()

    latents, labels = extract_latents(
        model,
        images,
        labels
    )

    same_class_distances = []
    different_class_distances = []

    for i in range(
        min(200, len(latents))
    ):

        for j in range(
            i + 1,
            min(200, len(latents))
        ):

            distance = latent_distance(
                latents[i],
                latents[j]
            )

            if labels[i] == labels[j]:

                same_class_distances.append(
                    distance
                )

            else:

                different_class_distances.append(
                    distance
                )

    same_mean = np.mean(
        same_class_distances
    )

    different_mean = np.mean(
        different_class_distances
    )

    print()
    print("=" * 80)
    print("LATENT DISTANCE EXPERIMENT")
    print("=" * 80)

    print(
        f"Mean same-class latent distance      : "
        f"{same_mean:.4f}"
    )

    print(
        f"Mean different-class latent distance : "
        f"{different_mean:.4f}"
    )

    return (
        same_mean,
        different_mean
    )


# =============================================================================
# RECONSTRUCTION ERROR EXPERIMENT
# =============================================================================

@torch.no_grad()
def compute_reconstruction_errors(
    model,
    images
):
    """
    Compute per-image reconstruction MSE.
    """

    model.eval()

    x = torch.from_numpy(
        images
    ).float().to(DEVICE)

    reconstructed, _ = model(
        x
    )

    errors = (
        (x - reconstructed) ** 2
    ).mean(
        dim=(1, 2)
    )

    return errors.cpu().numpy()


def plot_reconstruction_error_distribution(
    errors,
    labels
):
    """
    Plot reconstruction error distribution by class.

    This illustrates the basic idea behind reconstruction-based anomaly
    detection.
    """

    plt.figure(
        figsize=(8, 5)
    )

    for class_id in range(
        NUM_CLASSES
    ):

        mask = (
            labels == class_id
        )

        plt.hist(
            errors[mask],
            bins=25,
            alpha=0.5,
            label=f"Class {class_id}"
        )

    plt.xlabel(
        "Reconstruction MSE"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Reconstruction Error Distribution"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# RANDOM LATENT DECODING
# =============================================================================

@torch.no_grad()
def visualize_random_latent_decoding(
    model,
    num_examples=8
):
    """
    Sample random latent vectors and decode them.

    IMPORTANT:

    For an ordinary autoencoder, random latent vectors are not guaranteed
    to produce meaningful images.

    This experiment demonstrates that limitation.

    It prepares the conceptual motivation for VAE.
    """

    model.eval()

    z = torch.randn(
        num_examples,
        model.latent_dim
    ).to(DEVICE)

    generated = model.decode(
        z
    )

    generated = (
        generated
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(
        1,
        num_examples,
        figsize=(16, 2)
    )

    for index in range(
        num_examples
    ):

        axes[index].imshow(
            generated[index],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[index].axis(
            "off"
        )

    plt.suptitle(
        "Decoding Random Latent Vectors — Ordinary Autoencoder"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# TESTS
# =============================================================================

def test_autoencoder_shapes():
    """
    Verify encoder/decoder tensor shapes.
    """

    model = Autoencoder(
        latent_dim=16
    ).to(DEVICE)

    x = torch.rand(
        4,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    with torch.no_grad():

        reconstructed, latent = model(
            x
        )

    assert reconstructed.shape == (
        4,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert latent.shape == (
        4,
        16
    )

    print(
        "[PASS] Autoencoder shape test"
    )


def test_output_range():
    """
    Verify decoder output remains in [0,1].
    """

    model = Autoencoder(
        latent_dim=16
    ).to(DEVICE)

    x = torch.rand(
        4,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    with torch.no_grad():

        reconstructed, _ = model(
            x
        )

    assert torch.all(
        reconstructed >= 0.0
    )

    assert torch.all(
        reconstructed <= 1.0
    )

    print(
        "[PASS] Decoder output-range test"
    )


def test_gradient_flow():
    """
    Verify that reconstruction loss creates gradients in the model.
    """

    model = Autoencoder(
        latent_dim=16
    ).to(DEVICE)

    x = torch.rand(
        4,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    reconstructed, _ = model(
        x
    )

    loss = nn.MSELoss()(
        reconstructed,
        x
    )

    loss.backward()

    gradient_count = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            assert parameter.grad is not None

            assert torch.isfinite(
                parameter.grad
            ).all()

            gradient_count += 1

    assert gradient_count > 0

    print(
        "[PASS] Autoencoder gradient test"
    )


def test_reconstruction_loss_is_valid():
    """
    Verify MSE reconstruction loss is finite and non-negative.
    """

    model = Autoencoder(
        latent_dim=16
    ).to(DEVICE)

    x = torch.rand(
        4,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    reconstructed, _ = model(
        x
    )

    loss = nn.MSELoss()(
        reconstructed,
        x
    )

    assert torch.isfinite(
        loss
    )

    assert loss.item() >= 0.0

    print(
        "[PASS] Reconstruction-loss test"
    )


# =============================================================================
# REAL-WORLD APPLICATION
# =============================================================================

def real_world_application(
    test_error_small,
    test_error_large
):
    """
    Connect the lesson to practical computer vision.
    """

    print()
    print("=" * 80)
    print("REAL-WORLD APPLICATION")
    print("=" * 80)

    print(
        """
The autoencoder architecture can be used in a real vision system as:

        Camera
          |
          v
      RGB image
          |
          v
      Encoder
          |
          v
    latent representation
          |
          +----------------+
          |                |
          v                v
    downstream AI      Decoder
                           |
                           v
                     reconstruction

Possible applications:

    1. Industrial anomaly detection

       Train on normal products.

       Normal:
           reconstruction error -> low

       Defective:
           reconstruction error -> potentially high

    2. Robot visual representation

       RGB camera
           ->
       encoder
           ->
       compact latent state
           ->
       manipulation/control model

    3. Image retrieval

       image
         ->
       latent vector
         ->
       nearest-neighbor search

    4. Dataset visualization

       image
         ->
       latent vector
         ->
       2D visualization

    5. Compression

       image
         ->
       compact latent
         ->
       reconstruction

The exact architecture used in a production system would normally use
convolutional encoders/decoders rather than this simple fully-connected
demonstration.
"""
    )

    print(
        f"Test MSE, latent={LATENT_DIM_SMALL}: "
        f"{test_error_small:.6f}"
    )

    print(
        f"Test MSE, latent={LATENT_DIM_LARGE}: "
        f"{test_error_large:.6f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    print_environment()

    # =========================================================================
    # STEP 1 — ARCHITECTURE TESTS
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 1 — AUTOENCODER UNIT TESTS")
    print("=" * 80)

    test_autoencoder_shapes()

    test_output_range()

    test_gradient_flow()

    test_reconstruction_loss_is_valid()

    # =========================================================================
    # STEP 2 — DATASET
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 2 — CREATE DATASET")
    print("=" * 80)

    train_images, train_labels = create_dataset(
        TRAIN_SAMPLES
    )

    val_images, val_labels = create_dataset(
        VAL_SAMPLES
    )

    test_images, test_labels = create_dataset(
        TEST_SAMPLES
    )

    print(
        f"Train images: {train_images.shape}"
    )

    print(
        f"Val images  : {val_images.shape}"
    )

    print(
        f"Test images : {test_images.shape}"
    )

    visualize_dataset(
        train_images,
        train_labels
    )

    # =========================================================================
    # STEP 3 — DATA LOADERS
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 3 — DATA LOADERS")
    print("=" * 80)

    train_dataset = ImageDataset(
        train_images
    )

    val_dataset = ImageDataset(
        val_images
    )

    test_dataset = ImageDataset(
        test_images
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # =========================================================================
    # STEP 4 — SMALL LATENT MODEL
    # =========================================================================

    print()
    print("=" * 80)
    print(
        f"STEP 4 — TRAIN AUTOENCODER "
        f"(LATENT={LATENT_DIM_SMALL})"
    )
    print("=" * 80)

    small_model = Autoencoder(
        latent_dim=LATENT_DIM_SMALL
    )

    print(
        f"Parameters: "
        f"{count_parameters(small_model):,}"
    )

    small_history, small_state = train_autoencoder(
        small_model,
        train_loader,
        val_loader
    )

    # =========================================================================
    # STEP 5 — LARGE LATENT MODEL
    # =========================================================================

    print()
    print("=" * 80)
    print(
        f"STEP 5 — TRAIN AUTOENCODER "
        f"(LATENT={LATENT_DIM_LARGE})"
    )
    print("=" * 80)

    large_model = Autoencoder(
        latent_dim=LATENT_DIM_LARGE
    )

    print(
        f"Parameters: "
        f"{count_parameters(large_model):,}"
    )

    large_history, large_state = train_autoencoder(
        large_model,
        train_loader,
        val_loader
    )

    # =========================================================================
    # STEP 6 — TEST MODELS
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 6 — TEST RECONSTRUCTION")
    print("=" * 80)

    criterion = nn.MSELoss()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    small_test_loss = evaluate(
        small_model,
        test_loader,
        criterion
    )

    large_test_loss = evaluate(
        large_model,
        test_loader,
        criterion
    )

    print(
        f"Small latent test MSE "
        f"({LATENT_DIM_SMALL}) = "
        f"{small_test_loss:.6f}"
    )

    print(
        f"Large latent test MSE "
        f"({LATENT_DIM_LARGE}) = "
        f"{large_test_loss:.6f}"
    )

    # =========================================================================
    # STEP 7 — RECONSTRUCTION VISUALIZATION
    # =========================================================================

    visualize_reconstructions(
        small_model,
        test_dataset,
        f"Reconstruction — latent={LATENT_DIM_SMALL}"
    )

    visualize_reconstructions(
        large_model,
        test_dataset,
        f"Reconstruction — latent={LATENT_DIM_LARGE}"
    )

    # =========================================================================
    # STEP 8 — LOSS COMPARISON
    # =========================================================================

    plot_loss_comparison(
        small_history,
        large_history
    )

    # =========================================================================
    # STEP 9 — LATENT SPACE
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 9 — LATENT SPACE ANALYSIS")
    print("=" * 80)

    small_latents, small_labels = extract_latents(
        small_model,
        test_images,
        test_labels
    )

    large_latents, large_labels = extract_latents(
        large_model,
        test_images,
        test_labels
    )

    print(
        f"Small latent shape: "
        f"{small_latents.shape}"
    )

    print(
        f"Large latent shape: "
        f"{large_latents.shape}"
    )

    visualize_latent_space(
        small_latents,
        small_labels,
        f"Latent Space — latent={LATENT_DIM_SMALL}"
    )

    visualize_latent_space(
        large_latents,
        large_labels,
        f"Latent Space — latent={LATENT_DIM_LARGE}"
    )

    # =========================================================================
    # STEP 10 — LATENT DISTANCE
    # =========================================================================

    small_same_distance, \
        small_different_distance = \
        run_latent_distance_experiment(
            small_model,
            test_images,
            test_labels
        )

    large_same_distance, \
        large_different_distance = \
        run_latent_distance_experiment(
            large_model,
            test_images,
            test_labels
        )

    # =========================================================================
    # STEP 11 — RECONSTRUCTION ERROR
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 11 — RECONSTRUCTION ERROR")
    print("=" * 80)

    reconstruction_errors = (
        compute_reconstruction_errors(
            small_model,
            test_images
        )
    )

    print(
        f"Mean reconstruction error: "
        f"{reconstruction_errors.mean():.6f}"
    )

    print(
        f"Maximum reconstruction error: "
        f"{reconstruction_errors.max():.6f}"
    )

    plot_reconstruction_error_distribution(
        reconstruction_errors,
        test_labels
    )

    # =========================================================================
    # STEP 12 — RANDOM LATENT EXPERIMENT
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 12 — RANDOM LATENT VECTOR EXPERIMENT")
    print("=" * 80)

    print(
        """
We now sample:

    z ~ N(0, I)

and ask the ordinary autoencoder decoder to generate images.

There is no guarantee these samples will look meaningful.

This is intentional.

The experiment demonstrates the limitation of an ordinary autoencoder
and motivates the Variational Autoencoder in Lesson 10.
"""
    )

    visualize_random_latent_decoding(
        small_model
    )

    # =========================================================================
    # STEP 13 — REAL-WORLD APPLICATION
    # =========================================================================

    real_world_application(
        small_test_loss,
        large_test_loss
    )

    # =========================================================================
    # STEP 14 — FINAL VERIFICATION
    # =========================================================================

    print()
    print("=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)

    assert np.isfinite(
        small_test_loss
    )

    assert np.isfinite(
        large_test_loss
    )

    assert small_test_loss >= 0.0

    assert large_test_loss >= 0.0

    assert len(
        small_history["train_loss"]
    ) == EPOCHS

    assert len(
        large_history["train_loss"]
    ) == EPOCHS

    # Training should substantially reduce reconstruction error.
    assert (
        small_history["train_loss"][-1]
        <
        small_history["train_loss"][0]
    ), (
        "Small latent autoencoder did not reduce training loss."
    )

    assert (
        large_history["train_loss"][-1]
        <
        large_history["train_loss"][0]
    ), (
        "Large latent autoencoder did not reduce training loss."
    )

    assert (
        np.isfinite(
            small_same_distance
        )
        and
        np.isfinite(
            small_different_distance
        )
    )

    assert (
        np.isfinite(
            large_same_distance
        )
        and
        np.isfinite(
            large_different_distance
        )
    )

    print(
        "[PASS] Finite reconstruction-loss verification"
    )

    print(
        "[PASS] Training-loss reduction verification"
    )

    print(
        "[PASS] Latent-distance verification"
    )

    # =========================================================================
    # LESSON RESULT
    # =========================================================================

    print()
    print("=" * 80)
    print("LESSON RESULT")
    print("=" * 80)

    print(
        """
You have now built a complete autoencoder.

The key transformation is:

        IMAGE
          |
          v
       ENCODER
          |
          v
       LATENT
      VECTOR z
          |
          v
       DECODER
          |
          v
    RECONSTRUCTED
        IMAGE

You learned:

    [x] representation learning
    [x] encoder
    [x] decoder
    [x] latent representation
    [x] bottleneck
    [x] reconstruction loss
    [x] dimensionality reduction
    [x] latent-space analysis
    [x] reconstruction-based anomaly detection
    [x] limitations of ordinary autoencoders

Most importantly:

    A neural network can learn a useful visual representation without
    receiving an explicit class label.

This is a major conceptual step toward generative AI.
"""
    )

    # =========================================================================
    # NEXT LESSON
    # =========================================================================

    print()
    print("=" * 80)
    print("CONNECTION TO LESSON 10")
    print("=" * 80)

    print(
        """
The ordinary autoencoder gives us:

    image -> deterministic latent vector

But its latent space is not explicitly organized as a probability
distribution.

Therefore arbitrary random points in latent space may decode poorly.

Lesson 10 changes the model into:

                image
                  |
                  v
              ENCODER
                  |
             +----+----+
             |         |
             v         v
            mu       log_var
             |         |
             +----+----+
                  |
           reparameterization
                  |
                  v
                  z
                  |
                  v
               DECODER
                  |
                  v
              image

We will use:

    z = mu + sigma * epsilon

where:

    epsilon ~ N(0, I)

and:

    sigma = exp(0.5 * log_var)

The VAE objective becomes:

    L = reconstruction_loss
        + KL_divergence

The KL term regularizes the latent distribution toward:

    N(0, I)

That makes the latent space much more suitable for generation.

The course progression is therefore:

    CNN
      |
      v
    Deep CNN
      |
      v
    Representation Learning
      |
      v
    Autoencoder
      |
      v
    VAE          <-- NEXT
      |
      v
    GAN
      |
      v
    Attention / Transformers
      |
      v
    Diffusion
      |
      v
    Latent Diffusion
      |
      v
    Text Conditioning
      |
      v
    Own Text-to-Image Model
"""
    )

    print()
    print("=" * 80)
    print("END OF LESSON 9")
    print("=" * 80)


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()