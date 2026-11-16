"""
Filename: lesson10_variational_autoencoder.py

================================================================================
LESSON 10 — VARIATIONAL AUTOENCODER (VAE)
================================================================================

PURPOSE
-------
Lesson 9 introduced the ordinary autoencoder:

    image -> encoder -> latent vector -> decoder -> reconstructed image

The ordinary autoencoder can compress images, but its latent space is not
explicitly organized for generation.

This lesson introduces the Variational Autoencoder (VAE).

A VAE learns a probability distribution in latent space:

        image
          |
          v
       ENCODER
          |
      +---+---+
      |       |
      v       v
     mu     log_var
      |       |
      +---+---+
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
      image_hat

The VAE therefore becomes both:

    1. A representation-learning model
    2. A generative model

This is an important bridge from CNNs and autoencoders toward modern
generative vision models.

================================================================================
WHAT YOU WILL LEARN
================================================================================

1. Why ordinary autoencoders are not ideal generative models.
2. Probability distributions in latent space.
3. Mean and variance.
4. Gaussian latent variables.
5. The reparameterization trick.
6. KL divergence.
7. The VAE loss function.
8. Reconstruction loss.
9. KL regularization.
10. Sampling from a learned latent space.
11. Generating new images.
12. Interpolating between latent representations.
13. Posterior collapse intuition.
14. Reconstruction-vs-generation tradeoffs.
15. Why VAE concepts are important for latent diffusion.

================================================================================
COURSE ROADMAP
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
    Practical CNN

LESSON 8
    Deep CNN / residual learning

LESSON 9
    Autoencoder / representation learning

LESSON 10 <-- CURRENT
    Variational Autoencoder

LESSON 11
    GAN fundamentals

LESSON 12
    Attention

LESSON 13
    Transformers / Vision Transformers

LESSON 14+
    Generative modeling
    Diffusion
    DDPM
    U-Net
    DDIM
    Latent diffusion
    Cross-attention
    Text conditioning
    Text-to-image generation

================================================================================
REAL-WORLD PROBLEM
================================================================================

Suppose a robot camera observes thousands of objects.

We want a compact representation:

    image -> latent vector

but we also want:

    latent vector -> realistic image

An ordinary autoencoder does not guarantee that arbitrary latent vectors
produce meaningful images.

A VAE explicitly regularizes the latent space.

Instead of assigning a single point to each image:

    x -> z

the encoder predicts a distribution:

    x -> q(z|x)

usually modeled as a Gaussian:

    q(z|x) = N(mu(x), sigma(x)^2)

This gives us a structured latent space.

Then we can sample:

    z ~ N(0,I)

and decode:

    z -> decoder(z) -> generated image

This is our first complete generative vision pipeline.

================================================================================
THEORY
================================================================================

An ordinary autoencoder learns:

    z = f(x)

A VAE learns parameters of a probability distribution:

    mu = encoder_mu(x)

    log_var = encoder_logvar(x)

The variance is:

    sigma^2 = exp(log_var)

Therefore:

    sigma = exp(0.5 * log_var)

The latent distribution is:

    q(z|x) = N(mu, sigma^2)

--------------------------------------------------------------------------------
WHY LOG VARIANCE?
--------------------------------------------------------------------------------

Neural networks can output any real-valued number.

Variance must be positive.

Instead of directly predicting:

    sigma^2 > 0

we predict:

    log(sigma^2)

which can be any real number.

Then:

    sigma^2 = exp(log_var)

This guarantees positive variance.

--------------------------------------------------------------------------------
REPARAMETERIZATION TRICK
--------------------------------------------------------------------------------

A naive sampling operation:

    z ~ N(mu, sigma^2)

creates a problem for backpropagation.

The solution is:

    epsilon ~ N(0,I)

and:

    z = mu + sigma * epsilon

The random variable is now isolated in:

    epsilon

while:

    mu
    sigma

remain differentiable.

This is the reparameterization trick.

It allows gradient-based training.

--------------------------------------------------------------------------------
KL DIVERGENCE
--------------------------------------------------------------------------------

The VAE encourages:

    q(z|x)

to remain close to:

    p(z) = N(0,I)

For a diagonal Gaussian, the KL divergence has the closed form:

    KL(q(z|x) || N(0,I))

      =
    -1/2 * Σ [1 + log(sigma²) - mu² - sigma²]

Using:

    log_var = log(sigma²)

we obtain:

    KL
      =
    -1/2 * Σ
        [1 + log_var - mu² - exp(log_var)]

This term regularizes the latent space.

================================================================================
VAE OBJECTIVE
================================================================================

The VAE loss is:

    L = L_reconstruction + beta * L_KL

where:

    L_reconstruction
        measures how well the decoder reconstructs the image.

    L_KL
        encourages the latent distribution to remain close to N(0,I).

    beta
        controls the strength of the regularization.

For the standard VAE:

    beta = 1

This lesson uses a configurable beta.

================================================================================
RECONSTRUCTION LOSS
================================================================================

Our images are normalized:

    x ∈ [0,1]

The decoder uses:

    sigmoid

so:

    x_hat ∈ [0,1]

We use:

    MSE(x, x_hat)

for reconstruction.

Therefore:

    L_reconstruction
        =
    mean((x - x_hat)^2)

================================================================================
IMPORTANT SCALE ISSUE
================================================================================

There is an important implementation detail.

Suppose:

    reconstruction loss = mean over all pixels

while:

    KL loss = sum over latent dimensions

The two terms can have very different numerical scales.

For this lesson we explicitly calculate:

    reconstruction loss
        = mean squared error

and:

    KL loss
        = mean KL over the batch

This keeps the training behavior easier to inspect.

================================================================================
IMPLEMENTATION
================================================================================

Input:

    32 x 32 grayscale image

Flattened:

    1024 values

Encoder:

    1024 -> 512 -> 128

Then two parallel outputs:

    128 -> latent_dim       = mu

    128 -> latent_dim       = log_var

Sampling:

    epsilon ~ N(0,I)

    z = mu + exp(0.5*log_var)*epsilon

Decoder:

    latent_dim -> 128 -> 512 -> 1024

Output:

    sigmoid

================================================================================
EXPERIMENTS
================================================================================

EXPERIMENT 1
------------
Train a VAE.

EXPERIMENT 2
------------
Visualize:

    original
    reconstruction

EXPERIMENT 3
------------
Sample random latent vectors:

    z ~ N(0,I)

and decode them.

EXPERIMENT 4
------------
Interpolate between two latent vectors:

    z(t) = (1-t)z_a + t z_b

and decode each point.

This demonstrates the idea of a continuous latent space.

EXPERIMENT 5
------------
Compare:

    reconstruction loss
    KL loss
    total loss

================================================================================
REAL-WORLD APPLICATION
================================================================================

VAE-style latent representations can be used for:

    - anomaly detection
    - compact visual representation
    - data generation
    - synthetic data
    - latent-space interpolation
    - dataset exploration
    - robotics visual representation

For a robot:

    camera
       |
       v
    visual encoder
       |
       v
    latent state z
       |
       +------> perception
       |
       +------> manipulation policy
       |
       +------> anomaly detector
       |
       +------> generative model

The important future connection is:

    VAE encoder
         |
         v
    image -> latent space

Later:

    latent diffusion
         |
         v
    noise -> latent -> VAE decoder -> image

This is one of the foundations of latent diffusion systems.

================================================================================
IMPORTANT LIMITATIONS
================================================================================

A VAE usually produces smoother and blurrier reconstructions than a
high-quality GAN or diffusion model.

Why?

The model optimizes a probabilistic reconstruction objective and a
distributional regularization objective.

The VAE is not primarily optimized for maximum perceptual sharpness.

That is okay.

The important lesson is:

    VAE = structured probabilistic latent representation

rather than:

    VAE = state-of-the-art image generator

================================================================================
LESSON RESULT
================================================================================

After this lesson you should understand:

    [x] probabilistic latent variables
    [x] Gaussian latent distributions
    [x] mean and variance
    [x] log variance
    [x] reparameterization trick
    [x] KL divergence
    [x] VAE loss
    [x] random latent sampling
    [x] latent interpolation
    [x] structured latent spaces
    [x] why VAE matters for latent diffusion

================================================================================
NEXT LESSON
================================================================================

LESSON 11 introduces GANs.

We will move from:

    encoder + decoder + probability distribution

toward:

    generator + discriminator

The generator tries to create realistic images.

The discriminator tries to distinguish:

    real images
    vs
    generated images

This introduces adversarial learning.

Then we will compare:

    Autoencoder
    VAE
    GAN

before moving into attention and transformer architectures.

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

NUM_CLASSES = 4

TRAIN_SAMPLES = 2400
VAL_SAMPLES = 600
TEST_SAMPLES = 600

BATCH_SIZE = 64

EPOCHS = 35

LEARNING_RATE = 1e-3

LATENT_DIM = 16

BETA = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    """
    Make experiments reproducible as far as practical.
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
    print("LESSON 10 — VARIATIONAL AUTOENCODER")
    print("=" * 80)

    print(
        f"PyTorch version : {torch.__version__}"
    )

    print(
        f"Device          : {DEVICE}"
    )

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
    Generate a synthetic image.

    Classes:

        0 = horizontal bar
        1 = vertical bar
        2 = diagonal
        3 = cross

    Labels are NOT used by the VAE during training.

    They are retained only for visualization and analysis.
    """

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    thickness = np.random.randint(
        2,
        5
    )

    if label == 0:

        center = np.random.randint(
            size // 4,
            3 * size // 4
        )

        image[
            max(0, center - thickness):
            min(size, center + thickness),
            :
        ] = 1.0

    elif label == 1:

        center = np.random.randint(
            size // 4,
            3 * size // 4
        )

        image[
            :,
            max(0, center - thickness):
            min(size, center + thickness)
        ] = 1.0

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
            "Unknown class label."
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

    translated = np.zeros_like(
        image
    )

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
    # NOISE
    # -------------------------------------------------------------------------

    noise = np.random.normal(
        0.0,
        noise_level,
        image.shape
    ).astype(
        np.float32
    )

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
    Create a synthetic image dataset.
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

    images = images[
        indices
    ]

    labels = labels[
        indices
    ]

    return images, labels


# =============================================================================
# DATASET
# =============================================================================

class ImageDataset(Dataset):
    """
    Dataset for VAE training.

    The VAE target is the input itself.

    Therefore:

        input = target
    """

    def __init__(
        self,
        images
    ):

        self.images = torch.from_numpy(
            images
        ).float()

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        index
    ):

        return self.images[index]


# =============================================================================
# VARIATIONAL AUTOENCODER
# =============================================================================

class VAE(nn.Module):
    """
    Fully-connected Variational Autoencoder.

    Encoder:

        1024 -> 512 -> 128

    Then two outputs:

        128 -> latent_dim    mean
        128 -> latent_dim    log variance

    Reparameterization:

        z = mu + sigma * epsilon

    Decoder:

        latent_dim -> 128 -> 512 -> 1024
    """

    def __init__(
        self,
        latent_dim=LATENT_DIM
    ):

        super().__init__()

        self.latent_dim = latent_dim

        # ---------------------------------------------------------------------
        # SHARED ENCODER
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

            nn.ReLU()
        )

        # ---------------------------------------------------------------------
        # DISTRIBUTION PARAMETERS
        # ---------------------------------------------------------------------

        self.fc_mu = nn.Linear(
            128,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            128,
            latent_dim
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

    def encode(
        self,
        x
    ):
        """
        Encode image into:

            mu
            log_var
        """

        hidden = self.encoder(
            x
        )

        mu = self.fc_mu(
            hidden
        )

        log_var = self.fc_logvar(
            hidden
        )

        return mu, log_var

    def reparameterize(
        self,
        mu,
        log_var
    ):
        """
        Reparameterization trick.

        sigma = exp(0.5 * log_var)

        epsilon ~ N(0,I)

        z = mu + sigma * epsilon
        """

        std = torch.exp(
            0.5 * log_var
        )

        epsilon = torch.randn_like(
            std
        )

        z = (
            mu
            +
            std * epsilon
        )

        return z

    def decode(
        self,
        z
    ):
        """
        Decode latent vector.
        """

        reconstruction = self.decoder(
            z
        )

        return reconstruction.view(
            -1,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

    def forward(
        self,
        x
    ):
        """
        Complete VAE forward pass.
        """

        mu, log_var = self.encode(
            x
        )

        z = self.reparameterize(
            mu,
            log_var
        )

        reconstruction = self.decode(
            z
        )

        return (
            reconstruction,
            mu,
            log_var,
            z
        )


# =============================================================================
# PARAMETER COUNT
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
# VAE LOSS
# =============================================================================

def vae_loss(
    reconstruction,
    target,
    mu,
    log_var,
    beta=BETA
):
    """
    Calculate VAE loss.

    Reconstruction term:

        MSE(reconstruction, target)

    KL term:

        -0.5 * mean(
            1 + log_var
            - mu^2
            - exp(log_var)
        )

    Total:

        reconstruction_loss
        +
        beta * kl_loss
    """

    reconstruction_loss = torch.mean(
        (reconstruction - target) ** 2
    )

    kl_per_sample = -0.5 * torch.sum(
        1.0
        + log_var
        - mu.pow(2)
        - log_var.exp(),
        dim=1
    )

    kl_loss = torch.mean(
        kl_per_sample
    )

    total_loss = (
        reconstruction_loss
        +
        beta * kl_loss
    )

    return (
        total_loss,
        reconstruction_loss,
        kl_loss
    )


# =============================================================================
# DATASET VISUALIZATION
# =============================================================================

def visualize_dataset(
    images,
    labels
):
    """
    Display representative training images.
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
        "Synthetic VAE Dataset"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):
    """
    Train VAE for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total_samples = 0

    for images in loader:

        images = images.to(
            DEVICE
        )

        optimizer.zero_grad()

        reconstruction, mu, log_var, _ = model(
            images
        )

        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction,
            images,
            mu,
            log_var
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_reconstruction += (
            reconstruction_loss.item()
            * batch_size
        )

        total_kl += (
            kl_loss.item()
            * batch_size
        )

        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "reconstruction": (
            total_reconstruction
            / total_samples
        ),
        "kl": (
            total_kl
            / total_samples
        )
    }


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):
    """
    Evaluate VAE.
    """

    model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0

    total_samples = 0

    for images in loader:

        images = images.to(
            DEVICE
        )

        reconstruction, mu, log_var, _ = model(
            images
        )

        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction,
            images,
            mu,
            log_var
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_reconstruction += (
            reconstruction_loss.item()
            * batch_size
        )

        total_kl += (
            kl_loss.item()
            * batch_size
        )

        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "reconstruction": (
            total_reconstruction
            / total_samples
        ),
        "kl": (
            total_kl
            / total_samples
        )
    }


# =============================================================================
# COMPLETE TRAINING LOOP
# =============================================================================

def train_vae(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE
):
    """
    Complete VAE training loop.
    """

    model = model.to(
        DEVICE
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "train_loss": [],
        "train_reconstruction": [],
        "train_kl": [],
        "val_loss": [],
        "val_reconstruction": [],
        "val_kl": []
    }

    best_val_loss = float(
        "inf"
    )

    best_state = None

    for epoch in range(
        1,
        epochs + 1
    ):

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        val_metrics = evaluate(
            model,
            val_loader
        )

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["train_reconstruction"].append(
            train_metrics["reconstruction"]
        )

        history["train_kl"].append(
            train_metrics["kl"]
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["val_reconstruction"].append(
            val_metrics["reconstruction"]
        )

        history["val_kl"].append(
            val_metrics["kl"]
        )

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = (
                val_metrics["loss"]
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Total={train_metrics['loss']:.6f} | "
            f"Train Recon={train_metrics['reconstruction']:.6f} | "
            f"Train KL={train_metrics['kl']:.6f} | "
            f"Val Total={val_metrics['loss']:.6f} | "
            f"Val Recon={val_metrics['reconstruction']:.6f} | "
            f"Val KL={val_metrics['kl']:.6f}"
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
# LOSS VISUALIZATION
# =============================================================================

def plot_training_history(
    history
):
    """
    Plot:

        total loss
        reconstruction loss
        KL loss
    """

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Total"
    )

    plt.plot(
        epochs,
        history["train_reconstruction"],
        label="Reconstruction"
    )

    plt.plot(
        epochs,
        history["train_kl"],
        label="KL"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "VAE Training Loss Components"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


def plot_validation_history(
    history
):
    """
    Plot validation losses.
    """

    epochs = range(
        1,
        len(history["val_loss"]) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Total"
    )

    plt.plot(
        epochs,
        history["val_reconstruction"],
        label="Reconstruction"
    )

    plt.plot(
        epochs,
        history["val_kl"],
        label="KL"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation Loss"
    )

    plt.title(
        "VAE Validation Loss Components"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# RECONSTRUCTION VISUALIZATION
# =============================================================================

@torch.no_grad()
def visualize_reconstructions(
    model,
    dataset,
    num_examples=8
):
    """
    Show original and reconstructed images.
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

        reconstruction, _, _, _ = model(
            x
        )

        reconstruction = (
            reconstruction[0]
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
            reconstruction,
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
        "VAE"
    )

    plt.suptitle(
        "VAE Reconstruction"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# RANDOM GENERATION
# =============================================================================

@torch.no_grad()
def generate_random_images(
    model,
    num_images=12
):
    """
    Sample:

        z ~ N(0,I)

    and decode the samples.

    This is the generative capability introduced by the VAE.
    """

    model.eval()

    z = torch.randn(
        num_images,
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

    rows = 3
    cols = 4

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(8, 6)
    )

    for index, ax in enumerate(
        axes.flat
    ):

        ax.imshow(
            generated[index],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.axis(
            "off"
        )

    plt.suptitle(
        "Images Generated from Random VAE Latents"
    )

    plt.tight_layout()

    plt.show()

    return generated


# =============================================================================
# LATENT EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_latents(
    model,
    images
):
    """
    Extract posterior means.

    We use mu instead of sampled z for stable visualization.
    """

    model.eval()

    x = torch.from_numpy(
        images
    ).float().to(DEVICE)

    mu, log_var = model.encode(
        x
    )

    return (
        mu.cpu().numpy(),
        log_var.cpu().numpy()
    )


# =============================================================================
# LATENT SPACE VISUALIZATION
# =============================================================================

def visualize_latent_space(
    latent_vectors,
    labels
):
    """
    Visualize the first two latent dimensions.

    This is a simple visualization.

    It is not PCA or t-SNE.
    """

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
        "VAE Latent Space"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# LATENT INTERPOLATION
# =============================================================================

@torch.no_grad()
def latent_interpolation(
    model,
    dataset,
    index_a,
    index_b,
    steps=10
):
    """
    Interpolate between two encoded images.

    Formula:

        z(t)
          =
        (1-t) * z_a
        +
        t * z_b

    where:

        t ∈ [0,1]
    """

    model.eval()

    image_a = dataset[index_a]
    image_b = dataset[index_b]

    x_a = image_a.unsqueeze(
        0
    ).to(DEVICE)

    x_b = image_b.unsqueeze(
        0
    ).to(DEVICE)

    mu_a, _ = model.encode(
        x_a
    )

    mu_b, _ = model.encode(
        x_b
    )

    interpolation_values = torch.linspace(
        0.0,
        1.0,
        steps,
        device=DEVICE
    )

    generated = []

    for t in interpolation_values:

        z = (
            (1.0 - t) * mu_a
            +
            t * mu_b
        )

        image = model.decode(
            z
        )

        generated.append(
            image[0].cpu().numpy()
        )

    fig, axes = plt.subplots(
        1,
        steps,
        figsize=(16, 2)
    )

    for index, image in enumerate(
        generated
    ):

        axes[index].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[index].axis(
            "off"
        )

    plt.suptitle(
        "VAE Latent Interpolation"
    )

    plt.tight_layout()

    plt.show()

    return np.stack(
        generated
    )


# =============================================================================
# LATENT STATISTICS
# =============================================================================

@torch.no_grad()
def calculate_latent_statistics(
    model,
    images
):
    """
    Calculate empirical statistics of posterior means.

    The KL regularization should encourage latent means toward zero and
    latent variances toward one.

    We inspect the learned values rather than assuming perfect matching.
    """

    mu, log_var = extract_latents(
        model,
        images
    )

    mu_mean = float(
        np.mean(mu)
    )

    mu_std = float(
        np.std(mu)
    )

    variance = np.exp(
        log_var
    )

    variance_mean = float(
        np.mean(variance)
    )

    variance_std = float(
        np.std(variance)
    )

    print()
    print("=" * 80)
    print("LATENT DISTRIBUTION STATISTICS")
    print("=" * 80)

    print(
        f"Mean of posterior means     : "
        f"{mu_mean:.6f}"
    )

    print(
        f"Std of posterior means      : "
        f"{mu_std:.6f}"
    )

    print(
        f"Mean posterior variance     : "
        f"{variance_mean:.6f}"
    )

    print(
        f"Std posterior variance      : "
        f"{variance_std:.6f}"
    )

    return (
        mu,
        log_var
    )


# =============================================================================
# RECONSTRUCTION ERROR
# =============================================================================

@torch.no_grad()
def reconstruction_errors(
    model,
    images
):
    """
    Calculate per-image reconstruction MSE.
    """

    model.eval()

    x = torch.from_numpy(
        images
    ).float().to(DEVICE)

    reconstruction, _, _, _ = model(
        x
    )

    errors = (
        (x - reconstruction) ** 2
    ).mean(
        dim=(1, 2)
    )

    return errors.cpu().numpy()


# =============================================================================
# TESTS
# =============================================================================

def test_vae_shapes():
    """
    Verify all major VAE tensor shapes.
    """

    model = VAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    x = torch.rand(
        4,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    reconstruction, mu, log_var, z = model(
        x
    )

    assert reconstruction.shape == (
        4,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert mu.shape == (
        4,
        LATENT_DIM
    )

    assert log_var.shape == (
        4,
        LATENT_DIM
    )

    assert z.shape == (
        4,
        LATENT_DIM
    )

    print(
        "[PASS] VAE tensor-shape test"
    )


def test_reparameterization():
    """
    Verify that reparameterization preserves the latent shape and creates
    finite values.
    """

    model = VAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    mu = torch.zeros(
        8,
        LATENT_DIM,
        device=DEVICE
    )

    log_var = torch.zeros(
        8,
        LATENT_DIM,
        device=DEVICE
    )

    z = model.reparameterize(
        mu,
        log_var
    )

    assert z.shape == (
        8,
        LATENT_DIM
    )

    assert torch.isfinite(
        z
    ).all()

    print(
        "[PASS] Reparameterization test"
    )


def test_kl_zero_at_standard_normal():
    """
    If:

        mu = 0
        log_var = 0

    then:

        variance = 1

    and therefore:

        KL(N(0,1) || N(0,1)) = 0
    """

    reconstruction = torch.zeros(
        2,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    target = torch.zeros_like(
        reconstruction
    )

    mu = torch.zeros(
        2,
        LATENT_DIM
    )

    log_var = torch.zeros(
        2,
        LATENT_DIM
    )

    _, reconstruction_loss, kl_loss = vae_loss(
        reconstruction,
        target,
        mu,
        log_var
    )

    assert torch.allclose(
        kl_loss,
        torch.tensor(0.0),
        atol=1e-7
    )

    assert torch.allclose(
        reconstruction_loss,
        torch.tensor(0.0),
        atol=1e-7
    )

    print(
        "[PASS] KL standard-normal test"
    )


def test_kl_is_nonnegative():
    """
    KL divergence must be non-negative.

    We test several random Gaussian parameters.
    """

    mu = torch.randn(
        16,
        LATENT_DIM
    )

    log_var = torch.randn(
        16,
        LATENT_DIM
    )

    kl = -0.5 * torch.sum(
        1.0
        + log_var
        - mu.pow(2)
        - log_var.exp(),
        dim=1
    )

    assert torch.all(
        kl >= -1e-6
    )

    print(
        "[PASS] KL non-negativity test"
    )


def test_gradients():
    """
    Verify gradients propagate through:

        encoder
        mu/logvar
        reparameterization
        decoder
    """

    model = VAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    x = torch.rand(
        4,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    reconstruction, mu, log_var, _ = model(
        x
    )

    loss, _, _ = vae_loss(
        reconstruction,
        x,
        mu,
        log_var
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
        "[PASS] VAE gradient-flow test"
    )


def test_decoder_range():
    """
    Sigmoid decoder output must remain in [0,1].
    """

    model = VAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    z = torch.randn(
        4,
        LATENT_DIM
    ).to(DEVICE)

    with torch.no_grad():

        output = model.decode(
            z
        )

    assert torch.all(
        output >= 0.0
    )

    assert torch.all(
        output <= 1.0
    )

    print(
        "[PASS] Decoder output-range test"
    )


# =============================================================================
# REAL-WORLD APPLICATION
# =============================================================================

def real_world_application(
    test_metrics
):
    """
    Explain practical relevance.
    """

    print()
    print("=" * 80)
    print("REAL-WORLD APPLICATION")
    print("=" * 80)

    print(
        """
The VAE provides a structured probabilistic representation:

        camera image
             |
             v
          encoder
             |
       +-----+-----+
       |           |
       v           v
      mu         sigma
       |           |
       +-----+-----+
             |
             v
        latent z
             |
             v
          decoder
             |
             v
       reconstructed image

This can be useful for:

    1. Robot visual representation

       RGB camera
           ->
       VAE encoder
           ->
       compact latent state
           ->
       manipulation/control system

    2. Anomaly detection

       Normal observations:
           low reconstruction error

       Unusual observations:
           potentially higher reconstruction error

    3. Synthetic data

       Sample:
           z ~ N(0,I)

       Then:
           image = decoder(z)

    4. Latent interpolation

       z_A -> intermediate latent states -> z_B

       This can generate a smooth sequence of visual states.

    5. Latent diffusion

       The VAE can eventually provide:

           image <-> latent representation

       Diffusion can then operate in latent space rather than directly
       operating on full-resolution pixels.

This is one of the most important connections between VAEs and modern
latent generative models.
"""
    )

    print(
        f"Final test total loss          : "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Final test reconstruction loss : "
        f"{test_metrics['reconstruction']:.6f}"
    )

    print(
        f"Final test KL loss             : "
        f"{test_metrics['kl']:.6f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    print_environment()

    # =========================================================================
    # STEP 1 — UNIT TESTS
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 1 — VAE UNIT TESTS")
    print("=" * 80)

    test_vae_shapes()

    test_reparameterization()

    test_kl_zero_at_standard_normal()

    test_kl_is_nonnegative()

    test_gradients()

    test_decoder_range()

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
        f"Train shape: {train_images.shape}"
    )

    print(
        f"Val shape  : {val_images.shape}"
    )

    print(
        f"Test shape : {test_images.shape}"
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # =========================================================================
    # STEP 4 — CREATE VAE
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 4 — CREATE VAE")
    print("=" * 80)

    model = VAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    print(
        f"Latent dimension : "
        f"{LATENT_DIM}"
    )

    print(
        f"Trainable params  : "
        f"{count_parameters(model):,}"
    )

    # =========================================================================
    # STEP 5 — TRAIN VAE
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 5 — TRAIN VAE")
    print("=" * 80)

    history, best_state = train_vae(
        model,
        train_loader,
        val_loader
    )

    # =========================================================================
    # STEP 6 — TEST
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 6 — TEST VAE")
    print("=" * 80)

    test_metrics = evaluate(
        model,
        test_loader
    )

    print(
        f"Test total loss          : "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Test reconstruction loss : "
        f"{test_metrics['reconstruction']:.6f}"
    )

    print(
        f"Test KL loss             : "
        f"{test_metrics['kl']:.6f}"
    )

    # =========================================================================
    # STEP 7 — LOSS VISUALIZATION
    # =========================================================================

    plot_training_history(
        history
    )

    plot_validation_history(
        history
    )

    # =========================================================================
    # STEP 8 — RECONSTRUCTIONS
    # =========================================================================

    visualize_reconstructions(
        model,
        test_dataset
    )

    # =========================================================================
    # STEP 9 — LATENT SPACE
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 9 — LATENT SPACE")
    print("=" * 80)

    latent_means, latent_logvars = extract_latents(
        model,
        test_images
    )

    print(
        f"Latent mean shape: "
        f"{latent_means.shape}"
    )

    print(
        f"Latent logvar shape: "
        f"{latent_logvars.shape}"
    )

    visualize_latent_space(
        latent_means,
        test_labels
    )

    calculate_latent_statistics(
        model,
        test_images
    )

    # =========================================================================
    # STEP 10 — RANDOM GENERATION
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 10 — RANDOM LATENT GENERATION")
    print("=" * 80)

    generated_images = generate_random_images(
        model,
        num_images=12
    )

    assert generated_images.shape == (
        12,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    # =========================================================================
    # STEP 11 — LATENT INTERPOLATION
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 11 — LATENT INTERPOLATION")
    print("=" * 80)

    interpolation_images = latent_interpolation(
        model,
        test_dataset,
        index_a=0,
        index_b=1,
        steps=10
    )

    assert interpolation_images.shape == (
        10,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    # =========================================================================
    # STEP 12 — RECONSTRUCTION ERROR
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 12 — RECONSTRUCTION ERROR")
    print("=" * 80)

    errors = reconstruction_errors(
        model,
        test_images
    )

    print(
        f"Mean reconstruction error: "
        f"{errors.mean():.6f}"
    )

    print(
        f"Max reconstruction error: "
        f"{errors.max():.6f}"
    )

    assert np.isfinite(
        errors
    ).all()

    # =========================================================================
    # STEP 13 — REAL-WORLD APPLICATION
    # =========================================================================

    real_world_application(
        test_metrics
    )

    # =========================================================================
    # STEP 14 — FINAL VERIFICATION
    # =========================================================================

    print()
    print("=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)

    assert len(
        history["train_loss"]
    ) == EPOCHS

    assert len(
        history["val_loss"]
    ) == EPOCHS

    assert np.isfinite(
        np.array(
            history["train_loss"]
        )
    ).all()

    assert np.isfinite(
        np.array(
            history["train_kl"]
        )
    ).all()

    assert np.isfinite(
        test_metrics["loss"]
    )

    assert np.isfinite(
        test_metrics["reconstruction"]
    )

    assert np.isfinite(
        test_metrics["kl"]
    )

    assert test_metrics["loss"] >= 0.0

    assert test_metrics["reconstruction"] >= 0.0

    assert test_metrics["kl"] >= -1e-7

    # The model must actually learn something.
    assert (
        history["train_loss"][-1]
        <
        history["train_loss"][0]
    ), (
        "VAE total loss did not decrease."
    )

    print(
        "[PASS] Training history verification"
    )

    print(
        "[PASS] Finite VAE-loss verification"
    )

    print(
        "[PASS] Non-negative reconstruction loss verification"
    )

    print(
        "[PASS] Non-negative KL verification"
    )

    print(
        "[PASS] VAE training-loss reduction verification"
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
You have now built a complete Variational Autoencoder.

The key transformation is:

        IMAGE
          |
          v
       ENCODER
          |
      +---+---+
      |       |
      v       v
     mu     log_var
      |       |
      +---+---+
          |
          v
    reparameterization
          |
          v
          z
          |
          v
       DECODER
          |
          v
       IMAGE

The most important equations are:

    sigma = exp(0.5 * log_var)

    epsilon ~ N(0,I)

    z = mu + sigma * epsilon

and:

    L = L_reconstruction + beta * L_KL

You should now understand:

    [x] why a VAE predicts a distribution
    [x] mean and log variance
    [x] Gaussian latent variables
    [x] reparameterization
    [x] KL divergence
    [x] VAE training
    [x] latent-space regularization
    [x] random latent sampling
    [x] latent interpolation
    [x] why VAE is a generative model
    [x] why VAE is important for latent diffusion

This is a major milestone.

You have moved from:

    supervised visual recognition

to:

    probabilistic generative representation learning.
"""
    )

    # =========================================================================
    # NEXT LESSON
    # =========================================================================

    print()
    print("=" * 80)
    print("CONNECTION TO LESSON 11")
    print("=" * 80)

    print(
        """
LESSON 11 introduces GANs.

The VAE asks:

    "Can I learn a structured probabilistic latent space?"

GANs ask a different question:

    "Can a generator learn to produce images that look real?"

The architecture becomes:

                  RANDOM NOISE
                       |
                       v
                  GENERATOR
                       |
                       v
                 FAKE IMAGE
                       |
                       +----------------+
                                        |
                                        v
REAL IMAGE ------------------------> DISCRIMINATOR
                                        |
                                        v
                                 REAL / FAKE

This creates an adversarial game:

    Generator:
        produce realistic images

    Discriminator:
        distinguish real from generated

We will then compare:

    AUTOENCODER
        |
        v
    VAE
        |
        v
    GAN
        |
        v
    ATTENTION
        |
        v
    TRANSFORMERS
        |
        v
    DIFFUSION

The long-term connection remains:

    VAE
      |
      v
    image <-> latent representation
      |
      v
    latent diffusion
      |
      v
    conditional latent generation
      |
      v
    text-to-image model
"""
    )

    print()
    print("=" * 80)
    print("END OF LESSON 10")
    print("=" * 80)


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()