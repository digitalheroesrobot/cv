"""
Filename: lesson11_gan_from_scratch.py

================================================================================
LESSON 11 — GENERATIVE ADVERSARIAL NETWORK (GAN) FROM SCRATCH
================================================================================

PURPOSE
-------
In Lesson 10 we built a Variational Autoencoder (VAE).

The VAE learned:

    image -> latent distribution -> reconstructed image

and allowed:

    random latent vector -> generated image

However, VAE reconstructions often tend to be smooth or blurry.

This lesson introduces a different generative principle:

    GENERATIVE ADVERSARIAL NETWORK

A GAN contains two neural networks:

    1. Generator
    2. Discriminator

The Generator tries to create images that look real.

The Discriminator tries to distinguish:

    REAL image
        vs
    GENERATED image

The two networks compete during training.

This introduces the fundamental idea of adversarial learning.

================================================================================
WHAT YOU WILL LEARN
================================================================================

1. What a GAN is.
2. Generator architecture.
3. Discriminator architecture.
4. Random latent noise.
5. Binary classification for the discriminator.
6. Adversarial loss.
7. Minimax game intuition.
8. Why GAN training is unstable.
9. Generator gradients.
10. Discriminator gradients.
11. Fake-image generation.
12. Latent-space sampling.
13. GAN latent interpolation.
14. Mode collapse.
15. Generator/discriminator balance.
16. Why GANs produce sharper images than many VAEs.
17. How GANs differ from VAEs.
18. Why GANs matter historically for generative vision.
19. The connection from GANs to modern diffusion models.

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
    Autoencoder

LESSON 10
    Variational Autoencoder

LESSON 11 <-- CURRENT
    Generative Adversarial Network

LESSON 12
    Attention / self-attention

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

Imagine a robot dataset containing thousands of camera images.

Collecting every possible visual situation can be expensive.

We would like to generate additional realistic-looking images.

For example:

    real robot camera image
          |
          v
       dataset

and simultaneously:

    random noise
          |
          v
       generator
          |
          v
    synthetic image

The synthetic image should eventually resemble the real training distribution.

A GAN learns this without requiring an explicit pixel-by-pixel reconstruction
target for each generated image.

================================================================================
CORE GAN IDEA
================================================================================

The GAN consists of:

                    RANDOM NOISE
                         |
                         v
                    GENERATOR
                         |
                         v
                    FAKE IMAGE
                         |
                         v
              +---------------------+
              |                     |
              v                     v
          DISCRIMINATOR        DISCRIMINATOR
              ^                     ^
              |                     |
          REAL IMAGE          FAKE IMAGE

The discriminator learns:

    D(real) -> 1
    D(fake) -> 0

The generator learns:

    G(z) -> image

and tries to make:

    D(G(z)) -> 1

Therefore:

    Discriminator:
        "Can I detect fake images?"

    Generator:
        "Can I fool the discriminator?"

================================================================================
MATHEMATICAL FORMULATION
================================================================================

Let:

    z ~ p(z)

be random latent noise.

Usually:

    z ~ N(0,I)

The generator maps:

    G(z) -> x_fake

The discriminator maps:

    D(x) -> probability that x is real

The original GAN objective is:

    min_G max_D V(D,G)

where:

    V(D,G)
      =
    E_x~pdata [log D(x)]
    +
    E_z~pz [log(1-D(G(z)))]

The discriminator wants:

    D(real) -> 1
    D(fake) -> 0

The generator wants:

    D(fake) -> 1

================================================================================
PRACTICAL GENERATOR LOSS
================================================================================

In practice, the original minimax generator loss can produce weak gradients.

We therefore normally use the non-saturating generator objective:

    L_G
      =
    -E_z [log D(G(z))]

So the generator directly tries to maximize:

    D(G(z))

This is what this lesson implements.

================================================================================
DISCRIMINATOR LOSS
================================================================================

The discriminator performs binary classification.

For real images:

    target = 1

For fake images:

    target = 0

The binary cross entropy loss is:

    BCE(y_hat, y)

Therefore:

    L_D
      =
    BCE(D(real), 1)
    +
    BCE(D(fake), 0)

================================================================================
IMPORTANT IMPLEMENTATION DETAIL
================================================================================

We use:

    BCEWithLogitsLoss

rather than:

    Sigmoid + BCELoss

inside the discriminator.

The discriminator outputs raw logits:

    logits = D(x)

and:

    BCEWithLogitsLoss(logits, target)

combines sigmoid and binary cross entropy in a numerically stable way.

This is an important practical deep-learning pattern.

================================================================================
MODEL ARCHITECTURE
================================================================================

For this lesson the images are:

    32 x 32 grayscale

The generator starts from:

    latent vector z
        shape = [batch, LATENT_DIM]

and progressively creates:

    4 x 4
    8 x 8
    16 x 16
    32 x 32

Architecture:

    z
    |
    Linear
    |
    4x4 feature map
    |
    ConvTranspose2d
    |
    8x8
    |
    ConvTranspose2d
    |
    16x16
    |
    ConvTranspose2d
    |
    32x32
    |
    Tanh
    |
    generated image

The discriminator performs the opposite operation:

    32x32
       |
    Conv
       |
    16x16
       |
    Conv
       |
    8x8
       |
    Conv
       |
    4x4
       |
    Linear
       |
    real/fake logit

================================================================================
IMAGE NORMALIZATION
================================================================================

The generator uses:

    Tanh

Therefore generated pixels are:

    [-1,1]

So real images must also be normalized:

    x_normalized = 2*x - 1

This is important.

If real images are in:

    [0,1]

while fake images are in:

    [-1,1]

the discriminator can distinguish them using trivial statistics.

That would create a bad training signal.

Therefore:

    REAL:
        [0,1] -> [-1,1]

    FAKE:
        already [-1,1]

================================================================================
WHY BATCH NORMALIZATION?
================================================================================

GAN training can be unstable.

Batch normalization can help maintain reasonable activation statistics.

The generator uses BatchNorm in intermediate layers.

The discriminator uses convolutional layers with LeakyReLU.

================================================================================
WHY LEAKY ReLU?
================================================================================

Standard ReLU:

    f(x) = max(0,x)

has zero gradient for negative values.

LeakyReLU keeps a small negative slope:

    f(x) =
        x          if x > 0
        alpha*x    otherwise

This can help reduce dead neurons.

GAN discriminators commonly use LeakyReLU.

================================================================================
WHY TANH IN THE GENERATOR?
================================================================================

The generator output needs a bounded range.

Tanh produces:

    [-1,1]

which matches our normalized image representation.

================================================================================
GAN TRAINING LOOP
================================================================================

For every batch:

STEP 1
------
Generate random latent vectors:

    z ~ N(0,I)

STEP 2
------
Generate fake images:

    fake = G(z)

STEP 3
------
Train discriminator on real images:

    D(real) -> 1

STEP 4
------
Train discriminator on fake images:

    D(fake) -> 0

STEP 5
------
Generate a fresh fake batch.

STEP 6
------
Train generator.

The generator wants:

    D(G(z)) -> 1

This alternating optimization is the core of GAN training.

================================================================================
WHY GAN TRAINING IS DIFFICULT
================================================================================

Unlike ordinary supervised learning, GAN training is not simply:

    minimize one fixed objective

Instead:

    Generator changes
         |
         v
    Discriminator changes
         |
         v
    Generator changes
         |
         v
    ...

The objective is dynamic.

Problems can include:

    - mode collapse
    - discriminator overpowering generator
    - generator overpowering discriminator
    - unstable oscillation
    - vanishing gradients
    - sensitivity to learning rate
    - sensitivity to architecture
    - sensitivity to initialization

Understanding these problems is important even if later we use diffusion.

================================================================================
MODE COLLAPSE
================================================================================

Suppose the training dataset contains:

    horizontal bars
    vertical bars
    diagonals
    crosses

A bad generator might discover:

    "I can fool the discriminator by generating only horizontal bars."

Then:

    z1 -> horizontal
    z2 -> horizontal
    z3 -> horizontal
    z4 -> horizontal

The images may look realistic individually.

But diversity is poor.

This is called:

    MODE COLLAPSE

A good generative model should represent multiple modes of the data distribution.

================================================================================
VAE VS GAN
================================================================================

VAE:

    probabilistic encoder
        +
    decoder
        +
    reconstruction loss
        +
    KL regularization

GAN:

    generator
        +
    discriminator
        +
    adversarial loss

VAE strengths:

    - structured latent space
    - stable training
    - useful representation
    - explicit probabilistic formulation

VAE weaknesses:

    - often blurry images

GAN strengths:

    - sharp images
    - strong perceptual realism

GAN weaknesses:

    - difficult training
    - mode collapse
    - unstable optimization

================================================================================
REAL-WORLD APPLICATION
================================================================================

GAN-style generation has historically been used for:

    - synthetic images
    - image-to-image translation
    - super-resolution
    - face generation
    - data augmentation
    - domain adaptation
    - synthetic training datasets

For robotics, a generator could eventually help produce:

    robot scene
    object appearance
    background variation
    lighting variation
    camera variation

However, synthetic images must be validated carefully before being used
for safety-critical robot perception.

================================================================================
EXPERIMENTS
================================================================================

EXPERIMENT 1
------------
Train GAN.

EXPERIMENT 2
------------
Visualize generated images after each training stage.

EXPERIMENT 3
------------
Measure:

    discriminator(real)
    discriminator(fake)

EXPERIMENT 4
------------
Generate many samples from random latent vectors.

EXPERIMENT 5
------------
Interpolate between latent vectors.

EXPERIMENT 6
------------
Inspect whether multiple visual modes appear.

================================================================================
VERIFICATION / TESTS
================================================================================

We verify:

    [1] generator output shape
    [2] generator output range
    [3] discriminator output shape
    [4] discriminator gradients
    [5] generator gradients
    [6] BCE loss behavior
    [7] generated-image numerical validity
    [8] training loss history
    [9] latent interpolation shape
    [10] generated samples remain finite

================================================================================
REAL-WORLD RESULT
================================================================================

After training, the system should be able to perform:

    random noise
        |
        v
    generator
        |
        v
    synthetic 32x32 visual pattern

The generated image does NOT need to reproduce a specific training image.

Instead, the generator attempts to learn the distribution:

    p_data(x)

This is a fundamental conceptual difference from reconstruction-based
models.

================================================================================
LESSON RESULT
================================================================================

After this lesson you should understand:

    [x] GAN architecture
    [x] Generator
    [x] Discriminator
    [x] adversarial learning
    [x] binary cross entropy
    [x] generator objective
    [x] discriminator objective
    [x] alternating optimization
    [x] mode collapse
    [x] GAN instability
    [x] latent sampling
    [x] latent interpolation
    [x] why GANs can generate sharp images

================================================================================
CONNECTION TO NEXT LESSON
================================================================================

The next major problem is:

    How can a neural network understand relationships between distant
    elements of an image?

CNNs use local receptive fields.

GANs use generator/discriminator competition.

Modern vision models introduced another powerful idea:

    ATTENTION

Attention allows a model to dynamically determine which parts of the input
should influence another part.

This leads to:

    self-attention
        |
        v
    multi-head attention
        |
        v
    Transformer
        |
        v
    Vision Transformer
        |
        v
    cross-attention
        |
        v
    text-conditioned image generation

Eventually this becomes critical for:

    TEXT
      |
      v
    cross-attention
      |
      v
    diffusion U-Net
      |
      v
    IMAGE

================================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

import copy
import random

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

CHANNELS = 1

LATENT_DIM = 64

TRAIN_SAMPLES = 6000

BATCH_SIZE = 64

EPOCHS = 40

LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4

BETA1 = 0.5
BETA2 = 0.999

NUM_GENERATED = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    """
    Make the experiment as reproducible as practical.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =============================================================================
# ENVIRONMENT INFORMATION
# =============================================================================

def print_environment():

    print("=" * 80)
    print("LESSON 11 — GAN FROM SCRATCH")
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
# SYNTHETIC DATASET
# =============================================================================

def create_shape_image(
    label,
    size=IMAGE_SIZE,
    noise_level=0.05
):
    """
    Generate one synthetic grayscale image.

    Classes:

        0 = horizontal bar
        1 = vertical bar
        2 = diagonal
        3 = cross

    GAN training does NOT use the labels.

    Labels are retained only for dataset visualization and mode analysis.
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

        direction = np.random.choice(
            [-1, 1]
        )

        for i in range(size):

            if direction == 1:

                center = i

            else:

                center = size - 1 - i

            for offset in range(
                -thickness,
                thickness + 1
            ):

                j = center + offset

                if 0 <= j < size:

                    image[
                        i,
                        j
                    ] = 1.0

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
            "Invalid shape label."
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
    Create synthetic training data.

    The data distribution contains four visual modes.
    """

    images = []

    labels = []

    for index in range(
        num_samples
    ):

        label = index % 4

        image = create_shape_image(
            label
        )

        images.append(
            image
        )

        labels.append(
            label
        )

    images = np.stack(
        images
    )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    permutation = np.random.permutation(
        len(images)
    )

    images = images[
        permutation
    ]

    labels = labels[
        permutation
    ]

    return images, labels


# =============================================================================
# DATASET CLASS
# =============================================================================

class GANDataset(Dataset):
    """
    Dataset used by the discriminator.

    Images are normalized:

        [0,1] -> [-1,1]

    because the generator uses Tanh.
    """

    def __init__(
        self,
        images
    ):

        normalized = (
            images * 2.0
            - 1.0
        )

        self.images = torch.from_numpy(
            normalized
        ).float()

    def __len__(
        self
    ):

        return len(
            self.images
        )

    def __getitem__(
        self,
        index
    ):

        image = self.images[index]

        return image.unsqueeze(
            0
        )


# =============================================================================
# GENERATOR
# =============================================================================

class Generator(nn.Module):
    """
    DCGAN-style generator.

    Input:

        [B, LATENT_DIM]

    Output:

        [B, 1, 32, 32]
    """

    def __init__(
        self,
        latent_dim=LATENT_DIM
    ):

        super().__init__()

        self.latent_dim = latent_dim

        self.project = nn.Sequential(

            nn.Linear(
                latent_dim,
                256 * 4 * 4
            ),

            nn.BatchNorm1d(
                256 * 4 * 4
            ),

            nn.ReLU(
                inplace=True
            )
        )

        self.network = nn.Sequential(

            # 4x4 -> 8x8
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                64,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(
        self,
        z
    ):

        x = self.project(
            z
        )

        x = x.view(
            -1,
            256,
            4,
            4
        )

        return self.network(
            x
        )


# =============================================================================
# DISCRIMINATOR
# =============================================================================

class Discriminator(nn.Module):
    """
    DCGAN-style discriminator.

    Input:

        [B,1,32,32]

    Output:

        [B,1]

    The output is a LOGIT.

    We intentionally do NOT apply sigmoid here because
    BCEWithLogitsLoss performs the numerically stable sigmoid internally.
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # 32x32 -> 16x16
            nn.Conv2d(
                1,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 16x16 -> 8x8
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 8x8 -> 4x4
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Flatten(),

            nn.Linear(
                256 * 4 * 4,
                1
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(
            x
        ).view(
            -1
        )


# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================

def initialize_weights(
    model
):
    """
    Standard DCGAN-style initialization.

    Convolution weights:

        N(0, 0.02)

    BatchNorm weights:

        N(1, 0.02)

    Bias:

        0
    """

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d
            )
        ):

            nn.init.normal_(
                module.weight,
                0.0,
                0.02
            )

        elif isinstance(
            module,
            nn.BatchNorm2d
        ):

            nn.init.normal_(
                module.weight,
                1.0,
                0.02
            )

            nn.init.constant_(
                module.bias,
                0.0
            )

        elif isinstance(
            module,
            nn.BatchNorm1d
        ):

            nn.init.normal_(
                module.weight,
                1.0,
                0.02
            )

            nn.init.constant_(
                module.bias,
                0.0
            )

        elif isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                0.0,
                0.02
            )

            if module.bias is not None:

                nn.init.constant_(
                    module.bias,
                    0.0
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
# DATA VISUALIZATION
# =============================================================================

def denormalize_images(
    images
):
    """
    Convert:

        [-1,1] -> [0,1]
    """

    return (
        images + 1.0
    ) / 2.0


def visualize_real_dataset(
    images,
    labels
):
    """
    Display representative real images.
    """

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(10, 3)
    )

    for class_id in range(4):

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
            f"Mode {class_id}"
        )

        axes[class_id].axis(
            "off"
        )

    plt.suptitle(
        "Real Training Distribution"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# GENERATE IMAGES
# =============================================================================

@torch.no_grad()
def generate_images(
    generator,
    num_images=16,
    latent_vectors=None
):
    """
    Generate images from latent vectors.

    If latent_vectors is None:

        z ~ N(0,I)

    Otherwise the provided latent vectors are used.
    """

    generator.eval()

    if latent_vectors is None:

        z = torch.randn(
            num_images,
            LATENT_DIM,
            device=DEVICE
        )

    else:

        z = latent_vectors.to(
            DEVICE
        )

    fake_images = generator(
        z
    )

    return fake_images


# =============================================================================
# IMAGE GRID
# =============================================================================

def show_image_grid(
    images,
    title,
    nrow=4
):
    """
    Display a tensor or NumPy image collection.
    """

    if torch.is_tensor(
        images
    ):

        images = images.detach().cpu()

    else:

        images = torch.from_numpy(
            images
        )

    images = denormalize_images(
        images
    )

    images = images.clamp(
        0.0,
        1.0
    )

    num_images = len(
        images
    )

    ncol = int(
        np.ceil(
            num_images / nrow
        )
    )

    fig, axes = plt.subplots(
        ncol,
        nrow,
        figsize=(10, 2.5 * ncol)
    )

    axes = np.asarray(
        axes
    ).reshape(
        ncol,
        nrow
    )

    for index, ax in enumerate(
        axes.flat
    ):

        if index < num_images:

            image = images[index]

            if image.ndim == 3:

                image = image[0]

            ax.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1
            )

        ax.axis(
            "off"
        )

    plt.suptitle(
        title
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# TRAINING
# =============================================================================

def train_gan(
    generator,
    discriminator,
    loader,
    epochs=EPOCHS
):
    """
    Main GAN training loop.

    For each batch:

        1. Train discriminator on real.
        2. Train discriminator on fake.
        3. Train generator against discriminator.

    We keep a fixed latent vector for visualization throughout training.
    """

    generator.train()

    discriminator.train()

    criterion = nn.BCEWithLogitsLoss()

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE_G,
        betas=(BETA1, BETA2)
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE_D,
        betas=(BETA1, BETA2)
    )

    fixed_noise = torch.randn(
        NUM_GENERATED,
        LATENT_DIM,
        device=DEVICE
    )

    history = {
        "g_loss": [],
        "d_loss": [],
        "d_real": [],
        "d_fake": []
    }

    checkpoints = []

    for epoch in range(
        1,
        epochs + 1
    ):

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0

        batches = 0

        for real_images in loader:

            real_images = real_images.to(
                DEVICE
            )

            batch_size = (
                real_images.size(0)
            )

            # =================================================================
            # STEP A — TRAIN DISCRIMINATOR
            # =================================================================

            optimizer_d.zero_grad()

            # Real images.
            real_logits = discriminator(
                real_images
            )

            real_targets = torch.ones(
                batch_size,
                device=DEVICE
            )

            real_loss = criterion(
                real_logits,
                real_targets
            )

            # Fake images.
            z = torch.randn(
                batch_size,
                LATENT_DIM,
                device=DEVICE
            )

            fake_images = generator(
                z
            )

            fake_logits = discriminator(
                fake_images.detach()
            )

            fake_targets = torch.zeros(
                batch_size,
                device=DEVICE
            )

            fake_loss = criterion(
                fake_logits,
                fake_targets
            )

            discriminator_loss = (
                real_loss
                +
                fake_loss
            ) / 2.0

            discriminator_loss.backward()

            optimizer_d.step()

            # =================================================================
            # STEP B — TRAIN GENERATOR
            # =================================================================

            optimizer_g.zero_grad()

            z = torch.randn(
                batch_size,
                LATENT_DIM,
                device=DEVICE
            )

            fake_images = generator(
                z
            )

            fake_logits = discriminator(
                fake_images
            )

            generator_targets = torch.ones(
                batch_size,
                device=DEVICE
            )

            generator_loss = criterion(
                fake_logits,
                generator_targets
            )

            generator_loss.backward()

            optimizer_g.step()

            # =================================================================
            # STATISTICS
            # =================================================================

            with torch.no_grad():

                real_probability = torch.sigmoid(
                    real_logits
                ).mean().item()

                fake_probability = torch.sigmoid(
                    fake_logits
                ).mean().item()

            epoch_g_loss += (
                generator_loss.item()
            )

            epoch_d_loss += (
                discriminator_loss.item()
            )

            epoch_d_real += (
                real_probability
            )

            epoch_d_fake += (
                fake_probability
            )

            batches += 1

        avg_g = (
            epoch_g_loss
            / batches
        )

        avg_d = (
            epoch_d_loss
            / batches
        )

        avg_real = (
            epoch_d_real
            / batches
        )

        avg_fake = (
            epoch_d_fake
            / batches
        )

        history["g_loss"].append(
            avg_g
        )

        history["d_loss"].append(
            avg_d
        )

        history["d_real"].append(
            avg_real
        )

        history["d_fake"].append(
            avg_fake
        )

        # Keep several generated snapshots.
        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == epochs
        ):

            generator.eval()

            with torch.no_grad():

                snapshot = generator(
                    fixed_noise
                ).cpu()

            checkpoints.append(
                (
                    epoch,
                    snapshot
                )
            )

            generator.train()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"D={avg_d:.4f} | "
            f"G={avg_g:.4f} | "
            f"D(real)={avg_real:.3f} | "
            f"D(fake)={avg_fake:.3f}"
        )

    return (
        history,
        checkpoints,
        fixed_noise
    )


# =============================================================================
# TRAINING CURVES
# =============================================================================

def plot_training_history(
    history
):
    """
    Plot GAN losses.
    """

    epochs = np.arange(
        1,
        len(history["g_loss"]) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["g_loss"],
        label="Generator loss"
    )

    plt.plot(
        epochs,
        history["d_loss"],
        label="Discriminator loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "GAN Training Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


def plot_discriminator_history(
    history
):
    """
    Plot discriminator confidence.

    Important:

    These values are not accuracy.

    They are average sigmoid probabilities.
    """

    epochs = np.arange(
        1,
        len(history["d_real"]) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["d_real"],
        label="D(real)"
    )

    plt.plot(
        epochs,
        history["d_fake"],
        label="D(fake)"
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="0.5 reference"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Probability"
    )

    plt.title(
        "Discriminator Output"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# TRAINING PROGRESS VISUALIZATION
# =============================================================================

def visualize_training_progress(
    checkpoints
):
    """
    Display fixed-noise generations at different training epochs.

    The same latent vectors are used each time.

    Therefore visual changes correspond more directly to learning.
    """

    if len(
        checkpoints
    ) == 0:

        return

    rows = len(
        checkpoints
    )

    cols = NUM_GENERATED

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 2.2 * rows)
    )

    if rows == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row, (
        epoch,
        images
    ) in enumerate(
        checkpoints
    ):

        images = denormalize_images(
            images
        )

        for column in range(
            cols
        ):

            axes[
                row,
                column
            ].imshow(
                images[column, 0],
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[
                row,
                column
            ].axis(
                "off"
            )

        axes[
            row,
            0
        ].set_ylabel(
            f"Epoch {epoch}"
        )

    plt.suptitle(
        "GAN Learning Progress with Fixed Latent Vectors"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# LATENT INTERPOLATION
# =============================================================================

@torch.no_grad()
def latent_interpolation(
    generator,
    steps=12
):
    """
    Generate a smooth sequence between two latent vectors.

    z(t) = (1-t) z_A + t z_B
    """

    generator.eval()

    z_a = torch.randn(
        1,
        LATENT_DIM,
        device=DEVICE
    )

    z_b = torch.randn(
        1,
        LATENT_DIM,
        device=DEVICE
    )

    values = torch.linspace(
        0.0,
        1.0,
        steps,
        device=DEVICE
    )

    latent_vectors = []

    for t in values:

        z = (
            (1.0 - t) * z_a
            +
            t * z_b
        )

        latent_vectors.append(
            z
        )

    latent_vectors = torch.cat(
        latent_vectors,
        dim=0
    )

    generated = generator(
        latent_vectors
    ).cpu()

    show_image_grid(
        generated,
        "GAN Latent Interpolation",
        nrow=steps
    )

    return generated


# =============================================================================
# DISCRIMINATOR ANALYSIS
# =============================================================================

@torch.no_grad()
def analyze_discriminator(
    generator,
    discriminator,
    real_images
):
    """
    Compare discriminator outputs for real and generated images.
    """

    generator.eval()

    discriminator.eval()

    num_samples = min(
        64,
        len(real_images)
    )

    real = torch.from_numpy(
        real_images[:num_samples]
    ).float()

    real = (
        real * 2.0
        - 1.0
    )

    real = real.unsqueeze(
        1
    ).to(DEVICE)

    z = torch.randn(
        num_samples,
        LATENT_DIM,
        device=DEVICE
    )

    fake = generator(
        z
    )

    real_probability = torch.sigmoid(
        discriminator(real)
    )

    fake_probability = torch.sigmoid(
        discriminator(fake)
    )

    print()
    print("=" * 80)
    print("DISCRIMINATOR ANALYSIS")
    print("=" * 80)

    print(
        f"Average D(real): "
        f"{real_probability.mean().item():.4f}"
    )

    print(
        f"Average D(fake): "
        f"{fake_probability.mean().item():.4f}"
    )

    print(
        f"Std D(real): "
        f"{real_probability.std().item():.4f}"
    )

    print(
        f"Std D(fake): "
        f"{fake_probability.std().item():.4f}"
    )

    return (
        real_probability.cpu().numpy(),
        fake_probability.cpu().numpy()
    )


# =============================================================================
# MODE ANALYSIS
# =============================================================================

def simple_generated_mode_analysis(
    generated_images
):
    """
    Very simple heuristic mode analysis.

    This is NOT a learned classifier.

    We estimate whether an image is more horizontal, vertical, diagonal,
    or cross-like based on directional energy.

    The purpose is educational:

        Does the generator produce only one type of pattern?

    This gives intuition about mode collapse.
    """

    images = generated_images

    if torch.is_tensor(
        images
    ):

        images = images.cpu().numpy()

    images = denormalize_images(
        torch.from_numpy(
            images
        )
    ).numpy()

    if images.ndim == 4:

        images = images[:, 0]

    horizontal_energy = []

    vertical_energy = []

    diagonal_energy = []

    for image in images:

        horizontal = np.abs(
            np.diff(
                image,
                axis=0
            )
        ).mean()

        vertical = np.abs(
            np.diff(
                image,
                axis=1
            )
        ).mean()

        diagonal = np.abs(
            image[
                1:,
                1:
            ]
            -
            image[
                :-1,
                :-1
            ]
        ).mean()

        horizontal_energy.append(
            horizontal
        )

        vertical_energy.append(
            vertical
        )

        diagonal_energy.append(
            diagonal
        )

    print()
    print("=" * 80)
    print("SIMPLE DIVERSITY ANALYSIS")
    print("=" * 80)

    print(
        f"Mean horizontal variation : "
        f"{np.mean(horizontal_energy):.6f}"
    )

    print(
        f"Mean vertical variation   : "
        f"{np.mean(vertical_energy):.6f}"
    )

    print(
        f"Mean diagonal variation   : "
        f"{np.mean(diagonal_energy):.6f}"
    )

    print(
        """
This heuristic is intentionally simple.

It is NOT a proper mode-collapse metric.

A real generative-model evaluation system would use stronger methods such as:

    - learned classifiers
    - feature-space diversity
    - precision / recall for generative models
    - FID
    - KID
    - human evaluation
    - downstream task performance
"""
    )


# =============================================================================
# GRADIENT VERIFICATION
# =============================================================================

def test_gan_gradient_flow():
    """
    Verify that both networks receive gradients.

    This does not compare numerical and analytical gradients.

    Instead, it verifies the most important computational graph property:

        loss -> network -> parameters -> gradients
    """

    generator = Generator().to(
        DEVICE
    )

    discriminator = Discriminator().to(
        DEVICE
    )

    generator.apply(
        initialize_weights
    )

    discriminator.apply(
        initialize_weights
    )

    criterion = nn.BCEWithLogitsLoss()

    batch_size = 4

    z = torch.randn(
        batch_size,
        LATENT_DIM,
        device=DEVICE
    )

    fake = generator(
        z
    )

    logits = discriminator(
        fake
    )

    target = torch.ones(
        batch_size,
        device=DEVICE
    )

    loss = criterion(
        logits,
        target
    )

    loss.backward()

    generator_gradients = 0

    for parameter in generator.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all()

            generator_gradients += 1

    discriminator_gradients = 0

    # New discriminator for independent verification.
    discriminator.zero_grad()

    real = torch.randn(
        batch_size,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE
    )

    logits = discriminator(
        real
    )

    target = torch.ones(
        batch_size,
        device=DEVICE
    )

    loss_d = criterion(
        logits,
        target
    )

    loss_d.backward()

    for parameter in discriminator.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all()

            discriminator_gradients += 1

    assert generator_gradients > 0

    assert discriminator_gradients > 0

    print(
        "[PASS] Generator gradient-flow test"
    )

    print(
        "[PASS] Discriminator gradient-flow test"
    )


# =============================================================================
# SHAPE TESTS
# =============================================================================

def test_generator_shape():

    generator = Generator().to(
        DEVICE
    )

    z = torch.randn(
        4,
        LATENT_DIM,
        device=DEVICE
    )

    output = generator(
        z
    )

    assert output.shape == (
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    print(
        "[PASS] Generator output-shape test"
    )


def test_generator_range():

    generator = Generator().to(
        DEVICE
    )

    z = torch.randn(
        8,
        LATENT_DIM,
        device=DEVICE
    )

    with torch.no_grad():

        output = generator(
            z
        )

    assert torch.all(
        output >= -1.0
    )

    assert torch.all(
        output <= 1.0
    )

    print(
        "[PASS] Generator output-range test"
    )


def test_discriminator_shape():

    discriminator = Discriminator().to(
        DEVICE
    )

    x = torch.randn(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE
    )

    output = discriminator(
        x
    )

    assert output.shape == (
        4,
    )

    print(
        "[PASS] Discriminator output-shape test"
    )


# =============================================================================
# LOSS TEST
# =============================================================================

def test_bce_behavior():
    """
    Verify basic discriminator-loss behavior.

    Correct predictions should have lower BCE loss than strongly incorrect
    predictions.
    """

    criterion = nn.BCEWithLogitsLoss()

    good_real_logits = torch.tensor(
        [8.0, 7.0, 6.0]
    )

    real_targets = torch.ones(
        3
    )

    bad_real_logits = torch.tensor(
        [-8.0, -7.0, -6.0]
    )

    good_loss = criterion(
        good_real_logits,
        real_targets
    )

    bad_loss = criterion(
        bad_real_logits,
        real_targets
    )

    assert (
        good_loss < bad_loss
    )

    print(
        "[PASS] BCE discriminator-loss behavior test"
    )


# =============================================================================
# NUMERICAL OUTPUT TEST
# =============================================================================

def test_finite_outputs(
    generator,
    discriminator
):
    """
    Verify no NaN or infinite values appear.
    """

    z = torch.randn(
        8,
        LATENT_DIM,
        device=DEVICE
    )

    with torch.no_grad():

        fake = generator(
            z
        )

        logits = discriminator(
            fake
        )

    assert torch.isfinite(
        fake
    ).all()

    assert torch.isfinite(
        logits
    ).all()

    print(
        "[PASS] Finite GAN-output test"
    )


# =============================================================================
# REAL-WORLD APPLICATION
# =============================================================================

def real_world_application(
    generated_images
):
    """
    Explain how adversarial generation connects to practical vision systems.
    """

    print()
    print("=" * 80)
    print("REAL-WORLD APPLICATION")
    print("=" * 80)

    print(
        """
GAN-style models can generate synthetic visual data.

For a robot perception pipeline:

        REAL ROBOT DATA
              |
              v
        training dataset
              |
              v
          GENERATOR
              |
              v
        synthetic scenes
              |
              v
        perception training

Potential applications include:

    - lighting variation
    - object appearance variation
    - background variation
    - synthetic manipulation scenes
    - domain adaptation
    - image translation
    - data augmentation

However:

    synthetic data != automatically useful data

A generated image may look visually convincing while violating physical
constraints.

For robotics, useful evaluation should eventually include:

    1. visual realism
    2. physical plausibility
    3. task correctness
    4. diversity
    5. downstream robot-policy performance

The generated images in this lesson are intentionally simple.

The important real-world result is learning the mechanism:

        random latent vector
                |
                v
            generator
                |
                v
          learned data distribution

rather than merely reconstructing an existing image.
"""
    )

    if torch.is_tensor(
        generated_images
    ):

        generated_images = (
            generated_images
            .detach()
            .cpu()
            .numpy()
        )

    print(
        f"Generated batch shape: "
        f"{generated_images.shape}"
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
    print("STEP 1 — UNIT TESTS")
    print("=" * 80)

    test_generator_shape()

    test_generator_range()

    test_discriminator_shape()

    test_bce_behavior()

    test_gan_gradient_flow()

    # =========================================================================
    # STEP 2 — CREATE DATASET
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 2 — CREATE REAL DATASET")
    print("=" * 80)

    images, labels = create_dataset(
        TRAIN_SAMPLES
    )

    print(
        f"Dataset shape: {images.shape}"
    )

    print(
        f"Label shape  : {labels.shape}"
    )

    visualize_real_dataset(
        images,
        labels
    )

    # =========================================================================
    # STEP 3 — DATA LOADER
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 3 — CREATE DATALOADER")
    print("=" * 80)

    dataset = GANDataset(
        images
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    sample_batch = next(
        iter(loader)
    )

    print(
        f"Batch shape: {sample_batch.shape}"
    )

    print(
        f"Batch min  : {sample_batch.min().item():.4f}"
    )

    print(
        f"Batch max  : {sample_batch.max().item():.4f}"
    )

    assert sample_batch.min() >= -1.0

    assert sample_batch.max() <= 1.0

    # =========================================================================
    # STEP 4 — CREATE MODELS
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 4 — CREATE GENERATOR AND DISCRIMINATOR")
    print("=" * 80)

    generator = Generator(
        LATENT_DIM
    ).to(DEVICE)

    discriminator = Discriminator().to(
        DEVICE
    )

    generator.apply(
        initialize_weights
    )

    discriminator.apply(
        initialize_weights
    )

    print(
        f"Generator parameters     : "
        f"{count_parameters(generator):,}"
    )

    print(
        f"Discriminator parameters : "
        f"{count_parameters(discriminator):,}"
    )

    # =========================================================================
    # STEP 5 — INITIAL GENERATION
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 5 — INITIAL GENERATION")
    print("=" * 80)

    initial_images = generate_images(
        generator,
        NUM_GENERATED
    )

    test_finite_outputs(
        generator,
        discriminator
    )

    show_image_grid(
        initial_images,
        "Generator Before Training",
        nrow=4
    )

    # =========================================================================
    # STEP 6 — TRAIN GAN
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 6 — TRAIN GAN")
    print("=" * 80)

    history, checkpoints, fixed_noise = train_gan(
        generator,
        discriminator,
        loader
    )

    # =========================================================================
    # STEP 7 — FINAL GENERATION
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 7 — FINAL GENERATION")
    print("=" * 80)

    final_images = generate_images(
        generator,
        NUM_GENERATED
    )

    test_finite_outputs(
        generator,
        discriminator
    )

    show_image_grid(
        final_images,
        "GAN Generated Images",
        nrow=4
    )

    # =========================================================================
    # STEP 8 — FIXED-NOISE GENERATION
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 8 — FIXED-NOISE GENERATION")
    print("=" * 80)

    with torch.no_grad():

        fixed_images = generator(
            fixed_noise
        ).cpu()

    show_image_grid(
        fixed_images,
        "Final Fixed-Noise Samples",
        nrow=4
    )

    # =========================================================================
    # STEP 9 — TRAINING CURVES
    # =========================================================================

    plot_training_history(
        history
    )

    plot_discriminator_history(
        history
    )

    # =========================================================================
    # STEP 10 — TRAINING PROGRESS
    # =========================================================================

    visualize_training_progress(
        checkpoints
    )

    # =========================================================================
    # STEP 11 — DISCRIMINATOR ANALYSIS
    # =========================================================================

    real_probabilities, fake_probabilities = (
        analyze_discriminator(
            generator,
            discriminator,
            images
        )
    )

    assert np.isfinite(
        real_probabilities
    ).all()

    assert np.isfinite(
        fake_probabilities
    ).all()

    assert np.all(
        real_probabilities >= 0.0
    )

    assert np.all(
        real_probabilities <= 1.0
    )

    assert np.all(
        fake_probabilities >= 0.0
    )

    assert np.all(
        fake_probabilities <= 1.0
    )

    # =========================================================================
    # STEP 12 — LATENT INTERPOLATION
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 12 — LATENT INTERPOLATION")
    print("=" * 80)

    interpolation = latent_interpolation(
        generator,
        steps=12
    )

    assert interpolation.shape == (
        12,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    # =========================================================================
    # STEP 13 — DIVERSITY ANALYSIS
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 13 — GENERATED DIVERSITY")
    print("=" * 80)

    large_sample = generate_images(
        generator,
        num_images=64
    )

    simple_generated_mode_analysis(
        large_sample
    )

    # =========================================================================
    # STEP 14 — REAL-WORLD APPLICATION
    # =========================================================================

    real_world_application(
        final_images
    )

    # =========================================================================
    # STEP 15 — FINAL TESTS
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 15 — FINAL VERIFICATION")
    print("=" * 80)

    assert len(
        history["g_loss"]
    ) == EPOCHS

    assert len(
        history["d_loss"]
    ) == EPOCHS

    assert len(
        history["d_real"]
    ) == EPOCHS

    assert len(
        history["d_fake"]
    ) == EPOCHS

    assert np.isfinite(
        np.asarray(
            history["g_loss"]
        )
    ).all()

    assert np.isfinite(
        np.asarray(
            history["d_loss"]
        )
    ).all()

    assert np.isfinite(
        np.asarray(
            history["d_real"]
        )
    ).all()

    assert np.isfinite(
        np.asarray(
            history["d_fake"]
        )
    ).all()

    assert torch.isfinite(
        final_images
    ).all()

    assert final_images.shape == (
        NUM_GENERATED,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert torch.all(
        final_images >= -1.0
    )

    assert torch.all(
        final_images <= 1.0
    )

    print(
        "[PASS] Generator output verification"
    )

    print(
        "[PASS] Discriminator output verification"
    )

    print(
        "[PASS] GAN loss-history verification"
    )

    print(
        "[PASS] Generated-image finite-value verification"
    )

    print(
        "[PASS] Latent interpolation verification"
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
You have now implemented a complete GAN.

The essential architecture is:

              z ~ N(0,I)
                   |
                   v
              GENERATOR
                   |
                   v
                x_fake
                   |
                   v
             DISCRIMINATOR
                   ^
                   |
                x_real

The discriminator learns:

    real -> 1
    fake -> 0

The generator learns:

    fake -> discriminator predicts 1

The most important equations are:

    x_fake = G(z)

    L_D =
        BCE(D(real), 1)
        +
        BCE(D(fake), 0)

    L_G =
        BCE(D(G(z)), 1)

You have also seen why GANs are fundamentally different from VAEs.

VAE:

    reconstruct and regularize

GAN:

    compete and fool

The GAN therefore introduces an important generative-learning principle:

    LEARN THE DATA DISTRIBUTION THROUGH A COMPETITIVE OBJECTIVE.
"""
    )

    # =========================================================================
    # CONNECTION TO LESSON 12
    # =========================================================================

    print()
    print("=" * 80)
    print("CONNECTION TO LESSON 12")
    print("=" * 80)

    print(
        """
LESSON 12 will introduce ATTENTION.

CNNs answer:

    "Which nearby pixels should influence this feature?"

GANs answer:

    "Can a generator produce images that fool a learned critic?"

Attention asks:

    "Which elements of the input should this element pay attention to?"

The core attention operation is:

    Attention(Q,K,V)

        =
    softmax(
        QK^T / sqrt(d_k)
    ) V

We will build this operation from scratch.

Then we will progress through:

    self-attention
        |
        v
    multi-head attention
        |
        v
    transformer blocks
        |
        v
    Vision Transformer
        |
        v
    cross-attention
        |
        v
    text conditioning
        |
        v
    diffusion conditioning

This is the bridge from classical CNN-based generative models toward
modern text-conditioned image generation.
"""
    )

    print()
    print("=" * 80)
    print("END OF LESSON 11")
    print("=" * 80)


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()