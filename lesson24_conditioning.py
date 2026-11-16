"""
================================================================================
LESSON 24 — CONDITIONING FOR LATENT DIFFUSION
================================================================================

PURPOSE
-------
Learn how to condition a latent diffusion model so that generation depends on
an explicit semantic condition.

Lesson 23 learned latent diffusion:

    epsilon_theta(z_t, t)

This lesson adds semantic conditioning:

    epsilon_theta(z_t, t, c)

where:

    z_t = noisy latent
    t   = diffusion timestep
    c   = semantic condition

The condition is represented by a learned embedding.

NO PRETRAINED MODELS.
NO PRETRAINED WEIGHTS.
NO PRETRAINED TEXT ENCODER.

Everything is initialized and trained from scratch.


WHAT YOU WILL LEARN
-------------------
1. Why unconditional diffusion cannot directly control semantic content.
2. What conditional diffusion means.
3. Learned condition embeddings.
4. Timestep embeddings.
5. Conditional residual blocks.
6. Conditional latent diffusion.
7. Condition-sensitive generation.
8. Same-noise / different-condition experiments.
9. Device-safe conditional sampling.
10. Why simple vector conditioning is limited.
11. Why cross-attention is the next step.


TECHNICAL / COURSE ROADMAP
--------------------------
Lesson 23:
    Latent Diffusion

Lesson 24:
    Conditioning

Lesson 25:
    Cross-Attention

Lesson 26:
    Text Conditioning

Lesson 27:
    Own Text-to-Image Model


REAL-WORLD PROBLEM
------------------
A robot camera may observe many possible objects:

    bottle
    cup
    can
    box

An unconditional generative model does not receive an explicit semantic
instruction.

A conditional model can receive:

    condition = bottle

or:

    condition = cup

and generate according to that condition.

In robotics this idea can later support:

    synthetic data generation
    object-category variation
    visual robustness testing
    simulation variation
    perception training
    manipulation-policy training


THEORY
------
Forward diffusion:

    z_t =
        sqrt(alpha_bar_t) z_0
        +
        sqrt(1-alpha_bar_t) epsilon

The network learns:

    epsilon_theta(z_t, t, c)

The condition is embedded:

    c_emb = E_condition(c)

The timestep is embedded:

    t_emb = E_time(t)

The embeddings are projected into residual blocks:

    h' =
        h
        +
        W_t(t_emb)
        +
        W_c(c_emb)


CONDITIONAL U-NET
-----------------
The network used here is:

    latent
       |
       v
    input
       |
       v
    residual block
       |
       +----------------------+
       |                      |
       v                      |
    downsample                |
       |                      |
       v                      |
    residual block            |
       |                      |
       v                      |
    bottleneck                |
       |                      |
       v                      |
    upsample                  |
       |                      |
       +---- projected skip --+
       |
       v
    residual block
       |
       v
    predicted noise


IMPORTANT ARCHITECTURAL DETAIL
------------------------------
The decoder feature map and skip feature map do not necessarily have the
same number of channels.

In this implementation:

    h1 = 32 channels

while after the bottleneck and upsampling:

    h4 = 64 channels

Therefore this is INVALID:

    h4 + h1

because:

    [B,64,H,W] + [B,32,H,W]

has incompatible channel dimensions.

We solve this with a learned 1x1 projection:

    skip_projection:
        32 -> 64

Then:

    h4 + skip_projection(h1)

is valid:

    [B,64,H,W] + [B,64,H,W]

This is a standard and useful U-Net design pattern.


WHY THIS MATTERS
----------------
Tensor shape correctness is part of neural-network architecture design.

A network can have:

    correct spatial resolution
    correct batch size
    correct tensor dtype

and still fail because the channel dimensions are incompatible.

This lesson therefore explicitly tests the conditional U-Net architecture.


MATHEMATICS
-----------
Training objective:

    L =
        E[
            ||epsilon -
            epsilon_theta(z_t,t,c)||^2
        ]


Condition embedding:

    c_emb = Embedding(c)


Timestep embedding:

    t_emb = E(t)


Residual conditioning:

    h_conditioned =
        h
        +
        W_t(t_emb)
        +
        W_c(c_emb)


DDPM reverse mean:

    mu_theta =
        1/sqrt(alpha_t)
        [
            z_t
            -
            beta_t/sqrt(1-alpha_bar_t)
            epsilon_theta(z_t,t,c)
        ]


CONDITION SENSITIVITY
---------------------
We generate using the SAME initial noise:

    z_T

but different conditions:

    c_1
    c_2
    c_3
    c_4

Therefore:

    Sample(z_T,c_1)
    !=
    Sample(z_T,c_2)

if the model has learned useful conditional information.


IMPLEMENTATION
--------------
This script contains:

1. Synthetic structured image dataset.
2. Four semantic classes.
3. Small convolutional autoencoder.
4. Latent representation.
5. Diffusion schedule.
6. Timestep embedding.
7. Condition embedding.
8. Conditional residual blocks.
9. Conditional latent diffusion U-Net.
10. Proper channel-compatible skip projection.
11. DDPM training.
12. DDPM sampling.
13. Numerical tests.
14. Visualization.
15. Same-noise/different-condition experiment.
16. Device synchronization.


VERIFICATION / TESTS
--------------------
Tests verify:

1. GroupNorm.
2. Dataset-condition alignment.
3. Autoencoder shapes.
4. Diffusion schedule.
5. Timestep embedding.
6. Condition embedding.
7. Distinct condition embeddings.
8. Conditional U-Net output.
9. Condition changes model output.
10. Forward diffusion.
11. Conditional gradient flow.
12. Device consistency.
13. One-step conditional diffusion.
14. Complete conditional sampler.
15. Sampler device synchronization.


DEVICE DESIGN
-------------
The preferred global device is:

    DEVICE

but the actual model device is always obtained with:

    next(model.parameters()).device

The actual model device is authoritative.

Therefore:

    model
    autoencoder
    schedule
    labels
    timesteps
    latent tensors

are synchronized to the actual model device.


VISUALIZATION
-------------
The script visualizes:

1. Synthetic examples.
2. Autoencoder reconstructions.
3. Conditional generated images.
4. Same-noise/different-condition results.
5. Training curves.


EXPERIMENT
----------
The central experiment is:

    same initial latent noise
    +
    different semantic condition
    =
    different generated image


REAL-WORLD APPLICATION
----------------------
For a robot waste-sorting system:

    condition 0 -> bottle
    condition 1 -> cup
    condition 2 -> can
    condition 3 -> box

A conditional generator can create category-specific visual variations.

This can later help generate difficult synthetic scenes for robotic perception.


REAL-WORLD RESULT
-----------------
The important result is controllability.

Changing the semantic condition should influence the generated result even when
the initial random noise is held fixed.


LESSON RESULT
-------------
After this lesson the model has:

    latent diffusion
    +
    semantic conditioning
    +
    conditional residual blocks
    +
    condition-sensitive generation

The limitation is that the entire condition is still compressed into one
condition vector.

Lesson 25 solves this with cross-attention.


CONNECTION TO NEXT LESSON
-------------------------
Lesson 25 — Cross-Attention

We will learn:

    Query
    Key
    Value

with:

    Attention(Q,K,V)
        =
    softmax(QK^T / sqrt(d_k))V

Image latent tokens become Queries.

Condition tokens become Keys and Values.

This provides much richer interaction between visual features and semantic
information.

================================================================================
"""

import math
import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 32
CHANNELS = 1
NUM_CLASSES = 4

LATENT_CHANNELS = 4
LATENT_SIZE = 8

AE_BATCH_SIZE = 64
AE_TRAIN_STEPS = 900
AE_LEARNING_RATE = 2e-3

T = 1000
BETA_START = 1e-4
BETA_END = 0.02

DIFFUSION_BATCH_SIZE = 64
DIFFUSION_TRAIN_STEPS = 1400
DIFFUSION_LEARNING_RATE = 2e-3

BASE_CHANNELS = 32
TIME_EMBED_DIM = 128
CONDITION_EMBED_DIM = 128

NUM_SAMPLE_IMAGES = 8
DISPLAY_SAMPLES_PER_CLASS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DEVICE = torch.device("cpu")

EPS = 1e-8

print("=" * 80)
print("LESSON 24 — CONDITIONING FOR LATENT DIFFUSION")
print("=" * 80)
print(f"PyTorch version : {torch.__version__}")
print(f"Preferred device: {DEVICE}")
print(f"Test device     : {TEST_DEVICE}")
print(f"Image size      : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(
    f"Latent size     : "
    f"{LATENT_CHANNELS}x{LATENT_SIZE}x{LATENT_SIZE}"
)
print(f"Classes         : {NUM_CLASSES}")
print(f"Diffusion T     : {T}")
print("=" * 80)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# =============================================================================
# DEVICE UTILITIES
# =============================================================================

def get_model_device(
    model: nn.Module,
) -> torch.device:
    """
    Return the actual device containing the model parameters.

    The actual model device is authoritative.
    """
    return next(model.parameters()).device


def move_schedule_to_device(
    schedule: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:

    return {
        key: value.to(device)
        for key, value in schedule.items()
    }


def synchronize_model_and_schedule(
    model: nn.Module,
    schedule: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:

    model_device = get_model_device(model)

    return move_schedule_to_device(
        schedule,
        model_device,
    )


# =============================================================================
# GROUPNORM UTILITY
# =============================================================================

def valid_group_count(
    num_channels: int,
    preferred_groups: int = 8,
) -> int:

    num_channels = int(num_channels)
    preferred_groups = int(preferred_groups)

    if num_channels <= 0:
        raise ValueError(
            "num_channels must be positive."
        )

    preferred_groups = min(
        preferred_groups,
        num_channels,
    )

    for groups in range(
        preferred_groups,
        0,
        -1,
    ):
        if num_channels % groups == 0:
            return groups

    return 1


# =============================================================================
# SYNTHETIC DATASET
# =============================================================================

CLASS_NAMES = (
    "circle",
    "square",
    "diagonal",
    "cross",
)


def draw_circle(
    size: int = IMAGE_SIZE,
) -> torch.Tensor:

    yy, xx = torch.meshgrid(
        torch.linspace(
            -1.0,
            1.0,
            size,
        ),
        torch.linspace(
            -1.0,
            1.0,
            size,
        ),
        indexing="ij",
    )

    radius = random.uniform(
        0.25,
        0.50,
    )

    cx = random.uniform(
        -0.15,
        0.15,
    )

    cy = random.uniform(
        -0.15,
        0.15,
    )

    distance = torch.sqrt(
        (xx - cx) ** 2
        +
        (yy - cy) ** 2
    )

    image = (
        distance < radius
    ).float()

    return image.unsqueeze(0)


def draw_square(
    size: int = IMAGE_SIZE,
) -> torch.Tensor:

    image = torch.zeros(
        1,
        size,
        size,
    )

    side = random.randint(
        size // 3,
        size // 2,
    )

    x0 = random.randint(
        2,
        size - side - 2,
    )

    y0 = random.randint(
        2,
        size - side - 2,
    )

    image[
        :,
        y0:y0 + side,
        x0:x0 + side,
    ] = 1.0

    return image


def draw_diagonal(
    size: int = IMAGE_SIZE,
) -> torch.Tensor:

    image = torch.zeros(
        1,
        size,
        size,
    )

    thickness = random.choice(
        [1, 2]
    )

    for i in range(size):

        j = i + random.randint(
            -2,
            2,
        )

        if 0 <= j < size:

            image[
                :,
                max(
                    0,
                    j - thickness,
                ):min(
                    size,
                    j + thickness + 1,
                ),
                max(
                    0,
                    i - thickness,
                ):min(
                    size,
                    i + thickness + 1,
                ),
            ] = 1.0

    return image


def draw_cross(
    size: int = IMAGE_SIZE,
) -> torch.Tensor:

    image = torch.zeros(
        1,
        size,
        size,
    )

    center = (
        size // 2
        +
        random.randint(
            -2,
            2,
        )
    )

    thickness = random.choice(
        [1, 2, 3]
    )

    image[
        :,
        max(
            0,
            center - thickness,
        ):min(
            size,
            center + thickness + 1,
        ),
        :,
    ] = 1.0

    image[
        :,
        :,
        max(
            0,
            center - thickness,
        ):min(
            size,
            center + thickness + 1,
        ),
    ] = 1.0

    return image


def generate_class_image(
    class_id: int,
) -> torch.Tensor:

    if class_id == 0:
        image = draw_circle()

    elif class_id == 1:
        image = draw_square()

    elif class_id == 2:
        image = draw_diagonal()

    elif class_id == 3:
        image = draw_cross()

    else:
        raise ValueError(
            f"Unknown class: {class_id}"
        )

    image = (
        image
        +
        0.03 * torch.randn_like(image)
    )

    return image.clamp(
        0.0,
        1.0,
    )


class ConditionalShapesDataset(
    Dataset
):

    def __init__(
        self,
        length: int = 100000,
    ):
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        class_id = (
            index % NUM_CLASSES
        )

        image = generate_class_image(
            class_id
        )

        label = torch.tensor(
            class_id,
            dtype=torch.long,
        )

        return image, label


# =============================================================================
# AUTOENCODER
# =============================================================================

class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int = CHANNELS,
        latent_channels: int = LATENT_CHANNELS,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.SiLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.SiLU(),

            nn.Conv2d(
                64,
                latent_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(x)


class Decoder(nn.Module):

    def __init__(
        self,
        latent_channels: int = LATENT_CHANNELS,
        out_channels: int = CHANNELS,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                latent_channels,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.SiLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.SiLU(),

            nn.ConvTranspose2d(
                32,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(z)


class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.encoder(x)

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        z = self.encode(x)

        reconstruction = self.decode(z)

        return reconstruction, z


# =============================================================================
# DIFFUSION SCHEDULE
# =============================================================================

def make_diffusion_schedule(
    num_steps: int = T,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
    device: torch.device = TEST_DEVICE,
) -> Dict[str, torch.Tensor]:

    betas = torch.linspace(
        beta_start,
        beta_end,
        num_steps,
        device=device,
        dtype=torch.float32,
    )

    alphas = (
        1.0
        -
        betas
    )

    alpha_bars = torch.cumprod(
        alphas,
        dim=0,
    )

    sqrt_alphas = torch.sqrt(
        alphas
    )

    sqrt_alpha_bars = torch.sqrt(
        alpha_bars
    )

    sqrt_one_minus_alpha_bars = torch.sqrt(
        1.0
        -
        alpha_bars
    )

    alpha_bar_previous = torch.cat(
        [
            torch.ones(
                1,
                device=device,
            ),
            alpha_bars[:-1],
        ]
    )

    posterior_variance = (
        betas
        *
        (
            1.0
            -
            alpha_bar_previous
        )
        /
        (
            1.0
            -
            alpha_bars
        )
    )

    posterior_variance[0] = 0.0

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alphas": sqrt_alphas,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars":
            sqrt_one_minus_alpha_bars,
        "posterior_variance":
            posterior_variance,
    }


def extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    shape: Tuple[int, ...],
) -> torch.Tensor:

    if values.device != timesteps.device:
        raise RuntimeError(
            "extract device mismatch: "
            f"values={values.device}, "
            f"timesteps={timesteps.device}"
        )

    batch_size = (
        timesteps.shape[0]
    )

    output = values.gather(
        0,
        timesteps.long(),
    )

    return output.reshape(
        batch_size,
        *(
            [1]
            *
            (
                len(shape) - 1
            )
        ),
    )


# =============================================================================
# TIME EMBEDDING
# =============================================================================

def timestep_embedding(
    timesteps: torch.Tensor,
    dimension: int = TIME_EMBED_DIM,
) -> torch.Tensor:

    timesteps = timesteps.float()

    half = dimension // 2

    frequencies = torch.exp(
        -math.log(10000.0)
        *
        torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        )
        /
        max(
            half - 1,
            1,
        )
    )

    arguments = (
        timesteps[:, None]
        *
        frequencies[None, :]
    )

    embedding = torch.cat(
        [
            torch.sin(arguments),
            torch.cos(arguments),
        ],
        dim=-1,
    )

    if dimension % 2 == 1:

        embedding = F.pad(
            embedding,
            (0, 1),
        )

    return embedding


# =============================================================================
# CONDITIONAL RESIDUAL BLOCK
# =============================================================================

class ConditionalResidualBlock(
    nn.Module
):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int = TIME_EMBED_DIM,
        condition_dim: int = CONDITION_EMBED_DIM,
    ):
        super().__init__()

        groups_in = valid_group_count(
            in_channels
        )

        groups_out = valid_group_count(
            out_channels
        )

        self.norm1 = nn.GroupNorm(
            groups_in,
            in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Linear(
            time_dim,
            out_channels,
        )

        self.condition_projection = nn.Linear(
            condition_dim,
            out_channels,
        )

        self.norm2 = nn.GroupNorm(
            groups_out,
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if (
            in_channels
            !=
            out_channels
        ):

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

        else:

            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        condition_emb: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.skip(x)

        h = self.norm1(x)

        h = F.silu(h)

        h = self.conv1(h)

        time_bias = (
            self.time_projection(
                time_emb
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        condition_bias = (
            self.condition_projection(
                condition_emb
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        h = (
            h
            +
            time_bias
            +
            condition_bias
        )

        h = self.norm2(h)

        h = F.silu(h)

        h = self.conv2(h)

        return (
            h
            +
            residual
        )


# =============================================================================
# CONDITIONAL LATENT DIFFUSION U-NET
# =============================================================================

class ConditionalLatentUNet(
    nn.Module
):
    """
    Conditional latent diffusion U-Net.

    IMPORTANT CHANNEL FLOW
    ----------------------

    Input:
        4 channels

    input_conv:
        4 -> 32

    h1:
        32 channels
        spatial size 8x8

    down:
        32 -> 64
        spatial size 8x8 -> 4x4

    h2:
        64 channels
        spatial size 4x4

    bottleneck:
        64 -> 128

    up:
        128 -> 64
        spatial size 4x4 -> 8x8

    h4:
        64 channels
        spatial size 8x8

    h1:
        32 channels
        spatial size 8x8

    Therefore:

        h4 + h1

    is INVALID.

    We use:

        skip_projection(h1)

    where:

        32 -> 64

    Then:

        h4 + skip_projection(h1)

    is valid.
    """

    def __init__(
        self,
        latent_channels: int = LATENT_CHANNELS,
        base_channels: int = BASE_CHANNELS,
        time_dim: int = TIME_EMBED_DIM,
        condition_dim: int = CONDITION_EMBED_DIM,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.condition_embedding = (
            nn.Embedding(
                num_classes,
                condition_dim,
            )
        )

        self.time_mlp = nn.Sequential(

            nn.Linear(
                time_dim,
                time_dim,
            ),

            nn.SiLU(),

            nn.Linear(
                time_dim,
                time_dim,
            ),
        )

        self.input_conv = nn.Conv2d(
            latent_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.block1 = (
            ConditionalResidualBlock(
                base_channels,
                base_channels,
                time_dim,
                condition_dim,
            )
        )

        self.down = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.block2 = (
            ConditionalResidualBlock(
                base_channels * 2,
                base_channels * 2,
                time_dim,
                condition_dim,
            )
        )

        self.bottleneck = (
            ConditionalResidualBlock(
                base_channels * 2,
                base_channels * 4,
                time_dim,
                condition_dim,
            )
        )

        self.up = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # ---------------------------------------------------------------------
        # IMPORTANT FIX
        #
        # h1 has BASE_CHANNELS channels.
        #
        # h4 has BASE_CHANNELS * 2 channels.
        #
        # Therefore project h1 before adding it to h4.
        #
        # 32 -> 64
        # ---------------------------------------------------------------------

        self.skip_projection = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=1,
        )

        self.block3 = (
            ConditionalResidualBlock(
                base_channels * 2,
                base_channels,
                time_dim,
                condition_dim,
            )
        )

        self.output_norm = nn.GroupNorm(
            valid_group_count(
                base_channels
            ),
            base_channels,
        )

        self.output_conv = nn.Conv2d(
            base_channels,
            latent_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        time_features = timestep_embedding(
            timesteps,
            TIME_EMBED_DIM,
        )

        time_features = self.time_mlp(
            time_features
        )

        condition_features = (
            self.condition_embedding(
                labels
            )
        )

        h0 = self.input_conv(z)

        h1 = self.block1(
            h0,
            time_features,
            condition_features,
        )

        h2 = self.down(h1)

        h2 = self.block2(
            h2,
            time_features,
            condition_features,
        )

        h3 = self.bottleneck(
            h2,
            time_features,
            condition_features,
        )

        h4 = self.up(h3)

        # ---------------------------------------------------------------------
        # CORRECTED SKIP CONNECTION
        #
        # Before:
        #
        #     h4 = h4 + h1
        #
        # which attempted:
        #
        #     [B,64,8,8] + [B,32,8,8]
        #
        # Now:
        #
        #     projected_skip = [B,64,8,8]
        #
        #     h4 + projected_skip
        #
        # ---------------------------------------------------------------------

        projected_skip = (
            self.skip_projection(h1)
        )

        assert h4.shape == projected_skip.shape, (
            "Conditional U-Net skip connection "
            "shape mismatch: "
            f"h4={tuple(h4.shape)}, "
            f"projected_skip="
            f"{tuple(projected_skip.shape)}"
        )

        h4 = (
            h4
            +
            projected_skip
        )

        h5 = self.block3(
            h4,
            time_features,
            condition_features,
        )

        h5 = self.output_norm(h5)

        h5 = F.silu(h5)

        return self.output_conv(h5)


# =============================================================================
# FORWARD DIFFUSION
# =============================================================================

def latent_forward_diffusion(
    z0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
) -> torch.Tensor:

    sqrt_alpha_bar = extract(
        schedule[
            "sqrt_alpha_bars"
        ],
        timesteps,
        z0.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule[
            "sqrt_one_minus_alpha_bars"
        ],
        timesteps,
        z0.shape,
    )

    return (
        sqrt_alpha_bar * z0
        +
        sqrt_one_minus_alpha_bar * noise
    )


# =============================================================================
# REVERSE DIFFUSION
# =============================================================================

@torch.no_grad()
def conditional_ddpm_step(
    model: ConditionalLatentUNet,
    z_t: torch.Tensor,
    timestep: int,
    labels: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
) -> torch.Tensor:

    model_device = get_model_device(
        model
    )

    schedule = (
        synchronize_model_and_schedule(
            model,
            schedule,
        )
    )

    z_t = z_t.to(
        model_device
    )

    labels = labels.to(
        model_device,
        dtype=torch.long,
    )

    t = torch.full(
        (z_t.shape[0],),
        timestep,
        device=model_device,
        dtype=torch.long,
    )

    predicted_noise = model(
        z_t,
        t,
        labels,
    )

    beta_t = extract(
        schedule["betas"],
        t,
        z_t.shape,
    )

    alpha_t = extract(
        schedule["alphas"],
        t,
        z_t.shape,
    )

    alpha_bar_t = extract(
        schedule["alpha_bars"],
        t,
        z_t.shape,
    )

    mean = (
        1.0
        /
        torch.sqrt(alpha_t)
    ) * (
        z_t
        -
        (
            beta_t
            /
            torch.sqrt(
                1.0
                -
                alpha_bar_t
            )
        )
        *
        predicted_noise
    )

    if timestep == 0:
        return mean

    variance = extract(
        schedule[
            "posterior_variance"
        ],
        t,
        z_t.shape,
    )

    noise = torch.randn_like(
        z_t
    )

    return (
        mean
        +
        torch.sqrt(
            variance.clamp_min(
                0.0
            )
        )
        *
        noise
    )


@torch.no_grad()
def sample_conditional_latent_diffusion(
    model: ConditionalLatentUNet,
    schedule: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    latent_shape: Tuple[
        int,
        int,
        int,
        int,
    ],
    initial_noise: torch.Tensor = None,
    return_trajectory: bool = True,
) -> Tuple[
    torch.Tensor,
    Dict[int, torch.Tensor],
]:

    model_device = get_model_device(
        model
    )

    schedule = (
        synchronize_model_and_schedule(
            model,
            schedule,
        )
    )

    labels = labels.to(
        model_device,
        dtype=torch.long,
    )

    if initial_noise is None:

        z = torch.randn(
            latent_shape,
            device=model_device,
        )

    else:

        z = initial_noise.to(
            model_device
        ).clone()

    trajectory = {}

    checkpoints = {
        T - 1,
        750,
        500,
        250,
        100,
        50,
        25,
        10,
        1,
        0,
    }

    for timestep in reversed(
        range(T)
    ):

        if (
            return_trajectory
            and
            timestep in checkpoints
        ):

            trajectory[
                timestep
            ] = z.detach().clone()

        z = conditional_ddpm_step(
            model=model,
            z_t=z,
            timestep=timestep,
            labels=labels,
            schedule=schedule,
        )

    if (
        return_trajectory
        and
        0 not in trajectory
    ):

        trajectory[0] = (
            z.detach().clone()
        )

    return z, trajectory


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def infinite_loader(
    loader: DataLoader,
):

    while True:

        for batch in loader:
            yield batch


def train_autoencoder(
    autoencoder: Autoencoder,
    loader: DataLoader,
    device: torch.device,
    steps: int = AE_TRAIN_STEPS,
) -> List[float]:

    autoencoder = autoencoder.to(
        device
    )

    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=AE_LEARNING_RATE,
    )

    iterator = infinite_loader(
        loader
    )

    losses = []

    autoencoder.train()

    start_time = time.time()

    for step in range(
        1,
        steps + 1,
    ):

        images, _ = next(
            iterator
        )

        images = images.to(
            device
        )

        reconstruction, _ = (
            autoencoder(images)
        )

        loss = F.mse_loss(
            reconstruction,
            images,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        losses.append(
            float(
                loss.detach().cpu()
            )
        )

        if (
            step == 1
            or
            step % 200 == 0
            or
            step == steps
        ):

            print(
                f"AE step {step:4d}/{steps} "
                f"| loss = {loss.item():.6f}"
            )

    elapsed = (
        time.time()
        -
        start_time
    )

    print(
        f"Autoencoder training time: "
        f"{elapsed:.2f} sec"
    )

    return losses


def train_conditional_diffusion(
    model: ConditionalLatentUNet,
    autoencoder: Autoencoder,
    loader: DataLoader,
    schedule: Dict[str, torch.Tensor],
    device: torch.device,
    steps: int = DIFFUSION_TRAIN_STEPS,
) -> List[float]:

    model = model.to(
        device
    )

    autoencoder = autoencoder.to(
        device
    )

    autoencoder.eval()

    for parameter in (
        autoencoder.parameters()
    ):

        parameter.requires_grad_(
            False
        )

    schedule = move_schedule_to_device(
        schedule,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DIFFUSION_LEARNING_RATE,
    )

    iterator = infinite_loader(
        loader
    )

    losses = []

    model.train()

    start_time = time.time()

    for step in range(
        1,
        steps + 1,
    ):

        images, labels = next(
            iterator
        )

        images = images.to(
            device
        )

        labels = labels.to(
            device,
            dtype=torch.long,
        )

        with torch.no_grad():

            z0 = autoencoder.encode(
                images
            )

        noise = torch.randn_like(
            z0
        )

        timesteps = torch.randint(
            0,
            T,
            (
                z0.shape[0],
            ),
            device=device,
            dtype=torch.long,
        )

        z_t = latent_forward_diffusion(
            z0,
            timesteps,
            noise,
            schedule,
        )

        predicted_noise = model(
            z_t,
            timesteps,
            labels,
        )

        loss = F.mse_loss(
            predicted_noise,
            noise,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        losses.append(
            float(
                loss.detach().cpu()
            )
        )

        if (
            step == 1
            or
            step % 200 == 0
            or
            step == steps
        ):

            print(
                f"Diffusion step {step:4d}/{steps} "
                f"| loss = {loss.item():.6f}"
            )

    elapsed = (
        time.time()
        -
        start_time
    )

    print(
        f"Conditional diffusion training time: "
        f"{elapsed:.2f} sec"
    )

    return losses


# =============================================================================
# VISUALIZATION
# =============================================================================

def show_dataset_examples(
    dataset: Dataset,
    count: int = 8,
) -> None:

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(10, 5),
    )

    for index, ax in enumerate(
        axes.flat
    ):

        image, label = dataset[
            index
        ]

        ax.imshow(
            image.squeeze(0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(
            f"class={label.item()} "
            f"{CLASS_NAMES[label.item()]}"
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_reconstructions(
    autoencoder: Autoencoder,
    dataset: Dataset,
    device: torch.device,
    count: int = 8,
) -> None:

    autoencoder.eval()

    images = []
    labels = []

    for index in range(count):

        image, label = dataset[
            index
        ]

        images.append(image)
        labels.append(label)

    batch = torch.stack(
        images
    ).to(device)

    with torch.no_grad():

        reconstruction, _ = (
            autoencoder(batch)
        )

    reconstruction = (
        reconstruction
        .detach()
        .cpu()
    )

    batch = batch.cpu()

    fig, axes = plt.subplots(
        2,
        count,
        figsize=(
            2 * count,
            4,
        ),
    )

    for index in range(count):

        axes[
            0,
            index
        ].imshow(
            batch[index, 0],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        axes[
            0,
            index
        ].set_title(
            CLASS_NAMES[
                labels[index].item()
            ]
        )

        axes[
            0,
            index
        ].axis("off")

        axes[
            1,
            index
        ].imshow(
            reconstruction[
                index,
                0,
            ],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        axes[
            1,
            index
        ].axis("off")

    axes[0, 0].set_ylabel(
        "Original"
    )

    axes[1, 0].set_ylabel(
        "Reconstruction"
    )

    plt.tight_layout()
    plt.show()


def show_generated_samples(
    samples: torch.Tensor,
    labels: torch.Tensor,
    title: str,
) -> None:

    samples = (
        samples.detach().cpu()
    )

    labels = (
        labels.detach().cpu()
    )

    count = samples.shape[0]

    cols = min(
        4,
        count,
    )

    rows = math.ceil(
        count / cols
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            3 * cols,
            3 * rows,
        ),
        squeeze=False,
    )

    for index, ax in enumerate(
        axes.flat
    ):

        if index >= count:

            ax.axis("off")
            continue

        ax.imshow(
            samples[index, 0],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        label = int(
            labels[index].item()
        )

        ax.set_title(
            f"{label}: "
            f"{CLASS_NAMES[label]}"
        )

        ax.axis("off")

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def show_same_noise_conditions(
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> None:

    outputs = (
        outputs.detach().cpu()
    )

    labels = (
        labels.detach().cpu()
    )

    count = outputs.shape[0]

    fig, axes = plt.subplots(
        1,
        count,
        figsize=(
            3 * count,
            3,
        ),
        squeeze=False,
    )

    for index in range(count):

        axes[
            0,
            index
        ].imshow(
            outputs[
                index,
                0,
            ],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        label = int(
            labels[index].item()
        )

        axes[
            0,
            index
        ].set_title(
            f"condition={label}\n"
            f"{CLASS_NAMES[label]}"
        )

        axes[
            0,
            index
        ].axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
# TEST 1
# =============================================================================

def test_groupnorm() -> None:

    print(
        "\nTEST 1 — GROUPNORM"
    )

    for channels in [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ]:

        groups = valid_group_count(
            channels
        )

        assert (
            channels % groups
            == 0
        )

    block = ConditionalResidualBlock(
        1,
        4,
    )

    x = torch.randn(
        2,
        1,
        8,
        8,
    )

    t = torch.randn(
        2,
        TIME_EMBED_DIM,
    )

    c = torch.randn(
        2,
        CONDITION_EMBED_DIM,
    )

    y = block(
        x,
        t,
        c,
    )

    assert y.shape == (
        2,
        4,
        8,
        8,
    )

    print("PASS")


# =============================================================================
# TEST 2
# =============================================================================

def test_dataset_condition_alignment() -> None:

    print(
        "\nTEST 2 — DATASET CONDITION ALIGNMENT"
    )

    dataset = ConditionalShapesDataset(
        length=NUM_CLASSES * 4
    )

    seen = []

    for index in range(
        NUM_CLASSES * 4
    ):

        image, label = dataset[
            index
        ]

        assert image.shape == (
            CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        assert (
            0
            <= label.item()
            <
            NUM_CLASSES
        )

        seen.append(
            label.item()
        )

    for class_id in range(
        NUM_CLASSES
    ):

        assert class_id in seen

    print("PASS")


# =============================================================================
# TEST 3
# =============================================================================

def test_autoencoder_shapes() -> None:

    print(
        "\nTEST 3 — AUTOENCODER SHAPES"
    )

    model = Autoencoder().to(
        TEST_DEVICE
    )

    x = torch.randn(
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=TEST_DEVICE,
    )

    reconstruction, z = model(x)

    assert (
        reconstruction.shape
        ==
        x.shape
    )

    assert z.shape == (
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    print(
        f"Input          : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Latent         : "
        f"{tuple(z.shape)}"
    )

    print(
        f"Reconstruction : "
        f"{tuple(reconstruction.shape)}"
    )

    print("PASS")


# =============================================================================
# TEST 4
# =============================================================================

def test_diffusion_schedule() -> None:

    print(
        "\nTEST 4 — DIFFUSION SCHEDULE"
    )

    schedule = (
        make_diffusion_schedule(
            device=TEST_DEVICE
        )
    )

    assert (
        schedule["betas"].shape
        ==
        (T,)
    )

    assert torch.all(
        schedule["betas"] > 0
    )

    assert torch.all(
        schedule["betas"] < 1
    )

    assert torch.all(
        schedule["alpha_bars"] > 0
    )

    assert torch.all(
        schedule["alpha_bars"] <= 1
    )

    assert torch.all(
        schedule["alpha_bars"][1:]
        <=
        schedule["alpha_bars"][:-1]
    )

    print("PASS")


# =============================================================================
# TEST 5
# =============================================================================

def test_time_embedding() -> None:

    print(
        "\nTEST 5 — TIME EMBEDDING"
    )

    timesteps = torch.tensor(
        [
            0,
            1,
            100,
            999,
        ],
        dtype=torch.long,
    )

    embedding = (
        timestep_embedding(
            timesteps
        )
    )

    assert embedding.shape == (
        4,
        TIME_EMBED_DIM,
    )

    assert torch.isfinite(
        embedding
    ).all()

    assert not torch.allclose(
        embedding[0],
        embedding[-1],
    )

    print("PASS")


# =============================================================================
# TEST 6
# =============================================================================

def test_condition_embedding() -> None:

    print(
        "\nTEST 6 — CONDITION EMBEDDING"
    )

    model = ConditionalLatentUNet().to(
        TEST_DEVICE
    )

    labels = torch.arange(
        NUM_CLASSES,
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    embeddings = (
        model.condition_embedding(
            labels
        )
    )

    assert embeddings.shape == (
        NUM_CLASSES,
        CONDITION_EMBED_DIM,
    )

    assert torch.isfinite(
        embeddings
    ).all()

    print("PASS")


# =============================================================================
# TEST 7
# =============================================================================

def test_distinct_condition_embeddings() -> None:

    print(
        "\nTEST 7 — DISTINCT CONDITION EMBEDDINGS"
    )

    model = ConditionalLatentUNet().to(
        TEST_DEVICE
    )

    labels = torch.arange(
        NUM_CLASSES,
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    embeddings = (
        model.condition_embedding(
            labels
        )
    )

    differences = []

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            i + 1,
            NUM_CLASSES,
        ):

            differences.append(
                torch.norm(
                    embeddings[i]
                    -
                    embeddings[j]
                )
            )

    mean_difference = (
        torch.stack(
            differences
        ).mean()
    )

    print(
        f"Mean pairwise embedding "
        f"difference: "
        f"{mean_difference.item():.6f}"
    )

    assert (
        mean_difference.item()
        >
        0.0
    )

    print("PASS")


# =============================================================================
# TEST 8 — CONDITIONAL U-NET OUTPUT
# =============================================================================

def test_conditional_unet_output() -> None:

    print(
        "\nTEST 8 — CONDITIONAL U-NET OUTPUT"
    )

    model = ConditionalLatentUNet().to(
        TEST_DEVICE
    )

    model.eval()

    z = torch.randn(
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    t = torch.randint(
        0,
        T,
        (4,),
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    labels = torch.randint(
        0,
        NUM_CLASSES,
        (4,),
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    with torch.no_grad():

        output = model(
            z,
            t,
            labels,
        )

    assert output.shape == z.shape

    assert torch.isfinite(
        output
    ).all()

    # Explicit architecture checks.
    assert (
        model.skip_projection.in_channels
        ==
        BASE_CHANNELS
    )

    assert (
        model.skip_projection.out_channels
        ==
        BASE_CHANNELS * 2
    )

    print(
        f"Input shape : "
        f"{tuple(z.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        "Skip projection: "
        f"{BASE_CHANNELS} -> "
        f"{BASE_CHANNELS * 2}"
    )

    print("PASS")


# =============================================================================
# TEST 9
# =============================================================================

def test_condition_affects_model_output() -> None:

    print(
        "\nTEST 9 — CONDITION AFFECTS MODEL OUTPUT"
    )

    model = ConditionalLatentUNet().to(
        TEST_DEVICE
    )

    model.eval()

    z = torch.randn(
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    t = torch.full(
        (2,),
        500,
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    labels_a = torch.tensor(
        [0, 0],
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    labels_b = torch.tensor(
        [1, 1],
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    with torch.no_grad():

        output_a = model(
            z,
            t,
            labels_a,
        )

        output_b = model(
            z,
            t,
            labels_b,
        )

    difference = torch.mean(
        torch.abs(
            output_a
            -
            output_b
        )
    )

    print(
        f"Mean model-output difference: "
        f"{difference.item():.8f}"
    )

    assert (
        difference.item()
        >
        1e-8
    )

    print("PASS")


# =============================================================================
# TEST 10
# =============================================================================

def test_latent_forward_diffusion() -> None:

    print(
        "\nTEST 10 — LATENT FORWARD DIFFUSION"
    )

    schedule = (
        make_diffusion_schedule(
            device=TEST_DEVICE
        )
    )

    z0 = torch.randn(
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    noise = torch.randn_like(
        z0
    )

    timesteps = torch.tensor(
        [
            0,
            100,
            500,
            999,
        ],
        dtype=torch.long,
    )

    zt = latent_forward_diffusion(
        z0,
        timesteps,
        noise,
        schedule,
    )

    assert zt.shape == z0.shape

    assert torch.isfinite(
        zt
    ).all()

    print("PASS")


# =============================================================================
# TEST 11
# =============================================================================

def test_conditional_gradient_flow() -> None:

    print(
        "\nTEST 11 — CONDITIONAL GRADIENT FLOW"
    )

    model = ConditionalLatentUNet().to(
        TEST_DEVICE
    )

    model.train()

    z = torch.randn(
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    t = torch.randint(
        0,
        T,
        (4,),
        device=TEST_DEVICE,
    )

    labels = torch.randint(
        0,
        NUM_CLASSES,
        (4,),
        device=TEST_DEVICE,
    )

    target = torch.randn_like(
        z
    )

    output = model(
        z,
        t,
        labels,
    )

    loss = F.mse_loss(
        output,
        target,
    )

    model.zero_grad(
        set_to_none=True
    )

    loss.backward()

    gradient = (
        model.condition_embedding
        .weight
        .grad
    )

    assert gradient is not None

    gradient_norm = (
        gradient.norm().item()
    )

    print(
        f"Condition embedding "
        f"gradient norm: "
        f"{gradient_norm:.8f}"
    )

    assert gradient_norm > 0.0

    print("PASS")


# =============================================================================
# TEST 12
# =============================================================================

def test_device_consistency() -> None:

    print(
        "\nTEST 12 — DEVICE CONSISTENCY"
    )

    model = ConditionalLatentUNet().to(
        DEVICE
    )

    actual_model_device = (
        get_model_device(model)
    )

    schedule = (
        make_diffusion_schedule(
            device=TEST_DEVICE
        )
    )

    schedule = (
        synchronize_model_and_schedule(
            model,
            schedule,
        )
    )

    for key, value in schedule.items():

        assert (
            value.device
            ==
            actual_model_device
        ), (
            f"{key} is on "
            f"{value.device}, expected "
            f"{actual_model_device}"
        )

    z = torch.randn(
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=actual_model_device,
    )

    timesteps = torch.randint(
        0,
        T,
        (2,),
        device=actual_model_device,
    )

    labels = torch.randint(
        0,
        NUM_CLASSES,
        (2,),
        device=actual_model_device,
    )

    output = model(
        z,
        timesteps,
        labels,
    )

    assert (
        output.device
        ==
        actual_model_device
    )

    print(
        f"Preferred device: {DEVICE}"
    )

    print(
        f"Actual model device: "
        f"{actual_model_device}"
    )

    print("PASS")


# =============================================================================
# TEST 13
# =============================================================================

def test_one_step_conditional_diffusion() -> None:

    print(
        "\nTEST 13 — ONE-STEP CONDITIONAL DIFFUSION"
    )

    model = ConditionalLatentUNet().to(
        TEST_DEVICE
    )

    schedule = (
        make_diffusion_schedule(
            device=TEST_DEVICE
        )
    )

    z = torch.randn(
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.long,
    )

    result = conditional_ddpm_step(
        model=model,
        z_t=z,
        timestep=10,
        labels=labels,
        schedule=schedule,
    )

    assert result.shape == z.shape

    assert (
        result.device
        ==
        get_model_device(model)
    )

    assert torch.isfinite(
        result
    ).all()

    print("PASS")


# =============================================================================
# TEST 14
# =============================================================================

def test_conditional_sampler() -> None:

    print(
        "\nTEST 14 — COMPLETE CONDITIONAL SAMPLER"
    )

    model = ConditionalLatentUNet().to(
        TEST_DEVICE
    )

    schedule = (
        make_diffusion_schedule(
            device=TEST_DEVICE
        )
    )

    labels = torch.tensor(
        [
            0,
            1,
            2,
            3,
        ],
        dtype=torch.long,
    )

    samples, trajectory = (
        sample_conditional_latent_diffusion(
            model=model,
            schedule=schedule,
            labels=labels,
            latent_shape=(
                4,
                LATENT_CHANNELS,
                LATENT_SIZE,
                LATENT_SIZE,
            ),
            return_trajectory=True,
        )
    )

    assert samples.shape == (
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    required = {
        T - 1,
        750,
        500,
        250,
        100,
        50,
        25,
        10,
        1,
        0,
    }

    assert required.issubset(
        set(
            trajectory.keys()
        )
    ), (
        f"Missing trajectory "
        f"checkpoints. "
        f"Expected {required}, "
        f"got "
        f"{set(trajectory.keys())}"
    )

    assert (
        samples.device
        ==
        get_model_device(model)
    )

    print(
        f"Generated latent shape: "
        f"{tuple(samples.shape)}"
    )

    print(
        f"Trajectory checkpoints: "
        f"{sorted(trajectory.keys(), reverse=True)}"
    )

    print("PASS")


# =============================================================================
# TEST 15
# =============================================================================

def test_conditioned_sampler_device_sync() -> None:

    print(
        "\nTEST 15 — SAMPLER DEVICE SYNCHRONIZATION"
    )

    model = ConditionalLatentUNet().to(
        DEVICE
    )

    actual_model_device = (
        get_model_device(model)
    )

    schedule = (
        make_diffusion_schedule(
            device=TEST_DEVICE
        )
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.long,
        device=TEST_DEVICE,
    )

    initial_noise = torch.randn(
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    samples, trajectory = (
        sample_conditional_latent_diffusion(
            model=model,
            schedule=schedule,
            labels=labels,
            latent_shape=(
                2,
                LATENT_CHANNELS,
                LATENT_SIZE,
                LATENT_SIZE,
            ),
            initial_noise=initial_noise,
            return_trajectory=True,
        )
    )

    assert (
        samples.device
        ==
        actual_model_device
    )

    assert all(
        value.device
        ==
        actual_model_device
        for value in trajectory.values()
    )

    print(
        f"Model device: "
        f"{actual_model_device}"
    )

    print(
        f"Initial noise device: "
        f"{initial_noise.device}"
    )

    print(
        f"Final sample device: "
        f"{samples.device}"
    )

    print("PASS")


# =============================================================================
# CONDITION SENSITIVITY
# =============================================================================

@torch.no_grad()
def condition_sensitivity_experiment(
    model: ConditionalLatentUNet,
    autoencoder: Autoencoder,
    schedule: Dict[str, torch.Tensor],
) -> float:

    print("\n" + "=" * 80)
    print("CONDITION SENSITIVITY")
    print("=" * 80)

    model.eval()
    autoencoder.eval()

    model_device = get_model_device(
        model
    )

    schedule = (
        synchronize_model_and_schedule(
            model,
            schedule,
        )
    )

    same_noise = torch.randn(
        NUM_CLASSES,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=model_device,
    )

    labels = torch.arange(
        NUM_CLASSES,
        device=model_device,
        dtype=torch.long,
    )

    generated_latents, _ = (
        sample_conditional_latent_diffusion(
            model=model,
            schedule=schedule,
            labels=labels,
            latent_shape=(
                NUM_CLASSES,
                LATENT_CHANNELS,
                LATENT_SIZE,
                LATENT_SIZE,
            ),
            initial_noise=same_noise,
            return_trajectory=False,
        )
    )

    generated_images = (
        autoencoder.decode(
            generated_latents
        )
    )

    differences = []

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            i + 1,
            NUM_CLASSES,
        ):

            difference = torch.mean(
                torch.abs(
                    generated_images[i]
                    -
                    generated_images[j]
                )
            )

            differences.append(
                difference
            )

    mean_difference = (
        torch.stack(
            differences
        ).mean()
    )

    print(
        f"Mean pairwise generated-image "
        f"difference: "
        f"{mean_difference.item():.6f}"
    )

    return float(
        mean_difference.item()
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("\n")
    print("=" * 80)
    print("STARTING LESSON 24")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PRE-TRAINING TESTS
    # -------------------------------------------------------------------------

    test_groupnorm()

    test_dataset_condition_alignment()

    test_autoencoder_shapes()

    test_diffusion_schedule()

    test_time_embedding()

    test_condition_embedding()

    test_distinct_condition_embeddings()

    test_conditional_unet_output()

    test_condition_affects_model_output()

    test_latent_forward_diffusion()

    test_conditional_gradient_flow()

    test_device_consistency()

    test_one_step_conditional_diffusion()

    test_conditional_sampler()

    test_conditioned_sampler_device_sync()

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------

    dataset = ConditionalShapesDataset(
        length=100000
    )

    loader = DataLoader(
        dataset,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\n" + "=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        f"Dataset length : "
        f"{len(dataset)}"
    )

    print(
        f"Classes        : "
        f"{CLASS_NAMES}"
    )

    show_dataset_examples(
        dataset,
        count=8,
    )

    # -------------------------------------------------------------------------
    # AUTOENCODER
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING AUTOENCODER")
    print("=" * 80)

    autoencoder = Autoencoder().to(
        DEVICE
    )

    ae_losses = train_autoencoder(
        autoencoder=autoencoder,
        loader=loader,
        device=DEVICE,
        steps=AE_TRAIN_STEPS,
    )

    print(
        f"Final AE loss: "
        f"{ae_losses[-1]:.6f}"
    )

    show_reconstructions(
        autoencoder,
        dataset,
        DEVICE,
        count=8,
    )

    # -------------------------------------------------------------------------
    # CONDITIONAL DIFFUSION MODEL
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "BUILDING CONDITIONAL "
        "LATENT DIFFUSION MODEL"
    )
    print("=" * 80)

    diffusion_model = (
        ConditionalLatentUNet().to(
            DEVICE
        )
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in diffusion_model.parameters()
    )

    print(
        f"Diffusion model parameters: "
        f"{parameter_count:,}"
    )

    schedule = (
        make_diffusion_schedule(
            device=DEVICE
        )
    )

    # -------------------------------------------------------------------------
    # TRAIN CONDITIONAL DIFFUSION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "TRAINING CONDITIONAL "
        "LATENT DIFFUSION"
    )
    print("=" * 80)

    diffusion_losses = (
        train_conditional_diffusion(
            model=diffusion_model,
            autoencoder=autoencoder,
            loader=loader,
            schedule=schedule,
            device=DEVICE,
            steps=DIFFUSION_TRAIN_STEPS,
        )
    )

    print(
        f"Final diffusion loss: "
        f"{diffusion_losses[-1]:.6f}"
    )

    # -------------------------------------------------------------------------
    # FINAL DEVICE VERIFICATION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL DEVICE VERIFICATION")
    print("=" * 80)

    final_model_device = (
        get_model_device(
            diffusion_model
        )
    )

    final_autoencoder_device = (
        get_model_device(
            autoencoder
        )
    )

    schedule = (
        synchronize_model_and_schedule(
            diffusion_model,
            schedule,
        )
    )

    print(
        f"Preferred DEVICE       : "
        f"{DEVICE}"
    )

    print(
        f"Diffusion model device : "
        f"{final_model_device}"
    )

    print(
        f"Autoencoder device     : "
        f"{final_autoencoder_device}"
    )

    # Do not require:
    #
    #     final_model_device == DEVICE
    #
    # The actual model device is authoritative.

    assert (
        final_model_device
        ==
        final_autoencoder_device
    ), (
        "Diffusion model and autoencoder "
        "are on different devices."
    )

    for name, value in schedule.items():

        assert (
            value.device
            ==
            final_model_device
        ), (
            f"Schedule tensor '{name}' "
            f"is on {value.device}, "
            f"while model is on "
            f"{final_model_device}."
        )

    print(
        "Model, autoencoder, and schedule "
        "are device-consistent."
    )

    print("PASS")

    # -------------------------------------------------------------------------
    # CONDITION SENSITIVITY
    # -------------------------------------------------------------------------

    sensitivity = (
        condition_sensitivity_experiment(
            model=diffusion_model,
            autoencoder=autoencoder,
            schedule=schedule,
        )
    )

    assert math.isfinite(
        sensitivity
    )

    assert sensitivity >= 0.0

    # -------------------------------------------------------------------------
    # FINAL CONDITIONAL GENERATION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL CONDITIONAL GENERATION")
    print("=" * 80)

    diffusion_model.eval()
    autoencoder.eval()

    model_device = (
        get_model_device(
            diffusion_model
        )
    )

    labels = torch.arange(
        NUM_CLASSES,
        device=model_device,
        dtype=torch.long,
    )

    generated_latents, trajectory = (
        sample_conditional_latent_diffusion(
            model=diffusion_model,
            schedule=schedule,
            labels=labels,
            latent_shape=(
                NUM_CLASSES,
                LATENT_CHANNELS,
                LATENT_SIZE,
                LATENT_SIZE,
            ),
            return_trajectory=True,
        )
    )

    with torch.no_grad():

        generated_images = (
            autoencoder.decode(
                generated_latents
            )
        )

    print(
        f"Generated latent shape: "
        f"{tuple(generated_latents.shape)}"
    )

    print(
        f"Generated image shape: "
        f"{tuple(generated_images.shape)}"
    )

    show_generated_samples(
        generated_images,
        labels,
        title="Conditioned Generation",
    )

    # -------------------------------------------------------------------------
    # SAME NOISE / DIFFERENT CONDITION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAME NOISE / DIFFERENT CONDITION")
    print("=" * 80)

    comparison_noise = torch.randn(
        NUM_CLASSES,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=model_device,
    )

    comparison_labels = torch.arange(
        NUM_CLASSES,
        device=model_device,
        dtype=torch.long,
    )

    comparison_latents, _ = (
        sample_conditional_latent_diffusion(
            model=diffusion_model,
            schedule=schedule,
            labels=comparison_labels,
            latent_shape=(
                NUM_CLASSES,
                LATENT_CHANNELS,
                LATENT_SIZE,
                LATENT_SIZE,
            ),
            initial_noise=comparison_noise,
            return_trajectory=True,
        )
    )

    with torch.no_grad():

        comparison_images = (
            autoencoder.decode(
                comparison_latents
            )
        )

    pairwise_differences = []

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            i + 1,
            NUM_CLASSES,
        ):

            pairwise_differences.append(
                torch.mean(
                    torch.abs(
                        comparison_images[i]
                        -
                        comparison_images[j]
                    )
                ).item()
            )

    mean_pairwise_difference = float(
        np.mean(
            pairwise_differences
        )
    )

    print(
        f"Mean pairwise difference: "
        f"{mean_pairwise_difference:.6f}"
    )

    print(
        "The same initial latent noise "
        "was used for every condition."
    )

    print(
        "Only the semantic condition changed."
    )

    show_same_noise_conditions(
        comparison_images,
        comparison_labels,
    )

    # -------------------------------------------------------------------------
    # TRAINING CURVE
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    plt.plot(
        diffusion_losses
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "Conditional Latent Diffusion Training"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # LESSON RESULT
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("LESSON 24 RESULT")
    print("=" * 80)

    print(
        "The model now supports explicit "
        "semantic conditioning."
    )

    print(
        "Condition information is represented "
        "by a learned embedding."
    )

    print(
        "Timestep and condition information "
        "are injected into residual blocks."
    )

    print(
        "The U-Net skip connection uses an "
        "explicit channel projection."
    )

    print(
        "The generated result changes when "
        "the semantic condition changes."
    )

    print(
        "Device synchronization is based on "
        "the actual model device."
    )

    print(
        "No pretrained model, checkpoint, "
        "or pretrained encoder was used."
    )

    print("\nReal-world connection:")

    print(
        "A robot waste-sorting system can "
        "associate semantic conditions with "
        "different object categories."
    )

    print(
        "This enables category-controlled "
        "synthetic visual generation for "
        "perception and manipulation research."
    )

    print("\nNext lesson:")

    print(
        "Lesson 25 — Cross-Attention"
    )

    print(
        "Instead of compressing the entire "
        "condition into one vector, image "
        "features will attend to condition "
        "tokens using Q/K/V attention."
    )

    print("=" * 80)
    print("LESSON 24 COMPLETE")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()