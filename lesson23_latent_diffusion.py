"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 23 — LATENT DIFFUSION
================================================================================

Filename:
    lesson23_latent_diffusion.py

================================================================================
PURPOSE
================================================================================

This lesson introduces LATENT DIFFUSION.

Lessons 16–22 performed diffusion directly in pixel space.

    image -> noise -> denoise -> image

Latent diffusion changes the representation:

    image
      |
      v
    encoder
      |
      v
    compact latent
      |
      v
    diffusion in latent space
      |
      v
    clean latent
      |
      v
    decoder
      |
      v
    image

The central engineering idea is:

    LEARN A COMPACT VISUAL REPRESENTATION
        ->
    PERFORM DIFFUSION IN THAT REPRESENTATION
        ->
    DECODE THE RESULT BACK TO IMAGE SPACE

IMPORTANT COURSE RULE
---------------------

No pretrained models are used.

The autoencoder is trained from scratch.

The latent diffusion model is trained from scratch.

This lesson is NOT using Stable Diffusion.


================================================================================
WHAT YOU WILL LEARN
================================================================================

1. Why pixel-space diffusion becomes expensive.
2. What a latent representation is.
3. How an autoencoder compresses images.
4. How to train an autoencoder from scratch.
5. How to measure reconstruction quality.
6. How to inspect learned latent statistics.
7. How diffusion can operate in latent space.
8. How to train a latent-space noise predictor.
9. How to perform DDPM sampling in latent space.
10. How to decode generated latents into images.
11. Why latent diffusion reduces the representation size.
12. Why latent diffusion is useful for robotics and generative vision.
13. How this prepares us for conditioning in Lesson 24.


================================================================================
TECHNICAL / COURSE ROADMAP
================================================================================

Lessons 01–13
-------------
Computer vision foundations:
images, tensors, gradients, CNNs, autoencoders, VAE, GAN, transformers.

Lessons 14–15
-------------
Generative modeling and probability foundations.

Lessons 16–18
-------------
Diffusion:
forward process, schedules, noise prediction.

Lessons 19–20
-------------
DDPM and diffusion U-Net.

Lesson 21
---------
DDPM sampling.

Lesson 22
---------
DDIM and accelerated sampling.

Lesson 23  <-- CURRENT
----------------------
LATENT DIFFUSION.

Lesson 24
---------
CONDITIONING.

Lesson 25
---------
CROSS-ATTENTION.

Lesson 26
---------
TEXT CONDITIONING.

Lesson 27
---------
BUILD OUR OWN TEXT-TO-IMAGE MODEL.


================================================================================
REAL-WORLD PROBLEM
================================================================================

Suppose a robot camera produces a:

    256 x 256 RGB image.

That image contains:

    256 * 256 * 3 = 196,608 scalar values.

A diffusion model may process the representation hundreds or thousands of
times during generation.

This can become expensive.

Instead, an encoder can learn a compact representation.

For example:

    image:
        3 x 256 x 256

    latent:
        4 x 32 x 32

Then diffusion operates on the smaller latent representation.

The exact compression ratio depends on the architecture.

The important principle is:

    REPRESENTATION LEARNING
        +
    GENERATIVE MODELING


================================================================================
THEORY
================================================================================

AUTOENCODER
-----------

An autoencoder contains:

    Encoder E
    Decoder D

Encoding:

    z = E(x)

Decoding:

    x_hat = D(z)

Training minimizes reconstruction error:

    L_reconstruction
        =
    ||x - x_hat||^2


LATENT REPRESENTATION
---------------------

Original image:

    x in R^(C x H x W)

Latent:

    z in R^(C_latent x H_latent x W_latent)

The latent should contain enough information to reconstruct useful visual
structure while being smaller than the original image representation.


LATENT DIFFUSION
----------------

After training the autoencoder:

    z0 = E(x0)

We perform diffusion on z0 instead of x0.

Forward diffusion:

    z_t =
        sqrt(alpha_bar_t) z_0
        +
        sqrt(1-alpha_bar_t) epsilon

where:

    epsilon ~ N(0,I)

The diffusion model predicts:

    epsilon_theta(z_t,t)

The training objective is:

    L =
        E[
            ||epsilon -
              epsilon_theta(z_t,t)||^2
        ]


GENERATION
----------

Start with random latent noise:

    z_T ~ N(0,I)

Then:

    z_T -> z_(T-1) -> ... -> z_0

Finally:

    image = D(z_0)


IMPORTANT
---------

The diffusion model generates the LATENT.

The decoder converts the latent into an IMAGE.


================================================================================
MATHEMATICS
================================================================================

Autoencoder:

    z = E_theta(x)

    x_hat = D_phi(z)

Reconstruction loss:

    L_AE =
        mean((x - x_hat)^2)


Latent forward diffusion:

    z_t =
        sqrt(alpha_bar_t) z_0
        +
        sqrt(1-alpha_bar_t) epsilon


Noise prediction:

    epsilon_hat =
        epsilon_theta(z_t,t)


Diffusion loss:

    L_diffusion =
        mean((epsilon - epsilon_hat)^2)


The architecture therefore separates:

    REPRESENTATION

from:

    GENERATION


================================================================================
IMPLEMENTATION
================================================================================

We implement:

    1. Synthetic structured images.
    2. Convolutional encoder.
    3. Convolutional decoder.
    4. Autoencoder training.
    5. Reconstruction evaluation.
    6. Latent statistics.
    7. Diffusion schedule.
    8. Latent diffusion U-Net.
    9. Latent diffusion training.
    10. Latent DDPM sampling.
    11. Decoder reconstruction.
    12. Visualizations.
    13. Numerical tests.
    14. Device-consistency tests.


================================================================================
DEVICE SAFETY
================================================================================

This lesson explicitly tests and enforces device consistency.

A common failure in diffusion implementations is:

    model       -> CUDA
    schedule    -> CUDA
    timestep    -> CPU

Then:

    CUDA_tensor.gather(CPU_index)

fails.

Therefore the sampler always derives its runtime device from the model.

The schedule is moved to the same device.

Timesteps are created on the same device.

Latents are created on the same device.

This makes the sampling pipeline robust.


================================================================================
REAL-WORLD ROBOTICS APPLICATION
================================================================================

A robot camera can produce large visual observations.

A latent representation can compress those observations before expensive
generative processing.

Possible applications:

    - visual prediction
    - scene completion
    - synthetic training data
    - visual planning
    - manipulation-state prediction
    - simulation augmentation
    - uncertainty-aware perception
    - generative visual world models

For a dual-arm manipulation system, the pipeline could eventually look like:

    camera images
         |
         v
    visual encoder
         |
         v
    compact latent
         |
         +----> perception
         |
         +----> prediction
         |
         +----> generative planning
         |
         v
    decoder / visualization


================================================================================
REAL-WORLD RESULT
================================================================================

After this lesson we should demonstrate:

    1. Image -> latent compression.
    2. Latent -> image reconstruction.
    3. Latent dimensionality is smaller.
    4. Diffusion can learn noise prediction in latent space.
    5. Random latent noise can be denoised.
    6. The decoder converts generated latent representations into images.


================================================================================
LESSON RESULT
================================================================================

The central concept is:

    IMAGE
      ->
    REPRESENTATION
      ->
    DIFFUSION
      ->
    REPRESENTATION
      ->
    IMAGE


================================================================================
CONNECTION TO NEXT LESSON
================================================================================

The current diffusion model is unconditional.

It learns the distribution of the training data but has no explicit control
over WHAT should be generated.

Lesson 24 introduces CONDITIONING.

Conditioning allows generation to depend on information such as:

    class labels
    robot state
    task identity
    depth
    segmentation
    visual features
    text

This is the next step toward controllable generative vision.


================================================================================
"""

import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 32
CHANNELS = 1

# Latent representation.
LATENT_CHANNELS = 4
LATENT_SIZE = 8

# Autoencoder.
AE_BATCH_SIZE = 64
AE_TRAIN_STEPS = 1000
AE_LEARNING_RATE = 2e-3

# Diffusion.
T = 1000
BETA_START = 1e-4
BETA_END = 0.02

DIFFUSION_BATCH_SIZE = 64
DIFFUSION_TRAIN_STEPS = 1400
DIFFUSION_LEARNING_RATE = 2e-3

BASE_CHANNELS = 32
TIME_EMBED_DIM = 128

NUM_SAMPLE_IMAGES = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DEVICE = torch.device("cpu")

PRINT_EVERY_AE = 200
PRINT_EVERY_DIFFUSION = 200

VISUALIZE = True


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int) -> None:
    """Set random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def get_model_device(model: nn.Module) -> torch.device:
    """
    Return the actual device of the model.

    This is safer than assuming that the global DEVICE variable is always
    identical to the model's runtime device.
    """

    return next(model.parameters()).device


def move_schedule_to_device(
    schedule,
    device: torch.device,
):
    """
    Move every diffusion schedule tensor to one device.

    This prevents CPU/CUDA gather errors.
    """

    return {
        key: value.to(device)
        for key, value in schedule.items()
    }


def synchronize_model_and_schedule(
    model: nn.Module,
    schedule,
):
    """
    Make model and diffusion schedule use the same device.

    The model's actual parameter device is treated as authoritative.
    """

    model_device = get_model_device(model)

    schedule = move_schedule_to_device(
        schedule,
        model_device,
    )

    return schedule


def valid_group_count(
    num_channels: int,
    preferred_groups: int = 8,
) -> int:
    """
    Return a GroupNorm group count that divides num_channels.

    Examples:

        1   -> 1
        32  -> 8
        64  -> 8
        128 -> 8
    """

    num_channels = int(num_channels)
    preferred_groups = int(preferred_groups)

    if num_channels <= 0:
        raise ValueError(
            "num_channels must be positive"
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

def generate_structured_images(
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate simple structured grayscale images.

    The dataset contains:

        circles
        rectangles
        diagonal bands
        radial gradients

    Output:

        [B,1,H,W]

    Values are in approximately [0,1].
    """

    yy, xx = torch.meshgrid(
        torch.linspace(
            -1.0,
            1.0,
            image_size,
            device=device,
        ),
        torch.linspace(
            -1.0,
            1.0,
            image_size,
            device=device,
        ),
        indexing="ij",
    )

    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    images = torch.zeros(
        batch_size,
        1,
        image_size,
        image_size,
        device=device,
    )

    for i in range(batch_size):

        image = torch.zeros(
            1,
            image_size,
            image_size,
            device=device,
        )

        pattern_type = i % 4

        if pattern_type == 0:

            cx = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.35,
                0.35,
            )

            cy = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.35,
                0.35,
            )

            radius = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.25,
                0.55,
            )

            circle = (
                (xx[0] - cx) ** 2
                +
                (yy[0] - cy) ** 2
            ) < radius ** 2

            image[0, circle] = 1.0

        elif pattern_type == 1:

            cx = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.25,
                0.25,
            )

            cy = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.25,
                0.25,
            )

            width = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.35,
                0.75,
            )

            height = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.35,
                0.75,
            )

            rectangle = (
                (xx[0] - cx).abs()
                < width / 2
            ) & (
                (yy[0] - cy).abs()
                < height / 2
            )

            image[0, rectangle] = 1.0

        elif pattern_type == 2:

            slope = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.5,
                1.5,
            )

            offset = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.5,
                0.5,
            )

            thickness = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.08,
                0.18,
            )

            distance = (
                yy[0]
                -
                slope * xx[0]
                -
                offset
            ).abs()

            image[
                0,
                distance < thickness
            ] = 1.0

        else:

            radius = torch.sqrt(
                xx[0] ** 2
                +
                yy[0] ** 2
            )

            image[0] = torch.clamp(
                1.0 - radius,
                0.0,
                1.0,
            )

        noise = 0.03 * torch.randn_like(
            image
        )

        image = torch.clamp(
            image + noise,
            0.0,
            1.0,
        )

        images[i] = image

    return images


# =============================================================================
# AUTOENCODER
# =============================================================================

class Encoder(nn.Module):
    """
    Image encoder.

    Input:

        [B,1,32,32]

    Output:

        [B,4,8,8]
    """

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

            nn.GroupNorm(
                valid_group_count(32),
                32,
            ),

            nn.SiLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.GroupNorm(
                valid_group_count(64),
                64,
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
    """
    Latent decoder.

    Input:

        [B,4,8,8]

    Output:

        [B,1,32,32]
    """

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
                stride=1,
                padding=1,
            ),

            nn.GroupNorm(
                valid_group_count(64),
                64,
            ),

            nn.SiLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.GroupNorm(
                valid_group_count(32),
                32,
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
    """Complete image encoder/decoder."""

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
    ):

        z = self.encode(x)
        x_hat = self.decode(z)

        return x_hat, z


# =============================================================================
# AUTOENCODER TRAINING
# =============================================================================

def train_autoencoder(
    model: Autoencoder,
    device: torch.device,
) -> Autoencoder:

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=AE_LEARNING_RATE,
    )

    print()
    print("=" * 80)
    print("AUTOENCODER TRAINING")
    print("=" * 80)

    print(
        "Steps      :",
        AE_TRAIN_STEPS,
    )

    print(
        "Batch size :",
        AE_BATCH_SIZE,
    )

    print(
        "Parameters :",
        f"{count_parameters(model):,}",
    )

    start_time = time.perf_counter()

    first_loss = None
    last_loss = None
    best_loss = float("inf")

    for step in range(
        1,
        AE_TRAIN_STEPS + 1,
    ):

        images = generate_structured_images(
            AE_BATCH_SIZE,
            IMAGE_SIZE,
            device,
        )

        reconstruction, _ = model(
            images
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

        value = float(
            loss.detach().item()
        )

        if first_loss is None:
            first_loss = value

        last_loss = value
        best_loss = min(
            best_loss,
            value,
        )

        if (
            step == 1
            or step % PRINT_EVERY_AE == 0
        ):

            print(
                f"step {step:4d}/{AE_TRAIN_STEPS} | "
                f"reconstruction loss = {value:.6f}"
            )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    print()
    print(
        f"Initial loss : {first_loss:.6f}"
    )

    print(
        f"Final loss   : {last_loss:.6f}"
    )

    print(
        f"Best loss    : {best_loss:.6f}"
    )

    print(
        f"Training time: {elapsed:.2f} sec"
    )

    model.eval()

    return model


# =============================================================================
# DIFFUSION SCHEDULE
# =============================================================================

def make_schedule(
    T: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
):
    """
    Construct the DDPM diffusion schedule.
    """

    betas = torch.linspace(
        beta_start,
        beta_end,
        T,
        device=device,
        dtype=torch.float32,
    )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0,
    )

    sqrt_alpha_bars = torch.sqrt(
        alpha_bars
    )

    sqrt_one_minus_alpha_bars = torch.sqrt(
        1.0 - alpha_bars
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": (
            sqrt_one_minus_alpha_bars
        ),
    }


def extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    shape,
) -> torch.Tensor:
    """
    Extract schedule values for a batch of timesteps.

    IMPORTANT DEVICE RULE
    ---------------------

    values and timesteps MUST be on the same device.

    This function explicitly checks the condition so a future device bug
    produces a useful error instead of a confusing CUDA gather error.
    """

    if values.device != timesteps.device:

        raise RuntimeError(
            "Diffusion schedule and timestep tensor "
            "must be on the same device. "
            f"values={values.device}, "
            f"timesteps={timesteps.device}"
        )

    batch_size = timesteps.shape[0]

    out = values.gather(
        0,
        timesteps,
    )

    return out.reshape(
        batch_size,
        *([1] * (len(shape) - 1)),
    )


def latent_forward_diffusion(
    z0: torch.Tensor,
    timesteps: torch.Tensor,
    schedule,
):
    """
    Add noise to latent representation.

        z_t =
            sqrt(alpha_bar_t) z_0
            +
            sqrt(1-alpha_bar_t) epsilon
    """

    schedule = move_schedule_to_device(
        schedule,
        z0.device,
    )

    timesteps = timesteps.to(
        z0.device
    )

    noise = torch.randn_like(
        z0
    )

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

    zt = (
        sqrt_alpha_bar * z0
        +
        sqrt_one_minus_alpha_bar * noise
    )

    return zt, noise


# =============================================================================
# TIME EMBEDDING
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Convert diffusion timestep into a vector.
    """

    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

    def forward(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        half_dim = self.embedding_dim // 2

        scale = (
            math.log(10000.0)
            /
            max(
                half_dim - 1,
                1,
            )
        )

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            *
            -scale
        )

        arguments = (
            timesteps.float().unsqueeze(1)
            *
            frequencies.unsqueeze(0)
        )

        embedding = torch.cat(
            [
                torch.sin(arguments),
                torch.cos(arguments),
            ],
            dim=1,
        )

        if (
            embedding.shape[1]
            <
            self.embedding_dim
        ):

            embedding = F.pad(
                embedding,
                (
                    0,
                    self.embedding_dim
                    -
                    embedding.shape[1],
                ),
            )

        return embedding


# =============================================================================
# LATENT RESIDUAL BLOCK
# =============================================================================

class LatentResidualBlock(nn.Module):
    """
    Residual block conditioned on timestep embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            valid_group_count(
                in_channels
            ),
            in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Linear(
            time_embedding_dim,
            out_channels,
        )

        self.norm2 = nn.GroupNorm(
            valid_group_count(
                out_channels
            ),
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
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)

        h = self.conv1(h)

        time_bias = self.time_projection(
            time_embedding
        ).unsqueeze(-1).unsqueeze(-1)

        h = h + time_bias

        h = self.norm2(h)
        h = F.silu(h)

        h = self.conv2(h)

        return h + residual


# =============================================================================
# LATENT DIFFUSION U-NET
# =============================================================================

class LatentDiffusionUNet(nn.Module):
    """
    Small U-Net operating in latent space.

    Input:

        [B,4,8,8]

    Output:

        [B,4,8,8]
    """

    def __init__(
        self,
        latent_channels: int = LATENT_CHANNELS,
        base_channels: int = BASE_CHANNELS,
        time_embedding_dim: int = TIME_EMBED_DIM,
    ):
        super().__init__()

        self.time_embedding = (
            SinusoidalTimeEmbedding(
                time_embedding_dim
            )
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(
                time_embedding_dim,
                time_embedding_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                time_embedding_dim,
                time_embedding_dim,
            ),
        )

        self.input_conv = nn.Conv2d(
            latent_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.block1 = LatentResidualBlock(
            base_channels,
            base_channels,
            time_embedding_dim,
        )

        self.down = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.block2 = LatentResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_embedding_dim,
        )

        self.bottleneck = LatentResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_embedding_dim,
        )

        self.up = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.block3 = LatentResidualBlock(
            base_channels * 2,
            base_channels,
            time_embedding_dim,
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
    ) -> torch.Tensor:

        timesteps = timesteps.to(
            z.device
        )

        t = self.time_embedding(
            timesteps
        )

        t = self.time_mlp(t)

        x0 = self.input_conv(z)

        h1 = self.block1(
            x0,
            t,
        )

        h2 = self.down(h1)

        h2 = self.block2(
            h2,
            t,
        )

        h3 = self.bottleneck(
            h2,
            t,
        )

        h4 = self.up(h3)

        h4 = torch.cat(
            [
                h4,
                h1,
            ],
            dim=1,
        )

        h4 = self.block3(
            h4,
            t,
        )

        h4 = self.output_norm(h4)

        h4 = F.silu(h4)

        return self.output_conv(h4)


# =============================================================================
# LATENT DDPM REVERSE STEP
# =============================================================================

@torch.no_grad()
def latent_ddpm_step(
    model: nn.Module,
    z_t: torch.Tensor,
    timesteps: torch.Tensor,
    schedule,
):
    """
    Perform one reverse DDPM step in latent space.

    DEVICE SAFETY
    -------------

    The model device is authoritative.

    The schedule is moved to that device.

    z_t is moved to that device.

    timesteps are moved to that device.

    Therefore:

        schedule
        timestep
        latent
        model

    all operate on one device.
    """

    model_device = get_model_device(
        model
    )

    if z_t.device != model_device:

        z_t = z_t.to(
            model_device
        )

    timesteps = timesteps.to(
        model_device
    )

    schedule = synchronize_model_and_schedule(
        model,
        schedule,
    )

    betas = extract(
        schedule["betas"],
        timesteps,
        z_t.shape,
    )

    alphas = extract(
        schedule["alphas"],
        timesteps,
        z_t.shape,
    )

    alpha_bars = extract(
        schedule["alpha_bars"],
        timesteps,
        z_t.shape,
    )

    predicted_noise = model(
        z_t,
        timesteps,
    )

    mean = (
        1.0
        /
        torch.sqrt(alphas)
    ) * (
        z_t
        -
        (
            betas
            /
            torch.sqrt(
                1.0 - alpha_bars
            )
        )
        *
        predicted_noise
    )

    noise = torch.randn_like(
        z_t
    )

    nonzero_mask = (
        timesteps > 0
    ).float().reshape(
        z_t.shape[0],
        1,
        1,
        1,
    )

    variance = betas

    return (
        mean
        +
        nonzero_mask
        *
        torch.sqrt(variance)
        *
        noise
    )


# =============================================================================
# LATENT DDPM SAMPLER
# =============================================================================

@torch.no_grad()
def sample_latent_diffusion(
    model: nn.Module,
    schedule,
    shape,
    device=None,
):
    """
    Complete latent DDPM sampling.

    IMPORTANT DEVICE DESIGN
    -----------------------

    The model device is authoritative.

    If the caller supplies a different device, the sampler still uses the
    actual model device.

    This prevents mismatches such as:

        model     -> cuda:0
        schedule  -> cuda:0
        z         -> cpu
        timestep  -> cpu

    which would cause CUDA gather failures.
    """

    model.eval()

    model_device = get_model_device(
        model
    )

    # The caller's device is intentionally not trusted as the final device.
    # The model device is authoritative.
    if device is not None:
        device = torch.device(device)

    runtime_device = model_device

    schedule = synchronize_model_and_schedule(
        model,
        schedule,
    )

    z = torch.randn(
        shape,
        device=runtime_device,
    )

    for step in reversed(
        range(T)
    ):

        timesteps = torch.full(
            (
                shape[0],
            ),
            step,
            device=runtime_device,
            dtype=torch.long,
        )

        z = latent_ddpm_step(
            model,
            z,
            timesteps,
            schedule,
        )

    return z


# =============================================================================
# LATENT DIFFUSION TRAINING
# =============================================================================

def train_latent_diffusion(
    autoencoder: Autoencoder,
    diffusion_model: LatentDiffusionUNet,
    device: torch.device,
):

    autoencoder = autoencoder.to(
        device
    )

    diffusion_model = diffusion_model.to(
        device
    )

    autoencoder.eval()

    for parameter in autoencoder.parameters():
        parameter.requires_grad = False

    diffusion_model.train()

    schedule = make_schedule(
        T,
        BETA_START,
        BETA_END,
        device,
    )

    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=DIFFUSION_LEARNING_RATE,
    )

    print()
    print("=" * 80)
    print("LATENT DIFFUSION TRAINING")
    print("=" * 80)

    print(
        "Steps      :",
        DIFFUSION_TRAIN_STEPS,
    )

    print(
        "Batch size :",
        DIFFUSION_BATCH_SIZE,
    )

    print(
        "Latent     :",
        f"{LATENT_CHANNELS} x "
        f"{LATENT_SIZE} x "
        f"{LATENT_SIZE}",
    )

    print(
        "Parameters :",
        f"{count_parameters(diffusion_model):,}",
    )

    start_time = time.perf_counter()

    first_loss = None
    last_loss = None
    best_loss = float("inf")

    for step in range(
        1,
        DIFFUSION_TRAIN_STEPS + 1,
    ):

        images = generate_structured_images(
            DIFFUSION_BATCH_SIZE,
            IMAGE_SIZE,
            device,
        )

        with torch.no_grad():

            z0 = autoencoder.encode(
                images
            )

        timesteps = torch.randint(
            0,
            T,
            (
                images.shape[0],
            ),
            device=device,
        )

        zt, true_noise = latent_forward_diffusion(
            z0,
            timesteps,
            schedule,
        )

        predicted_noise = diffusion_model(
            zt,
            timesteps,
        )

        loss = F.mse_loss(
            predicted_noise,
            true_noise,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        value = float(
            loss.detach().item()
        )

        if first_loss is None:
            first_loss = value

        last_loss = value

        best_loss = min(
            best_loss,
            value,
        )

        if (
            step == 1
            or step % PRINT_EVERY_DIFFUSION == 0
        ):

            print(
                f"step {step:4d}/{DIFFUSION_TRAIN_STEPS} | "
                f"latent noise loss = {value:.6f}"
            )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    print()

    print(
        f"Initial loss : {first_loss:.6f}"
    )

    print(
        f"Final loss   : {last_loss:.6f}"
    )

    print(
        f"Best loss    : {best_loss:.6f}"
    )

    print(
        f"Training time: {elapsed:.2f} sec"
    )

    diffusion_model.eval()

    schedule = synchronize_model_and_schedule(
        diffusion_model,
        schedule,
    )

    return (
        diffusion_model,
        schedule,
    )


# =============================================================================
# VISUALIZATION — AUTOENCODER
# =============================================================================

@torch.no_grad()
def visualize_autoencoder(
    model: Autoencoder,
    device: torch.device,
):

    model.eval()

    images = generate_structured_images(
        NUM_SAMPLE_IMAGES,
        IMAGE_SIZE,
        device,
    )

    reconstruction, latent = model(
        images
    )

    images_cpu = images.cpu()
    reconstruction_cpu = reconstruction.cpu()
    latent_cpu = latent.cpu()

    fig, axes = plt.subplots(
        2,
        NUM_SAMPLE_IMAGES,
        figsize=(16, 4),
    )

    for i in range(
        NUM_SAMPLE_IMAGES
    ):

        axes[0, i].imshow(
            images_cpu[i, 0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[0, i].axis("off")

        axes[1, i].imshow(
            reconstruction_cpu[i, 0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[1, i].axis("off")

    axes[0, 0].set_ylabel(
        "Original",
        fontsize=12,
    )

    axes[1, 0].set_ylabel(
        "Reconstruction",
        fontsize=12,
    )

    fig.suptitle(
        "Lesson 23 — Autoencoder Reconstruction",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()

    return (
        images_cpu,
        reconstruction_cpu,
        latent_cpu,
    )


# =============================================================================
# VISUALIZATION — LATENT CHANNELS
# =============================================================================

@torch.no_grad()
def visualize_latent_channels(
    latent: torch.Tensor,
):

    sample = latent[0]

    channels = sample.shape[0]

    fig, axes = plt.subplots(
        1,
        channels,
        figsize=(12, 3),
    )

    if channels == 1:
        axes = [axes]

    for channel in range(
        channels
    ):

        axes[channel].imshow(
            sample[channel],
            cmap="viridis",
        )

        axes[channel].set_title(
            f"Latent {channel}"
        )

        axes[channel].axis("off")

    fig.suptitle(
        "Learned Latent Representation",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZATION — GENERATED SAMPLES
# =============================================================================

@torch.no_grad()
def generate_and_visualize_samples(
    autoencoder: Autoencoder,
    diffusion_model: LatentDiffusionUNet,
    schedule,
):

    autoencoder.eval()
    diffusion_model.eval()

    runtime_device = get_model_device(
        diffusion_model
    )

    # Ensure autoencoder and diffusion model are on the same device.
    autoencoder = autoencoder.to(
        runtime_device
    )

    schedule = synchronize_model_and_schedule(
        diffusion_model,
        schedule,
    )

    start_time = time.perf_counter()

    latent_samples = sample_latent_diffusion(
        diffusion_model,
        schedule,
        (
            NUM_SAMPLE_IMAGES,
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
        ),
        runtime_device,
    )

    images = autoencoder.decode(
        latent_samples
    )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    images_cpu = images.cpu()

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(10, 5),
    )

    for i, ax in enumerate(
        axes.flat
    ):

        ax.imshow(
            images_cpu[i, 0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        ax.axis("off")

    fig.suptitle(
        "Lesson 23 — Latent Diffusion Samples",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()

    print()
    print(
        "Latent sampling time:",
        f"{elapsed:.2f} sec",
    )

    return (
        latent_samples,
        images,
        elapsed,
    )


# =============================================================================
# TEST 1 — GROUPNORM
# =============================================================================

def test_groupnorm_fix():

    print()
    print("TEST 1 — GROUPNORM")

    for channels in [
        1,
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

        assert groups >= 1

        assert (
            channels % groups
            == 0
        )

        print(
            f"channels={channels:3d} "
            f"-> groups={groups}"
        )

    print("PASS")


# =============================================================================
# TEST 2 — AUTOENCODER SHAPES
# =============================================================================

def test_autoencoder_shapes():

    print()
    print("TEST 2 — AUTOENCODER SHAPES")

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

    reconstruction, latent = model(x)

    print(
        "Input shape         :",
        tuple(x.shape),
    )

    print(
        "Latent shape        :",
        tuple(latent.shape),
    )

    print(
        "Reconstruction shape:",
        tuple(reconstruction.shape),
    )

    assert latent.shape == (
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    assert reconstruction.shape == (
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    print("PASS")


# =============================================================================
# TEST 3 — LATENT COMPRESSION
# =============================================================================

def test_latent_compression():

    print()
    print("TEST 3 — LATENT COMPRESSION")

    image_values = (
        CHANNELS
        *
        IMAGE_SIZE
        *
        IMAGE_SIZE
    )

    latent_values = (
        LATENT_CHANNELS
        *
        LATENT_SIZE
        *
        LATENT_SIZE
    )

    ratio = (
        image_values
        /
        latent_values
    )

    print(
        "Image values  :",
        image_values,
    )

    print(
        "Latent values :",
        latent_values,
    )

    print(
        "Compression   :",
        f"{ratio:.2f}x",
    )

    assert latent_values < image_values
    assert ratio > 1.0

    print("PASS")


# =============================================================================
# TEST 4 — RECONSTRUCTION RANGE
# =============================================================================

def test_reconstruction_range():

    print()
    print("TEST 4 — RECONSTRUCTION RANGE")

    model = Autoencoder().to(
        TEST_DEVICE
    )

    x = torch.rand(
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=TEST_DEVICE,
    )

    reconstruction, _ = model(x)

    print(
        "Min:",
        float(
            reconstruction.min()
        ),
    )

    print(
        "Max:",
        float(
            reconstruction.max()
        ),
    )

    assert torch.isfinite(
        reconstruction
    ).all()

    assert (
        reconstruction.min()
        >=
        0.0
    )

    assert (
        reconstruction.max()
        <=
        1.0
    )

    print("PASS")


# =============================================================================
# TEST 5 — DIFFUSION SCHEDULE
# =============================================================================

def test_diffusion_schedule():

    print()
    print("TEST 5 — DIFFUSION SCHEDULE")

    schedule = make_schedule(
        T,
        BETA_START,
        BETA_END,
        TEST_DEVICE,
    )

    assert schedule[
        "betas"
    ].shape == (T,)

    assert schedule[
        "alphas"
    ].shape == (T,)

    assert schedule[
        "alpha_bars"
    ].shape == (T,)

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

    print(
        "Final alpha_bar:",
        float(
            schedule[
                "alpha_bars"
            ][-1]
        ),
    )

    print("PASS")


# =============================================================================
# TEST 6 — LATENT FORWARD DIFFUSION
# =============================================================================

def test_latent_forward_diffusion():

    print()
    print(
        "TEST 6 — LATENT FORWARD DIFFUSION"
    )

    autoencoder = Autoencoder().to(
        TEST_DEVICE
    )

    images = torch.rand(
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=TEST_DEVICE,
    )

    with torch.no_grad():

        z0 = autoencoder.encode(
            images
        )

    schedule = make_schedule(
        T,
        BETA_START,
        BETA_END,
        TEST_DEVICE,
    )

    timesteps = torch.randint(
        0,
        T,
        (
            4,
        ),
        device=TEST_DEVICE,
    )

    zt, noise = latent_forward_diffusion(
        z0,
        timesteps,
        schedule,
    )

    print(
        "z0 shape:",
        tuple(z0.shape),
    )

    print(
        "zt shape:",
        tuple(zt.shape),
    )

    print(
        "noise shape:",
        tuple(noise.shape),
    )

    assert zt.shape == z0.shape
    assert noise.shape == z0.shape

    assert torch.isfinite(
        zt
    ).all()

    assert torch.isfinite(
        noise
    ).all()

    print("PASS")


# =============================================================================
# TEST 7 — DIFFUSION MODEL OUTPUT
# =============================================================================

def test_diffusion_model_output():

    print()
    print(
        "TEST 7 — DIFFUSION MODEL OUTPUT"
    )

    model = LatentDiffusionUNet().to(
        TEST_DEVICE
    )

    z = torch.randn(
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    timesteps = torch.randint(
        0,
        T,
        (
            4,
        ),
        device=TEST_DEVICE,
    )

    predicted_noise = model(
        z,
        timesteps,
    )

    print(
        "Input shape :",
        tuple(z.shape),
    )

    print(
        "Output shape:",
        tuple(
            predicted_noise.shape
        ),
    )

    assert (
        predicted_noise.shape
        ==
        z.shape
    )

    assert torch.isfinite(
        predicted_noise
    ).all()

    print("PASS")


# =============================================================================
# TEST 8 — DEVICE CONSISTENCY
# =============================================================================

def test_device_consistency():

    print()
    print(
        "TEST 8 — DEVICE CONSISTENCY"
    )

    autoencoder = Autoencoder().to(
        TEST_DEVICE
    )

    diffusion_model = (
        LatentDiffusionUNet().to(
            TEST_DEVICE
        )
    )

    schedule = make_schedule(
        T,
        BETA_START,
        BETA_END,
        TEST_DEVICE,
    )

    schedule = synchronize_model_and_schedule(
        diffusion_model,
        schedule,
    )

    assert (
        get_model_device(
            autoencoder
        )
        ==
        TEST_DEVICE
    )

    assert (
        get_model_device(
            diffusion_model
        )
        ==
        TEST_DEVICE
    )

    for value in schedule.values():

        assert (
            value.device
            ==
            TEST_DEVICE
        )

    x = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=TEST_DEVICE,
    )

    with torch.no_grad():

        z = autoencoder.encode(
            x
        )

    timesteps = torch.randint(
        0,
        T,
        (
            2,
        ),
        device=TEST_DEVICE,
    )

    output = diffusion_model(
        z,
        timesteps,
    )

    assert (
        output.device
        ==
        TEST_DEVICE
    )

    assert (
        z.device
        ==
        TEST_DEVICE
    )

    print("PASS")


# =============================================================================
# TEST 9 — GRADIENT FLOW
# =============================================================================

def test_diffusion_gradient_flow():

    print()
    print(
        "TEST 9 — DIFFUSION GRADIENT FLOW"
    )

    model = LatentDiffusionUNet().to(
        TEST_DEVICE
    )

    z = torch.randn(
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    timesteps = torch.randint(
        0,
        T,
        (
            4,
        ),
        device=TEST_DEVICE,
    )

    target = torch.randn_like(z)

    output = model(
        z,
        timesteps,
    )

    loss = F.mse_loss(
        output,
        target,
    )

    loss.backward()

    gradient_norm_squared = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_norm_squared += (
                parameter.grad.detach()
                .float()
                .norm()
                .item()
                ** 2
            )

    gradient_norm = (
        gradient_norm_squared
        **
        0.5
    )

    print(
        "Gradient norm:",
        gradient_norm,
    )

    assert math.isfinite(
        gradient_norm
    )

    assert (
        gradient_norm
        >
        0.0
    )

    print("PASS")


# =============================================================================
# TEST 10 — LATENT STATISTICS
# =============================================================================

def test_latent_statistics(
    autoencoder: Autoencoder,
):

    print()
    print(
        "TEST 10 — LATENT STATISTICS"
    )

    autoencoder.eval()

    device = get_model_device(
        autoencoder
    )

    images = generate_structured_images(
        32,
        IMAGE_SIZE,
        device,
    )

    with torch.no_grad():

        latent = autoencoder.encode(
            images
        )

    mean = latent.mean().item()
    std = latent.std().item()
    minimum = latent.min().item()
    maximum = latent.max().item()

    print(
        f"mean : {mean:.6f}"
    )

    print(
        f"std  : {std:.6f}"
    )

    print(
        f"min  : {minimum:.6f}"
    )

    print(
        f"max  : {maximum:.6f}"
    )

    assert math.isfinite(mean)
    assert math.isfinite(std)
    assert math.isfinite(minimum)
    assert math.isfinite(maximum)

    assert std > 1e-6

    print("PASS")


# =============================================================================
# TEST 11 — COMPLETE AUTOENCODER PIPELINE
# =============================================================================

def test_complete_autoencoder_pipeline(
    autoencoder: Autoencoder,
):

    print()
    print(
        "TEST 11 — COMPLETE AUTOENCODER PIPELINE"
    )

    autoencoder.eval()

    device = get_model_device(
        autoencoder
    )

    images = generate_structured_images(
        4,
        IMAGE_SIZE,
        device,
    )

    with torch.no_grad():

        latent = autoencoder.encode(
            images
        )

        reconstruction = autoencoder.decode(
            latent
        )

    error = F.mse_loss(
        reconstruction,
        images,
    ).item()

    print(
        "Reconstruction MSE:",
        f"{error:.6f}",
    )

    assert (
        reconstruction.shape
        ==
        images.shape
    )

    assert math.isfinite(error)

    print("PASS")


# =============================================================================
# TEST 12 — ONE-STEP LATENT DIFFUSION TRAINING
# =============================================================================

def test_one_step_diffusion_training(
    autoencoder: Autoencoder,
):

    print()
    print(
        "TEST 12 — ONE-STEP LATENT DIFFUSION TRAINING"
    )

    autoencoder.eval()

    diffusion_model = (
        LatentDiffusionUNet().to(
            TEST_DEVICE
        )
    )

    schedule = make_schedule(
        T,
        BETA_START,
        BETA_END,
        TEST_DEVICE,
    )

    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=1e-3,
    )

    images = generate_structured_images(
        4,
        IMAGE_SIZE,
        TEST_DEVICE,
    )

    with torch.no_grad():

        z0 = autoencoder.encode(
            images
        )

    timesteps = torch.randint(
        0,
        T,
        (
            4,
        ),
        device=TEST_DEVICE,
    )

    zt, true_noise = latent_forward_diffusion(
        z0,
        timesteps,
        schedule,
    )

    predicted_noise = diffusion_model(
        zt,
        timesteps,
    )

    loss = F.mse_loss(
        predicted_noise,
        true_noise,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()

    print(
        "One-step loss:",
        f"{loss.item():.6f}",
    )

    assert math.isfinite(
        loss.item()
    )

    print("PASS")


# =============================================================================
# TEST 13 — COMPLETE LATENT SAMPLER
# =============================================================================

def test_latent_sampler(
    autoencoder: Autoencoder,
    diffusion_model: LatentDiffusionUNet,
    schedule,
):
    """
    Test the complete latent sampler.

    IMPORTANT FIX
    -------------

    The previous implementation used TEST_DEVICE here while the supplied
    schedule could be on CUDA.

    That created:

        schedule -> CUDA
        timesteps -> CPU

    and caused:

        RuntimeError:
        Expected all tensors to be on the same device

    The corrected implementation derives the device from the diffusion model.
    """

    print()
    print(
        "TEST 13 — COMPLETE LATENT SAMPLER"
    )

    diffusion_model.eval()
    autoencoder.eval()

    model_device = get_model_device(
        diffusion_model
    )

    # Keep the decoder on the same device.
    autoencoder = autoencoder.to(
        model_device
    )

    # Force schedule onto the model device.
    schedule = synchronize_model_and_schedule(
        diffusion_model,
        schedule,
    )

    for key, value in schedule.items():

        assert (
            value.device
            ==
            model_device
        ), (
            f"Schedule tensor {key} "
            f"is on {value.device}, "
            f"but model is on {model_device}"
        )

    start_time = time.perf_counter()

    latent = sample_latent_diffusion(
        diffusion_model,
        schedule,
        (
            2,
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
        ),
        model_device,
    )

    images = autoencoder.decode(
        latent
    )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    print(
        "Model device:",
        model_device,
    )

    print(
        "Latent device:",
        latent.device,
    )

    print(
        "Image device:",
        images.device,
    )

    print(
        "Latent shape:",
        tuple(latent.shape),
    )

    print(
        "Image shape:",
        tuple(images.shape),
    )

    print(
        "Sampling time:",
        f"{elapsed:.2f} sec",
    )

    assert latent.shape == (
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    assert images.shape == (
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert (
        latent.device
        ==
        model_device
    )

    assert (
        images.device
        ==
        model_device
    )

    assert torch.isfinite(
        latent
    ).all()

    assert torch.isfinite(
        images
    ).all()

    print("PASS")


# =============================================================================
# TEST 14 — CPU/GPU STYLE DEVICE TRANSITION
# =============================================================================

def test_sampler_device_transition(
    diffusion_model: LatentDiffusionUNet,
    schedule,
):
    """
    Explicitly test that the sampler synchronizes the schedule to the model.

    This test is specifically designed to catch the class of bug reported by
    the user.
    """

    print()
    print(
        "TEST 14 — SAMPLER DEVICE SYNCHRONIZATION"
    )

    model_device = get_model_device(
        diffusion_model
    )

    synchronized_schedule = (
        synchronize_model_and_schedule(
            diffusion_model,
            schedule,
        )
    )

    for key, value in synchronized_schedule.items():

        assert (
            value.device
            ==
            model_device
        )

    z = torch.randn(
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=model_device,
    )

    timesteps = torch.randint(
        0,
        T,
        (
            2,
        ),
        device=model_device,
    )

    # This directly exercises the gather path.
    zt, noise = latent_forward_diffusion(
        z,
        timesteps,
        synchronized_schedule,
    )

    assert (
        zt.device
        ==
        model_device
    )

    assert (
        noise.device
        ==
        model_device
    )

    print(
        "Model device:",
        model_device,
    )

    print(
        "Schedule synchronized: PASS"
    )

    print(
        "Gather device check   : PASS"
    )

    print("PASS")


# =============================================================================
# PRE-TRAINING TEST SUITE
# =============================================================================

def run_tests(
    autoencoder: Autoencoder,
):

    test_groupnorm_fix()

    test_autoencoder_shapes()

    test_latent_compression()

    test_reconstruction_range()

    test_diffusion_schedule()

    test_latent_forward_diffusion()

    test_diffusion_model_output()

    test_device_consistency()

    test_diffusion_gradient_flow()

    test_latent_statistics(
        autoencoder
    )

    test_complete_autoencoder_pipeline(
        autoencoder
    )

    test_one_step_diffusion_training(
        autoencoder
    )

    print()
    print("=" * 80)
    print(
        "ALL PRE-TRAINING TESTS PASSED"
    )
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "LESSON 23 — LATENT DIFFUSION"
    )
    print("=" * 80)

    print()
    print(
        "PyTorch version:",
        torch.__version__,
    )

    print(
        "Device         :",
        DEVICE,
    )

    print(
        "Test device    :",
        TEST_DEVICE,
    )

    print(
        "Image          :",
        f"{CHANNELS} x "
        f"{IMAGE_SIZE} x "
        f"{IMAGE_SIZE}",
    )

    print(
        "Latent         :",
        f"{LATENT_CHANNELS} x "
        f"{LATENT_SIZE} x "
        f"{LATENT_SIZE}",
    )

    print(
        "Diffusion T    :",
        T,
    )

    # =========================================================================
    # BUILD AUTOENCODER
    # =========================================================================

    autoencoder = Autoencoder().to(
        TEST_DEVICE
    )

    # =========================================================================
    # PRE-TRAINING TESTS
    # =========================================================================

    run_tests(
        autoencoder
    )

    # =========================================================================
    # TRAIN AUTOENCODER
    # =========================================================================

    autoencoder = train_autoencoder(
        autoencoder,
        DEVICE,
    )

    # =========================================================================
    # AUTOENCODER EVALUATION
    # =========================================================================

    autoencoder.eval()

    evaluation_images = (
        generate_structured_images(
            64,
            IMAGE_SIZE,
            DEVICE,
        )
    )

    with torch.no_grad():

        reconstruction, latent = (
            autoencoder(
                evaluation_images
            )
        )

    reconstruction_mse = F.mse_loss(
        reconstruction,
        evaluation_images,
    ).item()

    print()
    print("=" * 80)
    print(
        "AUTOENCODER EVALUATION"
    )
    print("=" * 80)

    print(
        "Reconstruction MSE:",
        f"{reconstruction_mse:.6f}",
    )

    print(
        "Latent shape:",
        tuple(latent.shape),
    )

    image_values = (
        CHANNELS
        *
        IMAGE_SIZE
        *
        IMAGE_SIZE
    )

    latent_values = (
        LATENT_CHANNELS
        *
        LATENT_SIZE
        *
        LATENT_SIZE
    )

    compression_ratio = (
        image_values
        /
        latent_values
    )

    print(
        "Image scalar values :",
        image_values,
    )

    print(
        "Latent scalar values:",
        latent_values,
    )

    print(
        "Compression ratio   :",
        f"{compression_ratio:.2f}x",
    )

    # =========================================================================
    # VISUALIZE AUTOENCODER
    # =========================================================================

    if VISUALIZE:

        (
            _,
            _,
            latent_visualization,
        ) = visualize_autoencoder(
            autoencoder,
            DEVICE,
        )

        visualize_latent_channels(
            latent_visualization
        )

    # =========================================================================
    # BUILD LATENT DIFFUSION MODEL
    # =========================================================================

    diffusion_model = (
        LatentDiffusionUNet().to(
            DEVICE
        )
    )

    print()
    print("=" * 80)
    print(
        "MODEL SIZES"
    )
    print("=" * 80)

    print(
        "Autoencoder parameters:",
        f"{count_parameters(autoencoder):,}",
    )

    print(
        "Latent diffusion parameters:",
        f"{count_parameters(diffusion_model):,}",
    )

    # =========================================================================
    # TRAIN LATENT DIFFUSION
    # =========================================================================

    (
        diffusion_model,
        schedule,
    ) = train_latent_diffusion(
        autoencoder,
        diffusion_model,
        DEVICE,
    )

    # =========================================================================
    # LATENT NOISE PREDICTION EVALUATION
    # =========================================================================

    test_images = generate_structured_images(
        8,
        IMAGE_SIZE,
        DEVICE,
    )

    with torch.no_grad():

        test_latent = (
            autoencoder.encode(
                test_images
            )
        )

    test_timesteps = torch.randint(
        0,
        T,
        (
            8,
        ),
        device=DEVICE,
    )

    noisy_latent, true_noise = (
        latent_forward_diffusion(
            test_latent,
            test_timesteps,
            schedule,
        )
    )

    with torch.no_grad():

        predicted_noise = (
            diffusion_model(
                noisy_latent,
                test_timesteps,
            )
        )

    noise_mse = F.mse_loss(
        predicted_noise,
        true_noise,
    ).item()

    predicted_std = (
        predicted_noise.std().item()
    )

    true_std = (
        true_noise.std().item()
    )

    print()
    print("=" * 80)
    print(
        "LATENT NOISE PREDICTION EVALUATION"
    )
    print("=" * 80)

    print(
        "Noise prediction MSE:",
        f"{noise_mse:.6f}",
    )

    print(
        "Predicted noise std :",
        f"{predicted_std:.6f}",
    )

    print(
        "True noise std      :",
        f"{true_std:.6f}",
    )

    # =========================================================================
    # EXPLICIT DEVICE SYNCHRONIZATION TEST
    # =========================================================================

    test_sampler_device_transition(
        diffusion_model,
        schedule,
    )

    # =========================================================================
    # COMPLETE LATENT SAMPLER TEST
    # =========================================================================

    test_latent_sampler(
        autoencoder,
        diffusion_model,
        schedule,
    )

    # =========================================================================
    # GENERATE FINAL SAMPLES
    # =========================================================================

    sampling_time = None

    if VISUALIZE:

        (
            generated_latents,
            generated_images,
            sampling_time,
        ) = generate_and_visualize_samples(
            autoencoder,
            diffusion_model,
            schedule,
        )

    # =========================================================================
    # FINAL RESULT
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "LESSON 23 RESULT"
    )
    print("=" * 80)

    print()

    print(
        "Complete pipeline:"
    )

    print()

    print(
        "    image"
    )

    print(
        "      |"
    )

    print(
        "      v"
    )

    print(
        "    encoder"
    )

    print(
        "      |"
    )

    print(
        "      v"
    )

    print(
        "    latent"
    )

    print(
        "      |"
    )

    print(
        "      v"
    )

    print(
        "    diffusion"
    )

    print(
        "      |"
    )

    print(
        "      v"
    )

    print(
        "    generated latent"
    )

    print(
        "      |"
    )

    print(
        "      v"
    )

    print(
        "    decoder"
    )

    print(
        "      |"
    )

    print(
        "      v"
    )

    print(
        "    generated image"
    )

    print()

    print(
        f"Image representation : "
        f"{CHANNELS}x"
        f"{IMAGE_SIZE}x"
        f"{IMAGE_SIZE}"
    )

    print(
        f"Latent representation: "
        f"{LATENT_CHANNELS}x"
        f"{LATENT_SIZE}x"
        f"{LATENT_SIZE}"
    )

    print(
        f"Representation compression: "
        f"{compression_ratio:.2f}x"
    )

    print(
        f"Autoencoder MSE: "
        f"{reconstruction_mse:.6f}"
    )

    print(
        f"Latent noise MSE: "
        f"{noise_mse:.6f}"
    )

    if sampling_time is not None:

        print(
            f"Generation time: "
            f"{sampling_time:.2f} sec"
        )

    print()

    print(
        "The diffusion model operates on the latent representation,"
    )

    print(
        "not directly on the original image pixels."
    )

    print()

    print(
        "DEVICE SAFETY RESULT:"
    )

    print(
        "    model"
    )

    print(
        "    schedule"
    )

    print(
        "    timestep"
    )

    print(
        "    latent"
    )

    print(
        "all use the same runtime device during sampling."
    )

    print()

    print(
        "NEXT LESSON: CONDITIONING"
    )

    print(
        "We will learn how to control the generation process."
    )

    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()