"""
================================================================================
LESSON 22 — DDIM SAMPLING
================================================================================

Filename:
    lesson22_ddim_fixed.py

================================================================================
PURPOSE
================================================================================

Learn DDIM (Denoising Diffusion Implicit Models) sampling.

Previous lessons:

    Lesson 16  Forward diffusion
    Lesson 17  Noise schedules
    Lesson 18  Noise prediction
    Lesson 19  DDPM
    Lesson 20  Diffusion U-Net
    Lesson 21  DDPM sampling

This lesson:

    Lesson 22  DDIM

The major idea is to reduce the number of reverse diffusion steps.

DDPM:

    x_999 -> x_998 -> x_997 -> ... -> x_0

DDIM:

    x_999 -> x_959 -> x_918 -> ... -> x_0

Therefore DDIM can perform substantially fewer U-Net evaluations.

================================================================================
WHAT YOU WILL LEARN
================================================================================

1. DDIM theory.
2. DDIM timestep subsampling.
3. x0 reconstruction from predicted noise.
4. Deterministic DDIM.
5. Stochastic DDIM.
6. The eta parameter.
7. DDIM sampling trajectories.
8. DDPM versus DDIM.
9. Sampling speed versus number of steps.
10. Why diffusion inference speed matters in robotics.

================================================================================
TECHNICAL ROADMAP
================================================================================

Image
  |
  +-- CNN
  +-- Autoencoder
  +-- VAE
  +-- GAN
  +-- Vision Transformer
  |
  +-- Diffusion
       |
       +-- Forward diffusion
       +-- Noise schedule
       +-- Noise prediction
       +-- DDPM
       +-- U-Net
       +-- DDPM sampling
       +-- DDIM                  <-- CURRENT
       +-- Latent diffusion      <-- NEXT
       +-- Conditioning
       +-- Cross attention
       +-- Text conditioning
       +-- Own text-to-image model

================================================================================
REAL-WORLD PROBLEM
================================================================================

A diffusion model can require hundreds or thousands of neural-network
evaluations during generation.

That can be expensive for:

    robots
    embedded GPUs
    edge computers
    real-time visual systems

DDIM allows us to experiment with:

    1000 steps
    100 steps
    50 steps
    25 steps
    10 steps

and measure the computation/quality trade-off.

================================================================================
CRITICAL GROUPNORM FIX
================================================================================

The previous version contained an architectural bug.

The first residual block receives:

    input channels = 1

The invalid code was effectively:

    nn.GroupNorm(8, 1)

But GroupNorm requires:

    num_channels % num_groups == 0

For one channel:

    1 % 8 != 0

Therefore:

    GroupNorm(8, 1)

raises:

    ValueError:
    num_channels (1) must be divisible by num_groups (8)

This version uses:

    valid_group_count()

For example:

    channels=1
        groups=1

    channels=32
        groups=8

    channels=64
        groups=8

    channels=128
        groups=8

Thus every GroupNorm configuration is mathematically valid.

================================================================================
THEORY
================================================================================

Forward diffusion:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

The network predicts:

    epsilon_theta(x_t,t)

Therefore:

    x0_hat =
        [x_t -
         sqrt(1-alpha_bar_t) epsilon_theta]
        /
        sqrt(alpha_bar_t)

For DDIM, moving from timestep t to timestep s:

    s < t

we calculate:

    x_s =
        sqrt(alpha_bar_s) x0_hat
        +
        sqrt(1-alpha_bar_s-sigma_t^2) epsilon_theta
        +
        sigma_t z

where:

    z ~ N(0,I)

and:

    sigma_t =
        eta *
        sqrt((1-alpha_bar_s)/(1-alpha_bar_t)) *
        sqrt(1-alpha_bar_t/alpha_bar_s)

================================================================================
ETA
================================================================================

eta = 0:

    sigma = 0

The process is deterministic.

With:

    same model
    same initial noise
    same timestep sequence

we get the same result.

eta > 0:

    additional random noise is introduced.

Therefore the sampler becomes stochastic.

================================================================================
IMPLEMENTATION
================================================================================

This file implements:

    diffusion schedule
    forward diffusion
    synthetic images
    sinusoidal timestep embeddings
    residual blocks
    U-Net
    noise prediction
    x0 reconstruction
    DDIM sigma
    DDIM step
    DDIM sampler
    DDPM step
    DDPM sampler
    training
    tests
    visualizations
    benchmarks

No pretrained model is used.

================================================================================
REAL-WORLD ROBOTICS CONNECTION
================================================================================

Suppose an OpenArm robot has a camera observing a cluttered workspace.

A generative vision model might be used for:

    visual prediction
    synthetic image generation
    scene completion
    visual planning
    simulation augmentation

A 1000-step sampler may introduce too much latency.

A 25-step DDIM sampler can be much more practical.

Therefore:

    sampling speed
        |
        +-- affects perception latency
        |
        +-- affects control frequency
        |
        +-- affects robot responsiveness

================================================================================
REAL-WORLD RESULT
================================================================================

The important engineering lesson is:

    generative-model quality is only one part of deployment.

We must also consider:

    latency
    GPU computation
    memory
    throughput
    control-loop timing

================================================================================
NEXT LESSON
================================================================================

Lesson 23:

    LATENT DIFFUSION

Instead of applying diffusion directly to pixels:

    image -> pixel diffusion -> image

we will learn:

    image
       |
       v
    encoder
       |
       v
    latent
       |
       v
    diffusion
       |
       v
    latent
       |
       v
    decoder
       |
       v
    image

This introduces the central idea behind modern latent generative vision systems.

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 32
CHANNELS = 1

T = 1000

BETA_START = 1e-4
BETA_END = 0.02

BASE_CHANNELS = 32
TIME_EMBED_DIM = 128

BATCH_SIZE = 64
TRAIN_STEPS = 1800
LEARNING_RATE = 2e-3

NUM_SAMPLE_IMAGES = 8

DDIM_STEP_COUNTS = [100, 50, 25, 10]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DEVICE = torch.device("cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# =============================================================================
# DEVICE HELPERS
# =============================================================================

def get_model_device(model):

    return next(
        model.parameters()
    ).device


def move_schedule_to_device(
    schedule,
    device,
):

    return {
        key: value.to(device)
        for key, value in schedule.items()
    }


# =============================================================================
# CRITICAL GROUPNORM FIX
# =============================================================================

def valid_group_count(
    num_channels,
    preferred_groups=8,
):
    """
    Return a valid GroupNorm group count.

    The returned group count ALWAYS divides num_channels.

    Examples:

        valid_group_count(1)
            -> 1

        valid_group_count(32)
            -> 8

        valid_group_count(64)
            -> 8

        valid_group_count(128)
            -> 8
    """

    num_channels = int(
        num_channels
    )

    preferred_groups = int(
        preferred_groups
    )

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
# TEST GROUPNORM FIX FIRST
# =============================================================================

def test_groupnorm_fix():

    print()
    print("=" * 80)
    print("TEST 1 — GROUPNORM FIX")
    print("=" * 80)

    groups_1 = valid_group_count(1)
    groups_32 = valid_group_count(32)
    groups_64 = valid_group_count(64)
    groups_128 = valid_group_count(128)

    print(
        "channels=1   -> groups=",
        groups_1,
    )

    print(
        "channels=32  -> groups=",
        groups_32,
    )

    print(
        "channels=64  -> groups=",
        groups_64,
    )

    print(
        "channels=128 -> groups=",
        groups_128,
    )

    assert groups_1 == 1

    assert (
        32 % groups_32 == 0
    )

    assert (
        64 % groups_64 == 0
    )

    assert (
        128 % groups_128 == 0
    )

    # This is the exact operation that previously failed.
    norm = nn.GroupNorm(
        groups_1,
        1,
    )

    x = torch.randn(
        2,
        1,
        8,
        8,
    )

    y = norm(x)

    assert y.shape == x.shape

    assert torch.isfinite(
        y
    ).all()

    print(
        "GroupNorm(1, 1): PASS"
    )

    print("PASS")


# =============================================================================
# DIFFUSION SCHEDULE
# =============================================================================

def make_schedule():

    betas = torch.linspace(
        BETA_START,
        BETA_END,
        T,
        dtype=torch.float32,
    )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0,
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": torch.sqrt(
            alpha_bars
        ),
        "sqrt_one_minus_alpha_bars": torch.sqrt(
            1.0 - alpha_bars
        ),
    }


# =============================================================================
# EXTRACT
# =============================================================================

def extract(
    values,
    timesteps,
    shape,
):

    batch_size = timesteps.shape[0]

    result = values.gather(
        0,
        timesteps,
    )

    return result.reshape(
        batch_size,
        *([1] * (len(shape) - 1)),
    )


# =============================================================================
# FORWARD DIFFUSION
# =============================================================================

def forward_diffusion(
    x0,
    timesteps,
    noise,
    schedule,
):

    sqrt_alpha_bar = extract(
        schedule["sqrt_alpha_bars"],
        timesteps,
        x0.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule[
            "sqrt_one_minus_alpha_bars"
        ],
        timesteps,
        x0.shape,
    )

    return (
        sqrt_alpha_bar * x0
        +
        sqrt_one_minus_alpha_bar * noise
    )


# =============================================================================
# SYNTHETIC DATA
# =============================================================================

def generate_structured_images(
    batch_size,
    image_size=IMAGE_SIZE,
    device=DEVICE,
):

    images = torch.zeros(
        batch_size,
        1,
        image_size,
        image_size,
        device=device,
    )

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

    for i in range(batch_size):

        pattern = random.randint(
            0,
            4,
        )

        if pattern == 0:

            radius = random.uniform(
                0.25,
                0.55,
            )

            cx = random.uniform(
                -0.25,
                0.25,
            )

            cy = random.uniform(
                -0.25,
                0.25,
            )

            mask = (
                (xx - cx) ** 2
                +
                (yy - cy) ** 2
                <
                radius ** 2
            )

        elif pattern == 1:

            cx = random.uniform(
                -0.2,
                0.2,
            )

            cy = random.uniform(
                -0.2,
                0.2,
            )

            width = random.uniform(
                0.25,
                0.65,
            )

            height = random.uniform(
                0.25,
                0.65,
            )

            mask = (
                torch.abs(
                    xx - cx
                )
                < width / 2
            ) & (
                torch.abs(
                    yy - cy
                )
                < height / 2
            )

        elif pattern == 2:

            y0 = random.uniform(
                -0.5,
                0.5,
            )

            thickness = random.uniform(
                0.08,
                0.22,
            )

            mask = (
                torch.abs(
                    yy - y0
                )
                <
                thickness
            )

        elif pattern == 3:

            x0 = random.uniform(
                -0.5,
                0.5,
            )

            thickness = random.uniform(
                0.08,
                0.22,
            )

            mask = (
                torch.abs(
                    xx - x0
                )
                <
                thickness
            )

        else:

            slope = random.uniform(
                0.5,
                1.5,
            )

            offset = random.uniform(
                -0.4,
                0.4,
            )

            distance = torch.abs(
                yy
                -
                slope * xx
                -
                offset
            ) / math.sqrt(
                slope ** 2 + 1.0
            )

            thickness = random.uniform(
                0.05,
                0.12,
            )

            mask = (
                distance
                <
                thickness
            )

        images[
            i,
            0,
        ][mask] = 1.0

    return images


# =============================================================================
# TIME EMBEDDING
# =============================================================================

class SinusoidalTimeEmbedding(
    nn.Module
):

    def __init__(
        self,
        embedding_dim,
    ):

        super().__init__()

        self.embedding_dim = (
            embedding_dim
        )

    def forward(
        self,
        timesteps,
    ):

        half_dim = (
            self.embedding_dim // 2
        )

        exponent = (
            math.log(10000.0)
            /
            max(
                half_dim - 1,
                1,
            )
        )

        frequencies = torch.exp(
            -exponent
            *
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
        )

        arguments = (
            timesteps.float()
            .unsqueeze(1)
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

        if self.embedding_dim % 2:

            embedding = F.pad(
                embedding,
                (0, 1),
            )

        return embedding


# =============================================================================
# RESIDUAL BLOCK
# =============================================================================

class ResidualBlock(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
    ):

        super().__init__()

        # ================================================================
        # THIS IS THE IMPORTANT FIX
        # ================================================================

        groups_in = valid_group_count(
            in_channels
        )

        groups_out = valid_group_count(
            out_channels
        )

        self.norm1 = nn.GroupNorm(
            num_groups=groups_in,
            num_channels=in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Linear(
            time_embed_dim,
            out_channels,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
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
        x,
        time_embedding,
    ):

        residual = self.skip(x)

        h = self.norm1(x)

        h = F.silu(h)

        h = self.conv1(h)

        time_bias = (
            self.time_projection(
                time_embedding
            )
        )

        h = (
            h
            +
            time_bias[:, :, None, None]
        )

        h = self.norm2(h)

        h = F.silu(h)

        h = self.conv2(h)

        return h + residual


# =============================================================================
# DIFFUSION U-NET
# =============================================================================

class DiffusionUNet(
    nn.Module
):

    def __init__(
        self,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        base_channels=BASE_CHANNELS,
        time_embed_dim=TIME_EMBED_DIM,
    ):

        super().__init__()

        self.time_embedding = (
            SinusoidalTimeEmbedding(
                time_embed_dim
            )
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(
                time_embed_dim,
                time_embed_dim * 4,
            ),
            nn.SiLU(),
            nn.Linear(
                time_embed_dim * 4,
                time_embed_dim,
            ),
        )

        # ----------------------------------------------------------------
        # Encoder
        # ----------------------------------------------------------------

        self.enc1 = ResidualBlock(
            in_channels,
            base_channels,
            time_embed_dim,
        )

        self.down1 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc2 = ResidualBlock(
            base_channels,
            base_channels * 2,
            time_embed_dim,
        )

        self.down2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # ----------------------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------------------

        self.mid1 = ResidualBlock(
            base_channels * 2,
            base_channels * 4,
            time_embed_dim,
        )

        self.mid2 = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            time_embed_dim,
        )

        # ----------------------------------------------------------------
        # Decoder
        # ----------------------------------------------------------------

        self.up1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.dec1 = ResidualBlock(
            base_channels * 4,
            base_channels * 2,
            time_embed_dim,
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.dec2 = ResidualBlock(
            base_channels * 2,
            base_channels,
            time_embed_dim,
        )

        self.output_norm = nn.GroupNorm(
            num_groups=valid_group_count(
                base_channels
            ),
            num_channels=base_channels,
        )

        self.output_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x,
        timesteps,
    ):

        t = self.time_embedding(
            timesteps
        )

        t = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(
            x,
            t,
        )

        d1 = self.down1(
            e1
        )

        e2 = self.enc2(
            d1,
            t,
        )

        d2 = self.down2(
            e2
        )

        # Bottleneck
        h = self.mid1(
            d2,
            t,
        )

        h = self.mid2(
            h,
            t,
        )

        # Decoder
        h = self.up1(
            h
        )

        h = torch.cat(
            [
                h,
                e2,
            ],
            dim=1,
        )

        h = self.dec1(
            h,
            t,
        )

        h = self.up2(
            h
        )

        h = torch.cat(
            [
                h,
                e1,
            ],
            dim=1,
        )

        h = self.dec2(
            h,
            t,
        )

        h = self.output_norm(
            h
        )

        h = F.silu(
            h
        )

        return self.output_conv(
            h
        )


# =============================================================================
# X0 RECONSTRUCTION
# =============================================================================

def predict_x0(
    x_t,
    timesteps,
    predicted_noise,
    schedule,
):

    sqrt_alpha_bar = extract(
        schedule["sqrt_alpha_bars"],
        timesteps,
        x_t.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule[
            "sqrt_one_minus_alpha_bars"
        ],
        timesteps,
        x_t.shape,
    )

    return (
        x_t
        -
        sqrt_one_minus_alpha_bar
        *
        predicted_noise
    ) / torch.sqrt(
        torch.clamp(
            sqrt_alpha_bar ** 2,
            min=1e-12,
        )
    )


# =============================================================================
# DDIM SIGMA
# =============================================================================

def compute_ddim_sigma(
    alpha_bar_t,
    alpha_bar_s,
    eta,
):

    if eta == 0.0:

        return torch.zeros_like(
            alpha_bar_t
        )

    first = (
        1.0
        -
        alpha_bar_s
    ) / torch.clamp(
        1.0 - alpha_bar_t,
        min=1e-12,
    )

    second = (
        1.0
        -
        alpha_bar_t
        /
        torch.clamp(
            alpha_bar_s,
            min=1e-12,
        )
    )

    return (
        eta
        *
        torch.sqrt(
            torch.clamp(
                first,
                min=0.0,
            )
        )
        *
        torch.sqrt(
            torch.clamp(
                second,
                min=0.0,
            )
        )
    )


# =============================================================================
# DDIM STEP
# =============================================================================

@torch.no_grad()
def ddim_step(
    model,
    x_t,
    t,
    s,
    schedule,
    eta=0.0,
):

    device = x_t.device

    batch_size = (
        x_t.shape[0]
    )

    t_tensor = torch.full(
        (batch_size,),
        t,
        dtype=torch.long,
        device=device,
    )

    predicted_noise = model(
        x_t,
        t_tensor,
    )

    alpha_bar_t = schedule[
        "alpha_bars"
    ][t].to(
        device=device,
        dtype=x_t.dtype,
    )

    alpha_bar_s = schedule[
        "alpha_bars"
    ][s].to(
        device=device,
        dtype=x_t.dtype,
    )

    # ---------------------------------------------------------------
    # Estimate x0
    # ---------------------------------------------------------------

    x0_hat = (
        x_t
        -
        torch.sqrt(
            torch.clamp(
                1.0 - alpha_bar_t,
                min=0.0,
            )
        )
        *
        predicted_noise
    ) / torch.sqrt(
        torch.clamp(
            alpha_bar_t,
            min=1e-12,
        )
    )

    # ---------------------------------------------------------------
    # DDIM variance
    # ---------------------------------------------------------------

    sigma = compute_ddim_sigma(
        alpha_bar_t,
        alpha_bar_s,
        eta,
    )

    # ---------------------------------------------------------------
    # Direction toward x_t
    # ---------------------------------------------------------------

    direction_variance = (
        1.0
        -
        alpha_bar_s
        -
        sigma ** 2
    )

    direction_variance = torch.clamp(
        direction_variance,
        min=0.0,
    )

    direction = (
        torch.sqrt(
            direction_variance
        )
        *
        predicted_noise
    )

    # ---------------------------------------------------------------
    # Stochastic term
    # ---------------------------------------------------------------

    if (
        eta == 0.0
        or
        s == 0
    ):

        stochastic = torch.zeros_like(
            x_t
        )

    else:

        stochastic = (
            sigma
            *
            torch.randn_like(
                x_t
            )
        )

    # ---------------------------------------------------------------
    # DDIM equation
    # ---------------------------------------------------------------

    x_s = (
        torch.sqrt(
            torch.clamp(
                alpha_bar_s,
                min=1e-12,
            )
        )
        *
        x0_hat
        +
        direction
        +
        stochastic
    )

    return x_s


# =============================================================================
# DDPM STEP
# =============================================================================

@torch.no_grad()
def ddpm_step(
    model,
    x_t,
    t,
    schedule,
):

    device = x_t.device

    batch_size = x_t.shape[0]

    t_tensor = torch.full(
        (batch_size,),
        t,
        dtype=torch.long,
        device=device,
    )

    predicted_noise = model(
        x_t,
        t_tensor,
    )

    beta_t = schedule[
        "betas"
    ][t].to(
        device=device,
        dtype=x_t.dtype,
    )

    alpha_t = schedule[
        "alphas"
    ][t].to(
        device=device,
        dtype=x_t.dtype,
    )

    alpha_bar_t = schedule[
        "alpha_bars"
    ][t].to(
        device=device,
        dtype=x_t.dtype,
    )

    x0_hat = (
        x_t
        -
        torch.sqrt(
            1.0 - alpha_bar_t
        )
        *
        predicted_noise
    ) / torch.sqrt(
        torch.clamp(
            alpha_bar_t,
            min=1e-12,
        )
    )

    x0_hat = torch.clamp(
        x0_hat,
        -1.0,
        1.0,
    )

    if t == 0:

        return x0_hat

    alpha_bar_previous = schedule[
        "alpha_bars"
    ][t - 1].to(
        device=device,
        dtype=x_t.dtype,
    )

    posterior_variance = (
        beta_t
        *
        (
            1.0
            -
            alpha_bar_previous
        )
        /
        torch.clamp(
            1.0 - alpha_bar_t,
            min=1e-12,
        )
    )

    posterior_mean = (
        (
            torch.sqrt(
                alpha_bar_previous
            )
            *
            beta_t
            /
            torch.clamp(
                1.0 - alpha_bar_t,
                min=1e-12,
            )
            *
            x0_hat
        )
        +
        (
            torch.sqrt(
                alpha_t
            )
            *
            (
                1.0
                -
                alpha_bar_previous
            )
            /
            torch.clamp(
                1.0 - alpha_bar_t,
                min=1e-12,
            )
            *
            x_t
        )
    )

    return (
        posterior_mean
        +
        torch.sqrt(
            torch.clamp(
                posterior_variance,
                min=0.0,
            )
        )
        *
        torch.randn_like(x_t)
    )


# =============================================================================
# DDIM TIMESTEPS
# =============================================================================

def make_ddim_timesteps(
    num_steps,
):

    if num_steps < 2:

        raise ValueError(
            "num_steps must be >= 2"
        )

    if num_steps > T:

        raise ValueError(
            "num_steps cannot exceed T"
        )

    timesteps = torch.linspace(
        T - 1,
        0,
        num_steps,
    )

    timesteps = torch.round(
        timesteps
    ).long()

    # Make sure there are no duplicate values.
    if len(
        torch.unique(timesteps)
    ) != num_steps:

        raise RuntimeError(
            "DDIM timestep duplication detected."
        )

    return timesteps


# =============================================================================
# DDIM SAMPLER
# =============================================================================

@torch.no_grad()
def sample_ddim(
    model,
    schedule,
    num_steps,
    batch_size,
    eta=0.0,
    return_trajectory=False,
):

    device = get_model_device(
        model
    )

    schedule = move_schedule_to_device(
        schedule,
        device,
    )

    x = torch.randn(
        batch_size,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    timesteps = make_ddim_timesteps(
        num_steps
    )

    trajectory = {}

    if return_trajectory:

        trajectory[
            int(timesteps[0])
        ] = x.clone()

    for i in range(
        len(timesteps) - 1
    ):

        t = int(
            timesteps[i]
        )

        s = int(
            timesteps[i + 1]
        )

        x = ddim_step(
            model,
            x,
            t,
            s,
            schedule,
            eta,
        )

        if return_trajectory:

            trajectory[s] = x.clone()

    if return_trajectory:

        trajectory[0] = x.clone()

        return x, trajectory

    return x


# =============================================================================
# DDPM SAMPLER
# =============================================================================

@torch.no_grad()
def sample_ddpm(
    model,
    schedule,
    batch_size,
):

    device = get_model_device(
        model
    )

    schedule = move_schedule_to_device(
        schedule,
        device,
    )

    x = torch.randn(
        batch_size,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    for t in reversed(
        range(T)
    ):

        x = ddpm_step(
            model,
            x,
            t,
            schedule,
        )

    return x


# =============================================================================
# TEST 2 — SCHEDULE
# =============================================================================

def test_schedule():

    print()
    print("=" * 80)
    print("TEST 2 — DIFFUSION SCHEDULE")
    print("=" * 80)

    schedule = make_schedule()

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
        schedule["alpha_bars"][1:]
        <=
        schedule["alpha_bars"][:-1]
    )

    print(
        "beta[0]       :",
        schedule[
            "betas"
        ][0].item(),
    )

    print(
        "beta[-1]      :",
        schedule[
            "betas"
        ][-1].item(),
    )

    print(
        "alpha_bar[-1] :",
        schedule[
            "alpha_bars"
        ][-1].item(),
    )

    print("PASS")


# =============================================================================
# TEST 3 — U-NET OUTPUT
# =============================================================================

def test_unet_output():

    print()
    print("=" * 80)
    print("TEST 3 — U-NET OUTPUT")
    print("=" * 80)

    # IMPORTANT:
    #
    # This constructor now calls:
    #
    #     valid_group_count(1)
    #
    # before creating GroupNorm.
    #
    # Therefore the first block uses:
    #
    #     GroupNorm(1, 1)
    #
    # instead of:
    #
    #     GroupNorm(8, 1)

    model = DiffusionUNet().to(
        TEST_DEVICE
    )

    model.eval()

    x = torch.randn(
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=TEST_DEVICE,
    )

    timesteps = torch.tensor(
        [0, 100, 500, 999],
        dtype=torch.long,
        device=TEST_DEVICE,
    )

    with torch.no_grad():

        output = model(
            x,
            timesteps,
        )

    assert output.shape == x.shape

    assert torch.isfinite(
        output
    ).all()

    print(
        "Input shape :",
        tuple(x.shape),
    )

    print(
        "Output shape:",
        tuple(output.shape),
    )

    print("PASS")


# =============================================================================
# TEST 4 — FORWARD DIFFUSION
# =============================================================================

def test_forward_diffusion():

    print()
    print("=" * 80)
    print("TEST 4 — FORWARD DIFFUSION")
    print("=" * 80)

    schedule = make_schedule()

    x0 = torch.rand(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    timesteps = torch.tensor(
        [0, 100, 500, 999],
        dtype=torch.long,
    )

    noise = torch.randn_like(
        x0
    )

    x_t = forward_diffusion(
        x0,
        timesteps,
        noise,
        schedule,
    )

    assert x_t.shape == x0.shape

    assert torch.isfinite(
        x_t
    ).all()

    print(
        "Input shape :",
        tuple(x0.shape),
    )

    print(
        "Output shape:",
        tuple(x_t.shape),
    )

    print("PASS")


# =============================================================================
# TEST 5 — X0 RECONSTRUCTION
# =============================================================================

def test_x0_reconstruction():

    print()
    print("=" * 80)
    print("TEST 5 — X0 RECONSTRUCTION")
    print("=" * 80)

    schedule = make_schedule()

    x0 = torch.rand(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    timesteps = torch.tensor(
        [10, 100, 500, 900],
        dtype=torch.long,
    )

    noise = torch.randn_like(
        x0
    )

    x_t = forward_diffusion(
        x0,
        timesteps,
        noise,
        schedule,
    )

    reconstructed = predict_x0(
        x_t,
        timesteps,
        noise,
        schedule,
    )

    error = torch.max(
        torch.abs(
            reconstructed - x0
        )
    ).item()

    assert error < 1e-5

    print(
        "Maximum error:",
        f"{error:.8e}",
    )

    print("PASS")


# =============================================================================
# TEST 6 — DDIM SIGMA
# =============================================================================

def test_ddim_sigma():

    print()
    print("=" * 80)
    print("TEST 6 — DDIM SIGMA")
    print("=" * 80)

    schedule = make_schedule()

    alpha_bar_t = schedule[
        "alpha_bars"
    ][500]

    alpha_bar_s = schedule[
        "alpha_bars"
    ][250]

    sigma_zero = compute_ddim_sigma(
        alpha_bar_t,
        alpha_bar_s,
        0.0,
    )

    sigma_one = compute_ddim_sigma(
        alpha_bar_t,
        alpha_bar_s,
        1.0,
    )

    assert torch.allclose(
        sigma_zero,
        torch.tensor(0.0),
        atol=1e-7,
    )

    assert sigma_one >= 0

    print(
        "sigma eta=0:",
        sigma_zero.item(),
    )

    print(
        "sigma eta=1:",
        sigma_one.item(),
    )

    print("PASS")


# =============================================================================
# TEST 7 — ONE STEP
# =============================================================================

def test_ddim_step():

    print()
    print("=" * 80)
    print("TEST 7 — ONE-STEP DDIM")
    print("=" * 80)

    schedule = make_schedule()

    model = DiffusionUNet().to(
        TEST_DEVICE
    )

    model.eval()

    x_t = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    x_s = ddim_step(
        model,
        x_t,
        500,
        250,
        schedule,
        eta=0.0,
    )

    assert x_s.shape == x_t.shape

    assert torch.isfinite(
        x_s
    ).all()

    print(
        "Input shape :",
        tuple(x_t.shape),
    )

    print(
        "Output shape:",
        tuple(x_s.shape),
    )

    print("PASS")


# =============================================================================
# TEST 8 — DETERMINISTIC DDIM
# =============================================================================

def test_deterministic_ddim():

    print()
    print("=" * 80)
    print("TEST 8 — DETERMINISTIC DDIM")
    print("=" * 80)

    schedule = make_schedule()

    model = DiffusionUNet().to(
        TEST_DEVICE
    )

    model.eval()

    torch.manual_seed(
        123
    )

    initial_noise = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    def run_once():

        x = initial_noise.clone()

        timesteps = make_ddim_timesteps(
            25
        )

        for i in range(
            len(timesteps) - 1
        ):

            t = int(
                timesteps[i]
            )

            s = int(
                timesteps[i + 1]
            )

            x = ddim_step(
                model,
                x,
                t,
                s,
                schedule,
                eta=0.0,
            )

        return x

    first = run_once()
    second = run_once()

    difference = torch.max(
        torch.abs(
            first - second
        )
    ).item()

    assert difference < 1e-5

    print(
        "Maximum difference:",
        f"{difference:.8e}",
    )

    print("PASS")


# =============================================================================
# TEST 9 — STOCHASTIC DDIM
# =============================================================================

def test_stochastic_ddim():

    print()
    print("=" * 80)
    print("TEST 9 — STOCHASTIC DDIM")
    print("=" * 80)

    schedule = make_schedule()

    model = DiffusionUNet().to(
        TEST_DEVICE
    )

    model.eval()

    initial_noise = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    def run_once(seed):

        torch.manual_seed(
            seed
        )

        x = initial_noise.clone()

        timesteps = make_ddim_timesteps(
            25
        )

        for i in range(
            len(timesteps) - 1
        ):

            t = int(
                timesteps[i]
            )

            s = int(
                timesteps[i + 1]
            )

            x = ddim_step(
                model,
                x,
                t,
                s,
                schedule,
                eta=1.0,
            )

        return x

    first = run_once(100)
    second = run_once(200)

    difference = torch.mean(
        torch.abs(
            first - second
        )
    ).item()

    assert difference > 1e-7

    print(
        "Mean difference:",
        f"{difference:.8e}",
    )

    print("PASS")


# =============================================================================
# TEST 10 — TIMESTEPS
# =============================================================================

def test_timesteps():

    print()
    print("=" * 80)
    print("TEST 10 — DDIM TIMESTEPS")
    print("=" * 80)

    for steps in DDIM_STEP_COUNTS:

        timesteps = (
            make_ddim_timesteps(
                steps
            )
        )

        assert len(
            timesteps
        ) == steps

        assert (
            timesteps[0].item()
            ==
            T - 1
        )

        assert (
            timesteps[-1].item()
            ==
            0
        )

        assert torch.all(
            timesteps[:-1]
            >
            timesteps[1:]
        )

        print(
            f"{steps:3d} steps -> PASS"
        )

    print("PASS")


# =============================================================================
# TEST 11 — COMPLETE SAMPLER
# =============================================================================

def test_complete_sampler():

    print()
    print("=" * 80)
    print("TEST 11 — COMPLETE DDIM SAMPLER")
    print("=" * 80)

    schedule = make_schedule()

    model = DiffusionUNet().to(
        TEST_DEVICE
    )

    model.eval()

    samples, trajectory = sample_ddim(
        model,
        schedule,
        num_steps=25,
        batch_size=4,
        eta=0.0,
        return_trajectory=True,
    )

    assert samples.shape == (
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert torch.isfinite(
        samples
    ).all()

    assert (
        T - 1
    ) in trajectory

    assert 0 in trajectory

    print(
        "Sample shape:",
        tuple(samples.shape),
    )

    print(
        "Trajectory entries:",
        len(trajectory),
    )

    print("PASS")


# =============================================================================
# TEST 12 — MULTIPLE STEP COUNTS
# =============================================================================

def test_multiple_steps():

    print()
    print("=" * 80)
    print("TEST 12 — MULTIPLE DDIM STEP COUNTS")
    print("=" * 80)

    schedule = make_schedule()

    model = DiffusionUNet().to(
        TEST_DEVICE
    )

    model.eval()

    for steps in DDIM_STEP_COUNTS:

        samples = sample_ddim(
            model,
            schedule,
            steps,
            batch_size=2,
            eta=0.0,
        )

        assert samples.shape == (
            2,
            CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        assert torch.isfinite(
            samples
        ).all()

        print(
            f"{steps:3d} steps -> PASS"
        )

    print("PASS")


# =============================================================================
# TEST 13 — DEVICE
# =============================================================================

def test_device():

    print()
    print("=" * 80)
    print("TEST 13 — DEVICE CONSISTENCY")
    print("=" * 80)

    model = DiffusionUNet().to(
        DEVICE
    )

    schedule = make_schedule()

    model_device = (
        get_model_device(model)
    )

    schedule = (
        move_schedule_to_device(
            schedule,
            model_device,
        )
    )

    x = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=model_device,
    )

    timesteps = torch.randint(
        0,
        T,
        (2,),
        device=model_device,
    )

    output = model(
        x,
        timesteps,
    )

    assert output.device == model_device

    for tensor in schedule.values():

        assert (
            tensor.device
            ==
            model_device
        )

    print(
        "Model device   :",
        model_device,
    )

    print(
        "Input device   :",
        x.device,
    )

    print(
        "Output device  :",
        output.device,
    )

    print("PASS")


# =============================================================================
# VISUALIZATION
# =============================================================================

def show_images(
    images,
    title,
    max_images=8,
):

    images = images.detach().cpu()

    images = images[
        :max_images
    ]

    count = len(images)

    columns = min(
        4,
        count,
    )

    rows = math.ceil(
        count / columns
    )

    plt.figure(
        figsize=(
            3 * columns,
            3 * rows,
        )
    )

    for i in range(count):

        ax = plt.subplot(
            rows,
            columns,
            i + 1,
        )

        ax.imshow(
            images[i, 0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        ax.axis("off")

    plt.suptitle(
        title
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# TRAINING LOSS PLOT
# =============================================================================

def plot_loss(
    losses,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        losses
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "DDIM / Diffusion U-Net Training"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# TRAJECTORY PLOT
# =============================================================================

def show_trajectory(
    trajectory,
):

    timesteps = sorted(
        trajectory.keys(),
        reverse=True,
    )

    if len(timesteps) > 8:

        indices = np.linspace(
            0,
            len(timesteps) - 1,
            8,
        ).round().astype(int)

        timesteps = [
            timesteps[i]
            for i in indices
        ]

    plt.figure(
        figsize=(
            3 * len(timesteps),
            3,
        )
    )

    for i, timestep in enumerate(
        timesteps
    ):

        ax = plt.subplot(
            1,
            len(timesteps),
            i + 1,
        )

        image = (
            trajectory[timestep][0, 0]
            .detach()
            .cpu()
        )

        ax.imshow(
            image,
            cmap="gray",
        )

        ax.set_title(
            f"t={timestep}"
        )

        ax.axis("off")

    plt.suptitle(
        "DDIM Denoising Trajectory"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# COMPARISON
# =============================================================================

def show_ddpm_ddim(
    ddpm_samples,
    ddim_samples,
):

    count = min(
        4,
        ddpm_samples.shape[0],
        ddim_samples.shape[0],
    )

    plt.figure(
        figsize=(
            3 * count,
            6,
        )
    )

    for i in range(count):

        ax = plt.subplot(
            2,
            count,
            i + 1,
        )

        ax.imshow(
            torch.clamp(
                ddpm_samples[i, 0],
                0,
                1,
            ).cpu(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            "DDPM"
        )

        ax.axis("off")

        ax = plt.subplot(
            2,
            count,
            count + i + 1,
        )

        ax.imshow(
            torch.clamp(
                ddim_samples[i, 0],
                0,
                1,
            ).cpu(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            "DDIM"
        )

        ax.axis("off")

    plt.suptitle(
        "DDPM versus DDIM"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# BENCHMARK
# =============================================================================

@torch.no_grad()
def benchmark(
    model,
    schedule,
):

    results = []

    for steps in DDIM_STEP_COUNTS:

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        samples = sample_ddim(
            model,
            schedule,
            steps,
            batch_size=4,
            eta=0.0,
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            -
            start
        )

        results.append(
            (
                steps,
                elapsed,
                samples,
            )
        )

    return results


# =============================================================================
# BENCHMARK PLOT
# =============================================================================

def plot_benchmark(
    results,
):

    steps = [
        item[0]
        for item in results
    ]

    times = [
        item[1]
        for item in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        steps,
        times,
        marker="o",
    )

    plt.xlabel(
        "DDIM sampling steps"
    )

    plt.ylabel(
        "Time (seconds)"
    )

    plt.title(
        "DDIM Sampling Cost"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def count_parameters(
    model,
):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    seed_everything()

    print()
    print("=" * 80)
    print("LESSON 22 — DDIM")
    print("=" * 80)

    print(
        "PyTorch version :",
        torch.__version__,
    )

    print(
        "Device          :",
        DEVICE,
    )

    print(
        "Test device     :",
        TEST_DEVICE,
    )

    print(
        "T               :",
        T,
    )

    print(
        "Image size      :",
        f"{IMAGE_SIZE}x{IMAGE_SIZE}",
    )

    # =========================================================================
    # TESTS
    # =========================================================================

    # The GroupNorm test is intentionally FIRST.
    test_groupnorm_fix()

    test_schedule()

    test_unet_output()

    test_forward_diffusion()

    test_x0_reconstruction()

    test_ddim_sigma()

    test_ddim_step()

    test_deterministic_ddim()

    test_stochastic_ddim()

    test_timesteps()

    test_complete_sampler()

    test_multiple_steps()

    test_device()

    print()
    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)

    # =========================================================================
    # MODEL
    # =========================================================================

    model = DiffusionUNet().to(
        DEVICE
    )

    schedule = make_schedule()

    schedule = (
        move_schedule_to_device(
            schedule,
            DEVICE,
        )
    )

    print()
    print(
        "Model parameters:",
        f"{count_parameters(model):,}",
    )

    # =========================================================================
    # TRAINING DATA VISUALIZATION
    # =========================================================================

    training_images = (
        generate_structured_images(
            8,
            IMAGE_SIZE,
            DEVICE,
        )
    )

    show_images(
        training_images,
        "Synthetic Training Images",
    )

    # =========================================================================
    # TRAINING
    # =========================================================================

    print()
    print("=" * 80)
    print("TRAINING DIFFUSION U-NET")
    print("=" * 80)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    losses = []

    start_time = time.perf_counter()

    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):

        x0 = generate_structured_images(
            BATCH_SIZE,
            IMAGE_SIZE,
            DEVICE,
        )

        timesteps = torch.randint(
            0,
            T,
            (BATCH_SIZE,),
            dtype=torch.long,
            device=DEVICE,
        )

        noise = torch.randn_like(
            x0
        )

        x_t = forward_diffusion(
            x0,
            timesteps,
            noise,
            schedule,
        )

        predicted_noise = model(
            x_t,
            timesteps,
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
            1.0,
        )

        optimizer.step()

        loss_value = loss.item()

        losses.append(
            loss_value
        )

        if (
            step == 1
            or step % 100 == 0
            or step == TRAIN_STEPS
        ):

            print(
                f"step {step:4d}/{TRAIN_STEPS} "
                f"| loss={loss_value:.6f}"
            )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    print()
    print(
        f"Training time: {elapsed:.2f} seconds"
    )

    print(
        f"Initial loss: {losses[0]:.6f}"
    )

    print(
        f"Final loss:   {losses[-1]:.6f}"
    )

    print(
        f"Best loss:    {min(losses):.6f}"
    )

    plot_loss(
        losses
    )

    # =========================================================================
    # NOISE PREDICTION TEST
    # =========================================================================

    print()
    print("=" * 80)
    print("NOISE PREDICTION EVALUATION")
    print("=" * 80)

    model.eval()

    with torch.no_grad():

        x0 = generate_structured_images(
            8,
            IMAGE_SIZE,
            DEVICE,
        )

        timesteps = torch.randint(
            0,
            T,
            (8,),
            dtype=torch.long,
            device=DEVICE,
        )

        true_noise = torch.randn_like(
            x0
        )

        x_t = forward_diffusion(
            x0,
            timesteps,
            true_noise,
            schedule,
        )

        predicted_noise = model(
            x_t,
            timesteps,
        )

        mse = F.mse_loss(
            predicted_noise,
            true_noise,
        )

        print(
            "Noise prediction MSE:",
            mse.item(),
        )

        print(
            "True noise std:",
            true_noise.std().item(),
        )

        print(
            "Predicted noise std:",
            predicted_noise.std().item(),
        )

    # =========================================================================
    # DDIM 50 STEP
    # =========================================================================

    print()
    print("=" * 80)
    print("DDIM — 50 STEPS — ETA=0")
    print("=" * 80)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    ddim_samples, trajectory = sample_ddim(
        model,
        schedule,
        num_steps=50,
        batch_size=NUM_SAMPLE_IMAGES,
        eta=0.0,
        return_trajectory=True,
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    print(
        f"DDIM 50-step time: "
        f"{elapsed:.4f} seconds"
    )

    show_images(
        torch.clamp(
            ddim_samples,
            0,
            1,
        ),
        "DDIM — 50 Steps — eta=0",
    )

    show_trajectory(
        trajectory
    )

    # =========================================================================
    # STOCHASTIC DDIM
    # =========================================================================

    print()
    print("=" * 80)
    print("DDIM — 50 STEPS — ETA=1")
    print("=" * 80)

    stochastic_samples = sample_ddim(
        model,
        schedule,
        num_steps=50,
        batch_size=NUM_SAMPLE_IMAGES,
        eta=1.0,
    )

    show_images(
        torch.clamp(
            stochastic_samples,
            0,
            1,
        ),
        "DDIM — 50 Steps — eta=1",
    )

    # =========================================================================
    # DDPM
    # =========================================================================

    print()
    print("=" * 80)
    print("DDPM — 1000 STEPS")
    print("=" * 80)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    ddpm_samples = sample_ddpm(
        model,
        schedule,
        NUM_SAMPLE_IMAGES,
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    print(
        f"DDPM 1000-step time: "
        f"{elapsed:.4f} seconds"
    )

    show_images(
        torch.clamp(
            ddpm_samples,
            0,
            1,
        ),
        "DDPM — 1000 Steps",
    )

    # =========================================================================
    # COMPARISON
    # =========================================================================

    show_ddpm_ddim(
        ddpm_samples,
        ddim_samples,
    )

    # =========================================================================
    # BENCHMARK
    # =========================================================================

    print()
    print("=" * 80)
    print("DDIM STEP COUNT BENCHMARK")
    print("=" * 80)

    results = benchmark(
        model,
        schedule,
    )

    for steps, elapsed, _ in results:

        print(
            f"{steps:3d} steps -> "
            f"{elapsed:.4f} seconds"
        )

    plot_benchmark(
        results
    )

    # =========================================================================
    # FINAL RESULT
    # =========================================================================

    print()
    print("=" * 80)
    print("LESSON 22 — RESULT")
    print("=" * 80)

    print()
    print(
        "1. DDPM uses a long reverse diffusion trajectory."
    )

    print(
        "2. DDIM can use a shorter timestep sequence."
    )

    print(
        "3. eta=0 produces deterministic sampling."
    )

    print(
        "4. eta>0 introduces stochasticity."
    )

    print(
        "5. Fewer DDIM steps reduce U-Net evaluations."
    )

    print(
        "6. Sampling speed matters for real-time robotics."
    )

    print()
    print(
        "REAL-WORLD ROBOTICS CONNECTION:"
    )

    print(
        "A faster generative vision model can reduce "
        "visual inference latency in a robot perception "
        "or planning pipeline."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Lesson 23 — Latent Diffusion"
    )

    print()
    print("=" * 80)
    print("LESSON 22 COMPLETE")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()