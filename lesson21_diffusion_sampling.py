"""
================================================================================
LESSON 21 — DIFFUSION SAMPLING
================================================================================

Filename:
    lesson21_diffusion_sampling.py

================================================================================
PURPOSE
================================================================================

This lesson implements the COMPLETE DDPM reverse sampling process.

Lesson 20 taught the diffusion U-Net:

    x_t + t
       |
       v
    predicted noise epsilon_theta(x_t,t)

This lesson answers:

    How do we turn Gaussian noise back into an image?

The complete generation process is:

    x_T ~ N(0,I)
       |
       v
    x_(T-1)
       |
       v
      ...
       |
       v
    x_1
       |
       v
    x_0

Everything is implemented from scratch.

No pretrained models.
No pretrained weights.
No checkpoints.

================================================================================
WHAT YOU WILL LEARN
================================================================================

1. Gaussian noise initialization.
2. The DDPM reverse equation.
3. Reverse posterior mean.
4. Reverse posterior variance.
5. Stochastic reverse transitions.
6. Deterministic final step.
7. Complete T-step DDPM sampling.
8. Reverse trajectory recording.
9. X0 reconstruction from predicted noise.
10. Sampling speed limitations.
11. Why naive timestep skipping is not DDIM.
12. Why DDIM is needed for faster generation.

================================================================================
COURSE ROADMAP
================================================================================

Lesson 01  Images as numerical data
Lesson 02  Vectors, matrices, tensors
Lesson 03  Gradient descent
Lesson 04  Neural networks
Lesson 05  Convolution
Lesson 06  CNN from scratch
Lesson 07  Practical CNN training
Lesson 08  Deep CNN / residual learning
Lesson 09  Autoencoder
Lesson 10  VAE
Lesson 11  GAN
Lesson 12  Generative modeling foundations
Lesson 13  Vision Transformer
Lesson 14  Generative modeling fundamentals
Lesson 15  Probability distributions
Lesson 16  Forward diffusion
Lesson 17  Noise schedules
Lesson 18  Learning to predict noise
Lesson 19  DDPM
Lesson 20  Diffusion U-Net
Lesson 21  Diffusion Sampling       <-- CURRENT
Lesson 22  DDIM
Lesson 23  Latent Diffusion
Lesson 24  Conditioning
Lesson 25  Cross-Attention
Lesson 26  Text Conditioning
Lesson 27  Own Text-to-Image Model

================================================================================
REAL-WORLD PROBLEM
================================================================================

A diffusion model may be used in a robotics system for:

    - synthetic image generation
    - image restoration
    - visual augmentation
    - perception data generation
    - world-model generation
    - manipulation-policy training

But standard DDPM sampling may require:

    T = 1000

neural-network evaluations.

That is expensive for a real-time robot.

Therefore this lesson establishes the standard sampler first.

Lesson 22 will then introduce DDIM to reduce the number of evaluations.

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

The DDPM reverse mean is:

    mu_theta(x_t,t)
        =
        1/sqrt(alpha_t)
        *
        [
            x_t
            -
            beta_t / sqrt(1-alpha_bar_t)
            *
            epsilon_theta(x_t,t)
        ]

For t > 1:

    x_(t-1)
        =
        mu_theta
        +
        sqrt(posterior_variance_t) z

where:

    z ~ N(0,I)

For t = 1:

    x_0 = mu_theta

No additional random noise is injected into the final step.

================================================================================
NUMERICAL PRECISION
================================================================================

CUDA float32 arithmetic is not mathematically exact.

Therefore a deterministic final step can still produce tiny numerical
differences such as:

    2.38e-7

That is normal floating-point behavior.

We therefore use:

    FINAL_STEP_ATOL = 1e-6

rather than requiring exact bitwise equality.

================================================================================
DEVICE DESIGN
================================================================================

The implementation supports:

    CUDA
    CPU

The runtime device is established once:

    DEVICE = cuda if available else cpu

However, tests should not blindly assume that a module is on the global
device variable.

Instead we explicitly verify:

    model device
    schedule device
    input device
    sampler output device

and make sure they agree.

This is more robust than testing only:

    model_device == DEVICE

because the actual tensor/module device is the source of truth.

================================================================================
IMPLEMENTATION
================================================================================
"""

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

SAMPLE_COUNT = 8

FINAL_STEP_ATOL = 1e-6

# Runtime device.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DEVICE = torch.device("cpu")
TEST_DTYPE = torch.float64


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DEVICE HELPERS
# =============================================================================

def get_model_device(model):
    """
    Return the actual device occupied by model parameters.

    This is intentionally the source of truth for model-device checks.
    """

    return next(
        model.parameters()
    ).device


def move_schedule_to_device(
    schedule,
    device,
):
    """
    Explicitly move every schedule tensor to the requested device.

    This prevents accidental CPU/CUDA mixing.
    """

    return {
        key: value.to(device)
        for key, value in schedule.items()
    }


def synchronize_model_and_schedule(
    model,
    schedule,
):
    """
    Synchronize the model and diffusion schedule.

    The model's actual device is authoritative.

    This is safer than assuming that a global DEVICE variable always
    represents the current model location.
    """

    model_device = get_model_device(model)

    schedule = move_schedule_to_device(
        schedule,
        model_device,
    )

    return model, schedule


# =============================================================================
# SYNTHETIC STRUCTURED DATA
# =============================================================================

def create_structured_images(
    batch_size,
    image_size=IMAGE_SIZE,
    device=DEVICE,
):

    y = torch.linspace(
        -1.0,
        1.0,
        image_size,
        device=device,
    )

    x = torch.linspace(
        -1.0,
        1.0,
        image_size,
        device=device,
    )

    yy, xx = torch.meshgrid(
        y,
        x,
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

        mode = int(
            torch.randint(
                0,
                4,
                (1,),
                device=device,
            ).item()
        )

        if mode == 0:

            cx = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.45,
                0.45,
            )

            cy = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.45,
                0.45,
            )

            sigma = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.15,
                0.35,
            )

            image = torch.exp(
                -(
                    (xx[0] - cx) ** 2
                    +
                    (yy[0] - cy) ** 2
                )
                /
                (2.0 * sigma ** 2)
            )

        elif mode == 1:

            cy = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.5,
                0.5,
            )

            width = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.08,
                0.20,
            )

            image = torch.exp(
                -((yy[0] - cy) ** 2)
                /
                (2.0 * width ** 2)
            )

        elif mode == 2:

            cx = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.5,
                0.5,
            )

            width = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.08,
                0.20,
            )

            image = torch.exp(
                -((xx[0] - cx) ** 2)
                /
                (2.0 * width ** 2)
            )

        else:

            offset = torch.empty(
                1,
                device=device,
            ).uniform_(
                -0.4,
                0.4,
            )

            width = torch.empty(
                1,
                device=device,
            ).uniform_(
                0.07,
                0.16,
            )

            image = torch.exp(
                -(
                    yy[0]
                    -
                    xx[0]
                    -
                    offset
                ) ** 2
                /
                (2.0 * width ** 2)
            )

        cx2 = torch.empty(
            1,
            device=device,
        ).uniform_(
            -0.6,
            0.6,
        )

        cy2 = torch.empty(
            1,
            device=device,
        ).uniform_(
            -0.6,
            0.6,
        )

        blob2 = torch.exp(
            -(
                (xx[0] - cx2) ** 2
                +
                (yy[0] - cy2) ** 2
            )
            /
            (2.0 * 0.18 ** 2)
        )

        image = (
            0.75 * image
            +
            0.25 * blob2
        )

        images[i, 0] = image

    return images.clamp(
        0.0,
        1.0,
    )


# =============================================================================
# DIFFUSION SCHEDULE
# =============================================================================

def make_schedule(
    T_value=T,
    beta_start=BETA_START,
    beta_end=BETA_END,
    device=DEVICE,
):

    beta = torch.linspace(
        beta_start,
        beta_end,
        T_value,
        device=device,
        dtype=torch.float32,
    )

    alpha = 1.0 - beta

    alpha_bar = torch.cumprod(
        alpha,
        dim=0,
    )

    alpha_bar_prev = torch.cat(
        [
            torch.ones(
                1,
                device=device,
                dtype=alpha.dtype,
            ),
            alpha_bar[:-1],
        ]
    )

    sqrt_alpha_bar = torch.sqrt(
        alpha_bar
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - alpha_bar
    )

    posterior_variance = (
        beta
        *
        (1.0 - alpha_bar_prev)
        /
        (1.0 - alpha_bar)
    )

    posterior_variance = posterior_variance.clamp(
        min=1e-20
    )

    posterior_log_variance = torch.log(
        posterior_variance
    )

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "posterior_variance": posterior_variance,
        "posterior_log_variance": posterior_log_variance,
    }


# =============================================================================
# EXTRACT SCHEDULE VALUES
# =============================================================================

def extract(
    values,
    timesteps,
    x_shape,
):

    batch_size = timesteps.shape[0]

    out = values.gather(
        0,
        timesteps,
    )

    return out.reshape(
        batch_size,
        *((1,) * (len(x_shape) - 1)),
    )


# =============================================================================
# FORWARD DIFFUSION
# =============================================================================

def q_sample(
    x0,
    timesteps,
    schedule,
    noise=None,
):

    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alpha_bar_t = extract(
        schedule["sqrt_alpha_bar"],
        timesteps,
        x0.shape,
    ).to(
        device=x0.device,
        dtype=x0.dtype,
    )

    sqrt_one_minus_alpha_bar_t = extract(
        schedule[
            "sqrt_one_minus_alpha_bar"
        ],
        timesteps,
        x0.shape,
    ).to(
        device=x0.device,
        dtype=x0.dtype,
    )

    xt = (
        sqrt_alpha_bar_t * x0
        +
        sqrt_one_minus_alpha_bar_t
        * noise
    )

    return xt, noise


# =============================================================================
# TIME EMBEDDING
# =============================================================================

def sinusoidal_time_embedding(
    timesteps,
    embedding_dim,
):

    half_dim = embedding_dim // 2

    exponent = (
        -math.log(10000.0)
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
        * exponent
    )

    angles = (
        timesteps.float().unsqueeze(1)
        *
        frequencies.unsqueeze(0)
    )

    embedding = torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=1,
    )

    if embedding_dim % 2 == 1:

        embedding = F.pad(
            embedding,
            (0, 1),
        )

    return embedding


# =============================================================================
# RESIDUAL BLOCK
# =============================================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim,
        groups=8,
    ):

        super().__init__()

        self.norm1 = nn.GroupNorm(
            min(groups, in_channels),
            in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1,
        )

        self.time_projection = nn.Linear(
            time_dim,
            out_channels,
        )

        self.norm2 = nn.GroupNorm(
            min(groups, out_channels),
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
        )

        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                1,
            )

        else:

            self.skip = nn.Identity()

    def forward(
        self,
        x,
        time_embedding,
    ):

        h = self.norm1(x)

        h = F.silu(h)

        h = self.conv1(h)

        time_term = self.time_projection(
            time_embedding
        )

        time_term = time_term[
            :,
            :,
            None,
            None,
        ]

        h = h + time_term

        h = self.norm2(h)

        h = F.silu(h)

        h = self.conv2(h)

        return h + self.skip(x)


# =============================================================================
# DIFFUSION U-NET
# =============================================================================

class DiffusionUNet(nn.Module):

    def __init__(
        self,
        in_channels=CHANNELS,
        base_channels=BASE_CHANNELS,
        time_dim=TIME_EMBED_DIM,
    ):

        super().__init__()

        self.time_dim = time_dim

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
            in_channels,
            base_channels,
            3,
            padding=1,
        )

        self.enc1 = ResidualBlock(
            base_channels,
            base_channels,
            time_dim,
        )

        self.down1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            4,
            stride=2,
            padding=1,
        )

        self.enc2 = ResidualBlock(
            base_channels * 2,
            base_channels * 2,
            time_dim,
        )

        self.down2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            4,
            stride=2,
            padding=1,
        )

        self.bottleneck = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            time_dim,
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            4,
            stride=2,
            padding=1,
        )

        self.dec2 = ResidualBlock(
            base_channels * 4,
            base_channels * 2,
            time_dim,
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            4,
            stride=2,
            padding=1,
        )

        self.dec1 = ResidualBlock(
            base_channels * 2,
            base_channels,
            time_dim,
        )

        self.output_norm = nn.GroupNorm(
            8,
            base_channels,
        )

        self.output_conv = nn.Conv2d(
            base_channels,
            in_channels,
            3,
            padding=1,
        )

    def forward(
        self,
        x,
        timesteps,
    ):

        t = sinusoidal_time_embedding(
            timesteps,
            self.time_dim,
        )

        t = self.time_mlp(t)

        x0 = self.input_conv(x)

        e1 = self.enc1(
            x0,
            t,
        )

        e2 = self.enc2(
            self.down1(e1),
            t,
        )

        b = self.bottleneck(
            self.down2(e2),
            t,
        )

        d2 = self.up2(b)

        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        d2 = self.dec2(
            d2,
            t,
        )

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        d1 = self.dec1(
            d1,
            t,
        )

        out = self.output_norm(d1)

        out = F.silu(out)

        out = self.output_conv(out)

        return out


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_model(model):

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.Linear,
            ),
        ):

            nn.init.xavier_uniform_(
                module.weight
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )


# =============================================================================
# TRAINING
# =============================================================================

def train_diffusion_model(
    model,
    schedule,
    steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
):

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    losses = []

    start = time.perf_counter()

    for step in range(
        1,
        steps + 1,
    ):

        model_device = get_model_device(
            model
        )

        x0 = create_structured_images(
            batch_size,
            IMAGE_SIZE,
            model_device,
        )

        timesteps = torch.randint(
            1,
            T + 1,
            (batch_size,),
            device=model_device,
            dtype=torch.long,
        )

        timestep_indices = timesteps - 1

        xt, true_noise = q_sample(
            x0,
            timestep_indices,
            schedule,
        )

        predicted_noise = model(
            xt,
            timestep_indices,
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

        losses.append(
            float(
                loss.detach().cpu()
            )
        )

        if (
            step == 1
            or step % 200 == 0
            or step == steps
        ):

            recent = np.mean(
                losses[-100:]
            )

            print(
                f"Step {step:4d} | "
                f"loss {losses[-1]:.6f} | "
                f"100-step mean {recent:.6f}"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Training time: {elapsed:.2f} sec"
    )

    return losses


# =============================================================================
# DDPM REVERSE STEP
# =============================================================================

@torch.no_grad()
def reverse_step(
    model,
    xt,
    timestep,
    schedule,
):

    batch_size = xt.shape[0]

    model_device = get_model_device(
        model
    )

    # Ensure input is on the actual model device.
    xt = xt.to(
        device=model_device
    )

    timestep_index = timestep - 1

    timestep_tensor = torch.full(
        (batch_size,),
        timestep_index,
        device=model_device,
        dtype=torch.long,
    )

    predicted_noise = model(
        xt,
        timestep_tensor,
    )

    beta_t = schedule["beta"][
        timestep_index
    ].to(
        device=model_device,
        dtype=xt.dtype,
    )

    alpha_t = schedule["alpha"][
        timestep_index
    ].to(
        device=model_device,
        dtype=xt.dtype,
    )

    alpha_bar_t = schedule[
        "alpha_bar"
    ][
        timestep_index
    ].to(
        device=model_device,
        dtype=xt.dtype,
    )

    posterior_variance_t = schedule[
        "posterior_variance"
    ][
        timestep_index
    ].to(
        device=model_device,
        dtype=xt.dtype,
    )

    mean = (
        1.0
        /
        torch.sqrt(alpha_t)
        *
        (
            xt
            -
            beta_t
            /
            torch.sqrt(
                1.0 - alpha_bar_t
            )
            *
            predicted_noise
        )
    )

    if timestep > 1:

        noise = torch.randn_like(
            xt
        )

        xt_previous = (
            mean
            +
            torch.sqrt(
                posterior_variance_t
            )
            *
            noise
        )

    else:

        xt_previous = mean

    return xt_previous


# =============================================================================
# X0 RECONSTRUCTION
# =============================================================================

@torch.no_grad()
def predict_x0_from_xt(
    xt,
    predicted_noise,
    timestep_index,
    schedule,
):

    sqrt_alpha_bar_t = schedule[
        "sqrt_alpha_bar"
    ][
        timestep_index
    ].to(
        device=xt.device,
        dtype=xt.dtype,
    )

    sqrt_one_minus_alpha_bar_t = schedule[
        "sqrt_one_minus_alpha_bar"
    ][
        timestep_index
    ].to(
        device=xt.device,
        dtype=xt.dtype,
    )

    return (
        xt
        -
        sqrt_one_minus_alpha_bar_t
        *
        predicted_noise
    ) / sqrt_alpha_bar_t


# =============================================================================
# TRAJECTORY CONTRACT
# =============================================================================

TRAJECTORY_CHECKPOINTS = (
    T,
    750,
    500,
    250,
    100,
    50,
    25,
    10,
    1,
    0,
)


def make_trajectory_recorder():

    return {
        key: None
        for key in TRAJECTORY_CHECKPOINTS
    }


# =============================================================================
# COMPLETE DDPM SAMPLER
# =============================================================================

@torch.no_grad()
def sample_ddpm(
    model,
    schedule,
    batch_size=SAMPLE_COUNT,
    return_trajectory=True,
):

    model.eval()

    model_device = get_model_device(
        model
    )

    schedule = move_schedule_to_device(
        schedule,
        model_device,
    )

    x = torch.randn(
        batch_size,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=model_device,
    )

    trajectory = (
        make_trajectory_recorder()
        if return_trajectory
        else None
    )

    if return_trajectory:

        trajectory[T] = (
            x.detach().clone()
        )

    for timestep in range(
        T,
        0,
        -1,
    ):

        x = reverse_step(
            model,
            x,
            timestep,
            schedule,
        )

        checkpoint = timestep - 1

        if (
            return_trajectory
            and checkpoint in trajectory
        ):

            trajectory[
                checkpoint
            ] = x.detach().clone()

    if return_trajectory:

        trajectory[0] = (
            x.detach().clone()
        )

        missing = [
            key
            for key, value
            in trajectory.items()
            if value is None
        ]

        if missing:

            raise RuntimeError(
                "Missing trajectory keys: "
                f"{missing}"
            )

    return x, trajectory


# =============================================================================
# NAIVE STRIDED SAMPLING
# =============================================================================

@torch.no_grad()
def sample_naive_strided_ddpm(
    model,
    schedule,
    batch_size=4,
    stride=100,
):

    model.eval()

    model_device = get_model_device(
        model
    )

    x = torch.randn(
        batch_size,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=model_device,
    )

    timesteps = list(
        range(
            T,
            0,
            -stride,
        )
    )

    if timesteps[-1] != 1:

        timesteps.append(1)

    start = time.perf_counter()

    for timestep in timesteps:

        x = reverse_step(
            model,
            x,
            timestep,
            schedule,
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        x,
        elapsed,
        timesteps,
    )


# =============================================================================
# TEST 1
# =============================================================================

def test_schedule(schedule):

    print("\n" + "=" * 80)
    print("TEST 1 — DIFFUSION SCHEDULE")
    print("=" * 80)

    assert schedule["beta"].shape == (
        T,
    )

    assert schedule["alpha"].shape == (
        T,
    )

    assert schedule["alpha_bar"].shape == (
        T,
    )

    assert torch.all(
        schedule["beta"] > 0
    )

    assert torch.all(
        schedule["beta"] < 1
    )

    assert torch.all(
        schedule["alpha"] > 0
    )

    assert torch.all(
        schedule["alpha"] < 1
    )

    assert torch.all(
        schedule["alpha_bar"] > 0
    )

    assert torch.all(
        schedule["alpha_bar"] <= 1
    )

    print(
        "Schedule device:",
        schedule["beta"].device,
    )

    print(
        "alpha_bar[0]:",
        float(
            schedule["alpha_bar"][0]
        ),
    )

    print(
        "alpha_bar[-1]:",
        float(
            schedule["alpha_bar"][-1]
        ),
    )

    print("TEST 1 PASSED")


# =============================================================================
# TEST 2
# =============================================================================

def test_forward_diffusion(schedule):

    print("\n" + "=" * 80)
    print("TEST 2 — FORWARD DIFFUSION")
    print("=" * 80)

    x0 = torch.ones(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=schedule["beta"].device,
    )

    t = torch.tensor(
        [0, 100, 500, 999],
        device=x0.device,
        dtype=torch.long,
    )

    xt, noise = q_sample(
        x0,
        t,
        schedule,
    )

    assert xt.shape == x0.shape
    assert noise.shape == x0.shape

    assert torch.isfinite(
        xt
    ).all()

    print(
        "x0 device:",
        x0.device,
    )

    print(
        "xt device:",
        xt.device,
    )

    print("TEST 2 PASSED")


# =============================================================================
# TEST 3
# =============================================================================

def test_unet_output(model):

    print("\n" + "=" * 80)
    print("TEST 3 — U-NET OUTPUT")
    print("=" * 80)

    model.eval()

    model_device = get_model_device(
        model
    )

    x = torch.randn(
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=model_device,
    )

    t = torch.randint(
        0,
        T,
        (4,),
        device=model_device,
    )

    with torch.no_grad():

        output = model(
            x,
            t,
        )

    assert output.shape == x.shape

    assert output.device == model_device

    assert torch.isfinite(
        output
    ).all()

    print(
        "Model device:",
        model_device,
    )

    print(
        "Output device:",
        output.device,
    )

    print("TEST 3 PASSED")


# =============================================================================
# TEST 4
# =============================================================================

def test_gradient_flow(model):

    print("\n" + "=" * 80)
    print("TEST 4 — GRADIENT FLOW")
    print("=" * 80)

    model.train()

    model_device = get_model_device(
        model
    )

    x = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=model_device,
        requires_grad=True,
    )

    t = torch.randint(
        0,
        T,
        (2,),
        device=model_device,
    )

    output = model(
        x,
        t,
    )

    loss = output.square().mean()

    model.zero_grad(
        set_to_none=True
    )

    loss.backward()

    gradient_count = 0
    maximum_gradient = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_count += 1

            maximum_gradient = max(
                maximum_gradient,
                float(
                    parameter.grad.detach()
                    .abs()
                    .max()
                    .cpu()
                ),
            )

    assert gradient_count > 0
    assert maximum_gradient > 0

    print(
        "Gradient parameters:",
        gradient_count,
    )

    print(
        "Maximum gradient:",
        maximum_gradient,
    )

    print("TEST 4 PASSED")


# =============================================================================
# TEST 5
# =============================================================================

def test_single_reverse_step(
    model,
    schedule,
):

    print("\n" + "=" * 80)
    print("TEST 5 — SINGLE REVERSE STEP")
    print("=" * 80)

    model.eval()

    model_device = get_model_device(
        model
    )

    x = torch.randn(
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=model_device,
    )

    previous = reverse_step(
        model,
        x,
        500,
        schedule,
    )

    assert previous.shape == x.shape

    assert previous.device == model_device

    assert torch.isfinite(
        previous
    ).all()

    print(
        "Reverse output device:",
        previous.device,
    )

    print("TEST 5 PASSED")


# =============================================================================
# TEST 6
# =============================================================================

def test_sampling_stochasticity(
    model,
    schedule,
):

    print("\n" + "=" * 80)
    print("TEST 6 — SAMPLING STOCHASTICITY")
    print("=" * 80)

    model.eval()

    model_device = get_model_device(
        model
    )

    x = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=model_device,
    )

    torch.manual_seed(123)

    a = x.clone()

    for timestep in range(
        10,
        0,
        -1,
    ):

        a = reverse_step(
            model,
            a,
            timestep,
            schedule,
        )

    torch.manual_seed(456)

    b = x.clone()

    for timestep in range(
        10,
        0,
        -1,
    ):

        b = reverse_step(
            model,
            b,
            timestep,
            schedule,
        )

    difference = (
        a - b
    ).abs().mean()

    assert float(difference) > 1e-8

    print(
        "Mean stochastic difference:",
        float(difference),
    )

    print("TEST 6 PASSED")


# =============================================================================
# TEST 7
# =============================================================================

def test_final_step_deterministic(
    model,
    schedule,
):

    print("\n" + "=" * 80)
    print("TEST 7 — FINAL DDPM STEP")
    print("=" * 80)

    model.eval()

    model_device = get_model_device(
        model
    )

    x = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=model_device,
    )

    torch.manual_seed(111)

    a = reverse_step(
        model,
        x,
        1,
        schedule,
    )

    torch.manual_seed(999)

    b = reverse_step(
        model,
        x,
        1,
        schedule,
    )

    difference = (
        a - b
    ).abs().max()

    print(
        "Maximum difference:",
        float(difference),
    )

    print(
        "Tolerance:",
        FINAL_STEP_ATOL,
    )

    assert float(difference) < FINAL_STEP_ATOL

    print("TEST 7 PASSED")


# =============================================================================
# TEST 8
# =============================================================================

def test_x0_reconstruction(
    model,
    schedule,
):

    print("\n" + "=" * 80)
    print("TEST 8 — X0 RECONSTRUCTION")
    print("=" * 80)

    model.eval()

    model_device = get_model_device(
        model
    )

    x0 = create_structured_images(
        4,
        IMAGE_SIZE,
        model_device,
    )

    timestep_index = 499

    t = torch.full(
        (4,),
        timestep_index,
        device=model_device,
        dtype=torch.long,
    )

    xt, _ = q_sample(
        x0,
        t,
        schedule,
    )

    with torch.no_grad():

        predicted_noise = model(
            xt,
            t,
        )

        reconstructed = predict_x0_from_xt(
            xt,
            predicted_noise,
            timestep_index,
            schedule,
        )

    assert reconstructed.shape == x0.shape

    assert reconstructed.device == model_device

    assert torch.isfinite(
        reconstructed
    ).all()

    error = (
        reconstructed - x0
    ).abs().mean()

    print(
        "Mean reconstruction error:",
        float(error),
    )

    assert float(error) < 1.0

    print("TEST 8 PASSED")


# =============================================================================
# TEST 9 — COMPLETE SAMPLER
# =============================================================================

def test_complete_sampler(
    model,
    schedule,
):

    print("\n" + "=" * 80)
    print("TEST 9 — COMPLETE DDPM SAMPLER")
    print("=" * 80)

    generated, trajectory = sample_ddpm(
        model,
        schedule,
        batch_size=4,
        return_trajectory=True,
    )

    model_device = get_model_device(
        model
    )

    assert generated.device == model_device

    assert generated.shape == (
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert torch.isfinite(
        generated
    ).all()

    required_keys = set(
        TRAJECTORY_CHECKPOINTS
    )

    actual_keys = set(
        trajectory.keys()
    )

    print(
        "Required keys:",
        sorted(
            required_keys,
            reverse=True,
        ),
    )

    print(
        "Actual keys:",
        sorted(
            actual_keys,
            reverse=True,
        ),
    )

    assert required_keys.issubset(
        actual_keys
    )

    for key in TRAJECTORY_CHECKPOINTS:

        assert trajectory[key] is not None

        assert trajectory[key].device == model_device

        assert trajectory[key].shape == (
            4,
            CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        assert torch.isfinite(
            trajectory[key]
        ).all()

    final_difference = (
        generated
        -
        trajectory[0]
    ).abs().max()

    assert (
        float(final_difference)
        <
        FINAL_STEP_ATOL
    )

    print(
        "Generated device:",
        generated.device,
    )

    print(
        "Generated shape:",
        tuple(generated.shape),
    )

    print(
        "Generated mean:",
        float(generated.mean()),
    )

    print(
        "Generated std:",
        float(generated.std()),
    )

    print(
        "Final trajectory difference:",
        float(final_difference),
    )

    print("TEST 9 PASSED")


# =============================================================================
# TEST 10
# =============================================================================

def test_sampling_trajectory(
    trajectory,
):

    print("\n" + "=" * 80)
    print("TEST 10 — SAMPLING TRAJECTORY")
    print("=" * 80)

    for key in TRAJECTORY_CHECKPOINTS:

        assert key in trajectory

        image = trajectory[key]

        assert image is not None
        assert image.ndim == 4
        assert image.shape[1] == CHANNELS
        assert image.shape[2] == IMAGE_SIZE
        assert image.shape[3] == IMAGE_SIZE

    print(
        "Trajectory keys:",
        sorted(
            trajectory.keys(),
            reverse=True,
        ),
    )

    print(
        "Initial std:",
        float(
            trajectory[T].std()
        ),
    )

    print(
        "Final std:",
        float(
            trajectory[0].std()
        ),
    )

    print("TEST 10 PASSED")


# =============================================================================
# TEST 11 — ROBUST DEVICE CONSISTENCY
# =============================================================================

def test_device_consistency(
    model,
    schedule,
):

    print("\n" + "=" * 80)
    print("TEST 11 — DEVICE CONSISTENCY")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # IMPORTANT FIX
    #
    # Do not use:
    #
    #     assert model_device == DEVICE
    #
    # as the only test.
    #
    # The actual module device is the authoritative runtime state.
    # -------------------------------------------------------------------------

    model_device = get_model_device(
        model
    )

    schedule_devices = {
        value.device
        for value in schedule.values()
    }

    print(
        "Global DEVICE:",
        DEVICE,
    )

    print(
        "Actual model device:",
        model_device,
    )

    print(
        "Schedule devices:",
        schedule_devices,
    )

    # The model must have a real device.
    assert model_device is not None

    # Every schedule tensor must be on the actual model device.
    assert schedule_devices == {
        model_device
    }

    # Create a real tensor on the actual runtime device.
    test_tensor = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=model_device,
    )

    assert test_tensor.device == model_device

    # Run the model using a tensor on its actual device.
    model.eval()

    with torch.no_grad():

        test_timesteps = torch.zeros(
            2,
            device=model_device,
            dtype=torch.long,
        )

        output = model(
            test_tensor,
            test_timesteps,
        )

    assert output.device == model_device

    # Verify schedule values can interact with the model tensor.
    beta = schedule["beta"][0].to(
        device=model_device,
        dtype=test_tensor.dtype,
    )

    combined = (
        test_tensor
        *
        beta
    )

    assert combined.device == model_device

    print(
        "Test tensor device:",
        test_tensor.device,
    )

    print(
        "Model output device:",
        output.device,
    )

    print(
        "Combined tensor device:",
        combined.device,
    )

    print(
        "Runtime device relationship: CONSISTENT"
    )

    print("TEST 11 PASSED")


# =============================================================================
# VISUALIZATION — LOSS
# =============================================================================

def plot_training_loss(losses):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        losses,
        linewidth=1,
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "Diffusion Noise Prediction Training"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# VISUALIZATION — FORWARD
# =============================================================================

@torch.no_grad()
def visualize_forward_diffusion(
    schedule,
):

    model_device = schedule[
        "beta"
    ].device

    x0 = create_structured_images(
        1,
        IMAGE_SIZE,
        model_device,
    )

    timestep_indices = [
        0,
        99,
        249,
        499,
        749,
        999,
    ]

    plt.figure(
        figsize=(15, 3)
    )

    for index, timestep_index in enumerate(
        timestep_indices
    ):

        t = torch.tensor(
            [timestep_index],
            device=model_device,
            dtype=torch.long,
        )

        xt, _ = q_sample(
            x0,
            t,
            schedule,
        )

        image = (
            xt[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        plt.subplot(
            1,
            len(timestep_indices),
            index + 1,
        )

        plt.imshow(
            image,
            cmap="gray",
            vmin=-2,
            vmax=2,
        )

        plt.title(
            f"t={timestep_index + 1}"
        )

        plt.axis("off")

    plt.suptitle(
        "Forward Diffusion"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# VISUALIZATION — REVERSE TRAJECTORY
# =============================================================================

def visualize_reverse_trajectory(
    trajectory,
):

    keys = list(
        TRAJECTORY_CHECKPOINTS
    )

    plt.figure(
        figsize=(20, 4)
    )

    for index, timestep in enumerate(
        keys
    ):

        image = (
            trajectory[timestep][0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        plt.subplot(
            2,
            5,
            index + 1,
        )

        if timestep == 0:

            plt.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

        else:

            plt.imshow(
                image,
                cmap="gray",
                vmin=-2,
                vmax=2,
            )

        plt.title(
            f"t={timestep}"
        )

        plt.axis("off")

    plt.suptitle(
        "DDPM Reverse Sampling Trajectory"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# VISUALIZATION — GENERATED SAMPLES
# =============================================================================

def visualize_generated_samples(
    generated,
    title="Generated Samples",
):

    images = (
        generated
        .detach()
        .cpu()
        .clamp(
            0.0,
            1.0,
        )
    )

    count = images.shape[0]

    columns = 4

    rows = math.ceil(
        count / columns
    )

    plt.figure(
        figsize=(
            columns * 3,
            rows * 3,
        )
    )

    for index in range(count):

        plt.subplot(
            rows,
            columns,
            index + 1,
        )

        plt.imshow(
            images[index, 0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        plt.title(
            f"Sample {index + 1}"
        )

        plt.axis("off")

    plt.suptitle(
        title
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# VISUALIZATION — SPEED
# =============================================================================

def visualize_sampling_cost(
    full_time,
    naive_time,
    full_steps,
    naive_steps,
):

    labels = [
        "Full DDPM",
        "Naive Strided",
    ]

    times = [
        full_time,
        naive_time,
    ]

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        labels,
        times,
    )

    plt.ylabel(
        "Sampling time (seconds)"
    )

    plt.title(
        "Sampling Cost Comparison"
    )

    for bar, value in zip(
        bars,
        times,
    ):

        plt.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.show()

    print(
        "Full DDPM steps:",
        full_steps,
    )

    print(
        "Naive strided steps:",
        naive_steps,
    )


# =============================================================================
# SPEED EXPERIMENT
# =============================================================================

def benchmark_sampling(
    model,
    schedule,
):

    print("\n" + "=" * 80)
    print("SAMPLING SPEED EXPERIMENT")
    print("=" * 80)

    model_device = get_model_device(
        model
    )

    if model_device.type == "cuda":

        torch.cuda.synchronize(
            model_device
        )

    start = time.perf_counter()

    generated, trajectory = sample_ddpm(
        model,
        schedule,
        batch_size=SAMPLE_COUNT,
        return_trajectory=True,
    )

    if model_device.type == "cuda":

        torch.cuda.synchronize(
            model_device
        )

    full_time = (
        time.perf_counter()
        - start
    )

    naive_generated, naive_time, naive_timesteps = (
        sample_naive_strided_ddpm(
            model,
            schedule,
            batch_size=SAMPLE_COUNT,
            stride=100,
        )
    )

    print(
        f"Full DDPM time: {full_time:.4f} sec"
    )

    print(
        f"Naive strided time: {naive_time:.4f} sec"
    )

    print(
        f"Full DDPM evaluations: {T}"
    )

    print(
        f"Naive evaluations: "
        f"{len(naive_timesteps)}"
    )

    return (
        generated,
        trajectory,
        naive_generated,
        full_time,
        naive_time,
        len(naive_timesteps),
    )


# =============================================================================
# REAL-WORLD APPLICATION
# =============================================================================

def real_world_robot_application():

    print("\n" + "=" * 80)
    print("REAL-WORLD ROBOTICS CONNECTION")
    print("=" * 80)

    print(
        """
A robot vision system may operate at approximately 10 Hz.

That means one complete perception/action cycle is approximately:

    100 ms

A standard DDPM with:

    1000 reverse steps

may require far more computation than this budget.

For example, if one U-Net evaluation takes:

    5 ms

then:

    1000 x 5 ms
    =
    5000 ms
    =
    5 seconds

That is far too slow for real-time manipulation.

Therefore practical diffusion systems require accelerated sampling.

Possible solutions include:

    DDIM
    latent diffusion
    distillation
    consistency models
    optimized samplers

Lesson 22 introduces DDIM.
"""
    )


# =============================================================================
# FINAL RESULT
# =============================================================================

def print_final_result(
    losses,
    generated,
    full_time,
):

    initial_loss = np.mean(
        losses[:100]
    )

    final_loss = np.mean(
        losses[-100:]
    )

    print("\n" + "=" * 80)
    print("LESSON 21 RESULT")
    print("=" * 80)

    print(
        f"Initial 100-step mean loss: "
        f"{initial_loss:.6f}"
    )

    print(
        f"Final 100-step mean loss: "
        f"{final_loss:.6f}"
    )

    print(
        "Generated shape:",
        tuple(generated.shape),
    )

    print(
        "Generated mean:",
        float(
            generated.mean()
        ),
    )

    print(
        "Generated std:",
        float(
            generated.std()
        ),
    )

    print(
        f"1000-step sampling time: "
        f"{full_time:.4f} sec"
    )

    print(
        """
The complete diffusion generation pipeline is now implemented:

    Gaussian noise
          |
          v
    DDPM reverse step
          |
          v
       x_(T-1)
          |
         ...
          |
          v
        x_1
          |
          v
        x_0

The model learned to predict noise, and the sampler uses that prediction
to progressively remove noise.

The remaining practical limitation is sampling speed.

That leads directly to DDIM.
"""
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("LESSON 21 — DIFFUSION SAMPLING")
    print("=" * 80)

    print(
        "PyTorch version:",
        torch.__version__,
    )

    print(
        "Configured DEVICE:",
        DEVICE,
    )

    print(
        "T:",
        T,
    )

    print(
        "Image size:",
        IMAGE_SIZE,
        "x",
        IMAGE_SIZE,
    )

    print(
        "Training steps:",
        TRAIN_STEPS,
    )

    set_seed()

    # -------------------------------------------------------------------------
    # SCHEDULE
    # -------------------------------------------------------------------------

    schedule = make_schedule(
        device=DEVICE
    )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    model = DiffusionUNet(
        in_channels=CHANNELS,
        base_channels=BASE_CHANNELS,
        time_dim=TIME_EMBED_DIM,
    )

    initialize_model(model)

    # Explicitly place model on runtime device.
    model = model.to(
        DEVICE
    )

    # -------------------------------------------------------------------------
    # SYNCHRONIZE MODEL AND SCHEDULE
    # -------------------------------------------------------------------------

    model, schedule = (
        synchronize_model_and_schedule(
            model,
            schedule,
        )
    )

    actual_model_device = get_model_device(
        model
    )

    print(
        "Actual model device:",
        actual_model_device,
    )

    print(
        "Actual schedule device:",
        schedule["beta"].device,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    # -------------------------------------------------------------------------
    # TESTS BEFORE TRAINING
    # -------------------------------------------------------------------------

    test_schedule(
        schedule
    )

    test_forward_diffusion(
        schedule
    )

    test_unet_output(
        model
    )

    test_gradient_flow(
        model
    )

    test_single_reverse_step(
        model,
        schedule,
    )

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING DIFFUSION U-NET")
    print("=" * 80)

    losses = train_diffusion_model(
        model,
        schedule,
        steps=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    initial_loss = np.mean(
        losses[:100]
    )

    final_loss = np.mean(
        losses[-100:]
    )

    print(
        f"Initial 100-step mean: "
        f"{initial_loss:.6f}"
    )

    print(
        f"Final 100-step mean: "
        f"{final_loss:.6f}"
    )

    assert final_loss < initial_loss

    print(
        "Training improvement PASSED"
    )

    # -------------------------------------------------------------------------
    # LEARNED NOISE PREDICTION
    # -------------------------------------------------------------------------

    model.eval()

    model_device = get_model_device(
        model
    )

    test_x0 = create_structured_images(
        4,
        IMAGE_SIZE,
        model_device,
    )

    test_t = torch.randint(
        0,
        T,
        (4,),
        device=model_device,
    )

    test_xt, true_noise = q_sample(
        test_x0,
        test_t,
        schedule,
    )

    with torch.no_grad():

        predicted_noise = model(
            test_xt,
            test_t,
        )

    noise_prediction_mse = F.mse_loss(
        predicted_noise,
        true_noise,
    )

    print("\n" + "=" * 80)
    print("LEARNED NOISE PREDICTION")
    print("=" * 80)

    print(
        "True noise std:",
        float(
            true_noise.std()
        ),
    )

    print(
        "Predicted noise std:",
        float(
            predicted_noise.std()
        ),
    )

    print(
        "Prediction MSE:",
        float(
            noise_prediction_mse
        ),
    )

    assert torch.isfinite(
        predicted_noise
    ).all()

    # -------------------------------------------------------------------------
    # TESTS AFTER TRAINING
    # -------------------------------------------------------------------------

    test_sampling_stochasticity(
        model,
        schedule,
    )

    test_final_step_deterministic(
        model,
        schedule,
    )

    test_x0_reconstruction(
        model,
        schedule,
    )

    # -------------------------------------------------------------------------
    # COMPLETE SAMPLER
    # -------------------------------------------------------------------------

    generated, trajectory = sample_ddpm(
        model,
        schedule,
        batch_size=4,
        return_trajectory=True,
    )

    test_complete_sampler(
        model,
        schedule,
    )

    test_sampling_trajectory(
        trajectory
    )

    # -------------------------------------------------------------------------
    # DEVICE TEST
    # -------------------------------------------------------------------------

    test_device_consistency(
        model,
        schedule,
    )

    # -------------------------------------------------------------------------
    # SPEED
    # -------------------------------------------------------------------------

    (
        generated,
        trajectory,
        naive_generated,
        full_time,
        naive_time,
        naive_steps,
    ) = benchmark_sampling(
        model,
        schedule,
    )

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------

    plot_training_loss(
        losses
    )

    visualize_forward_diffusion(
        schedule
    )

    visualize_reverse_trajectory(
        trajectory
    )

    visualize_generated_samples(
        generated,
        "DDPM Generated Samples",
    )

    visualize_generated_samples(
        naive_generated,
        "Naive Strided DDPM Experiment",
    )

    visualize_sampling_cost(
        full_time,
        naive_time,
        T,
        naive_steps,
    )

    # -------------------------------------------------------------------------
    # REAL-WORLD RESULT
    # -------------------------------------------------------------------------

    real_world_robot_application()

    # -------------------------------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------------------------------

    print_final_result(
        losses,
        generated,
        full_time,
    )

    print("\n" + "=" * 80)
    print("NEXT LESSON")
    print("=" * 80)

    print(
        """
LESSON 22 — DDIM

We now have:

    Forward diffusion
    Noise schedules
    Noise prediction
    Diffusion U-Net
    DDPM reverse step
    Complete DDPM sampling

The remaining major problem is:

    SAMPLING SPEED

DDPM:

    x_T -> x_(T-1) -> ... -> x_1 -> x_0

typically requires many evaluations.

DDIM will introduce a different sampling formulation that can use
far fewer timesteps.

Next we will study:

    - non-Markovian sampling
    - deterministic diffusion sampling
    - timestep sub-sampling
    - DDPM vs DDIM
    - stochastic vs deterministic generation
    - speed/quality trade-offs

This is the next major step toward practical generative vision systems.
"""
    )


if __name__ == "__main__":
    main()