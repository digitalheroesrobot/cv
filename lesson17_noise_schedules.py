"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 17 — NOISE SCHEDULES
================================================================================

Filename:
    lesson17_noise_schedules.py

================================================================================
PURPOSE
================================================================================

Lesson 16 introduced the forward diffusion process:

    x_t = sqrt(alpha_bar_t) * x_0
          +
          sqrt(1 - alpha_bar_t) * epsilon

Lesson 17 studies the NOISE SCHEDULE.

A diffusion model does not simply add random noise.

It defines a controlled trajectory:

    clean image
        |
        v
    slightly noisy image
        |
        v
    moderately noisy image
        |
        v
    heavily noisy image
        |
        v
    approximately Gaussian noise

The mathematical schedule determines how quickly useful image
information disappears.

This lesson compares:

    1. Linear beta schedule
    2. Cosine alpha_bar schedule

and studies:

    beta_t
    alpha_t
    alpha_bar_t
    signal power
    noise power
    SNR
    log-SNR
    numerical stability


================================================================================
WHAT YOU WILL LEARN
================================================================================

You will learn:

    - What beta_t represents.
    - What alpha_t represents.
    - What alpha_bar_t represents.
    - Why alpha_bar_t is more important than beta_t alone.
    - How linear beta schedules behave.
    - How cosine schedules behave.
    - How SNR changes during diffusion.
    - Why numerical clipping is required.
    - How to verify schedule mathematics numerically.
    - How schedule choice affects corrupted images.


================================================================================
TECHNICAL / COURSE ROADMAP
================================================================================

Completed:

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
    Lesson 16  Forward diffusion process

Current:

    Lesson 17  Noise schedules

Next:

    Lesson 18  Learning to Predict Noise
    Lesson 19  DDPM
    Lesson 20  Diffusion U-Net
    Lesson 21  Diffusion Sampling
    Lesson 22  DDIM
    Lesson 23  Latent Diffusion
    Lesson 24  Conditioning
    Lesson 25  Cross-Attention
    Lesson 26  Text Conditioning
    Lesson 27  Own Text-to-Image Model


================================================================================
REAL-WORLD PROBLEM
================================================================================

Suppose a robot camera sees:

    a bottle
    a can
    cardboard
    an irregular waste object

The diffusion training process intentionally corrupts the image.

The neural network later needs to learn:

    noisy image + timestep
        |
        v
    estimate noise
        |
        v
    recover useful visual structure

If noise is introduced too aggressively:

    useful information disappears too quickly.

If noise is introduced too slowly:

    some training timesteps may contain almost identical information.

Therefore:

    THE NOISE SCHEDULE DEFINES THE DIFFICULTY CURVE
    OF THE DENOISING PROBLEM.


================================================================================
THEORY
================================================================================

The forward diffusion process is:

    q(x_t | x_{t-1})
        =
    N(
        sqrt(alpha_t) x_{t-1},
        beta_t I
    )

where:

    beta_t  = noise variance
    alpha_t = 1 - beta_t

The cumulative product is:

    alpha_bar_t
        =
    alpha_1 alpha_2 ... alpha_t

Therefore:

    x_t
        =
    sqrt(alpha_bar_t) x_0
        +
    sqrt(1-alpha_bar_t) epsilon

where:

    epsilon ~ N(0, I)


================================================================================
MATHEMATICS — SIGNAL AND NOISE
================================================================================

The coefficient multiplying x_0 is:

    sqrt(alpha_bar_t)

The coefficient multiplying epsilon is:

    sqrt(1-alpha_bar_t)

Therefore:

    signal power = alpha_bar_t

    noise power = 1-alpha_bar_t

and:

    signal power + noise power = 1


================================================================================
MATHEMATICS — SNR
================================================================================

Signal-to-noise ratio:

    SNR_t
        =
    alpha_bar_t
    /
    (1-alpha_bar_t)

At the beginning:

    alpha_bar_t ≈ 1

therefore:

    SNR >> 1

The image contains strong signal.

At the end:

    alpha_bar_t ≈ 0

therefore:

    SNR << 1

The image is noise dominated.


================================================================================
MATHEMATICS — LINEAR BETA
================================================================================

A simple schedule is:

    beta_t =
        beta_start
        +
        t/(T-1)
        *
        (beta_end-beta_start)

Example:

    beta_start = 0.0001
    beta_end   = 0.02

However:

    LINEAR beta
        does not mean
    LINEAR alpha_bar.


================================================================================
MATHEMATICS — COSINE SCHEDULE
================================================================================

The cosine schedule directly defines alpha_bar.

Define:

    f(t)
        =
    cos^2(
        ((t/T)+s)/(1+s)
        *
        pi/2
    )

Then:

    alpha_bar_t
        =
    f(t)/f(0)

From:

    alpha_bar_t
        =
    alpha_bar_{t-1} alpha_t

we obtain:

    alpha_t
        =
    alpha_bar_t
    /
    alpha_bar_{t-1}

and therefore:

    beta_t
        =
    1-alpha_t


================================================================================
NUMERICAL STABILITY
================================================================================

The theoretical final beta can become extremely close to 1.

Therefore an implementation commonly uses:

    beta_safe
        =
    clamp(beta_raw, max=0.999)

This prevents:

    beta = 1

which would produce:

    alpha = 0

and can cause numerical problems.

IMPORTANT:

    raw_beta

and:

    safe_beta

are not always identical.

A correct test must distinguish them.


================================================================================
IMPLEMENTATION
================================================================================

This file implements:

    - linear beta schedule
    - cosine alpha_bar schedule
    - raw beta derivation
    - safe beta clipping
    - alpha
    - alpha_bar
    - SNR
    - closed-form diffusion
    - statistical tests
    - device consistency tests
    - visualizations
    - real-world schedule experiment


================================================================================
VERIFICATION / TESTS
================================================================================

The script verifies:

    TEST 1
        Schedule structure.

    TEST 2
        alpha = 1-beta.

    TEST 3
        alpha_bar decreases.

    TEST 4
        signal + noise = 1.

    TEST 5
        SNR decreases.

    TEST 6
        Cosine raw beta derivation.

    TEST 7
        Safe beta clipping.

    TEST 8
        Sampling statistics.

    TEST 9
        Final state is approximately noise.

    TEST 10
        Diffusion tensors remain device-consistent.

The device test intentionally avoids brittle direct comparison
between device objects.

Instead it checks:

    - device type
    - CUDA index
    - tensor-to-tensor device consistency


================================================================================
VISUALIZATION
================================================================================

The script visualizes:

    - beta schedules
    - alpha_bar
    - signal/noise power
    - SNR
    - corrupted images
    - pixel trajectories
    - corrupted-value distributions


================================================================================
EXPERIMENT
================================================================================

The linear and cosine schedules are compared at:

    t = 1
    t = 100
    t = 250
    t = 500
    t = 750
    t = 900
    t = 999
    t = 1000


================================================================================
REAL-WORLD APPLICATION
================================================================================

For a robot vision system, schedule design determines how the
visual information is gradually destroyed.

A good training trajectory gives the future denoising network
examples covering:

    almost clean
    lightly corrupted
    moderately corrupted
    heavily corrupted
    almost pure noise

This is directly relevant to future image-generation and
vision-generation systems.


================================================================================
REAL-WORLD RESULT
================================================================================

The result of this lesson is a quantitative description of the
camera-image corruption trajectory.

We can now answer:

    How much signal remains at timestep t?

    How much noise has been added?

    What is the SNR?

    How does the answer change between schedules?


================================================================================
LESSON RESULT
================================================================================

The central relationship is:

    beta_t
       |
       v
    alpha_t = 1-beta_t
       |
       v
    alpha_bar_t
       |
       +--------------------+
       |                    |
       v                    v
    signal power         noise power
       |                    |
       +---------+----------+
                 |
                 v
                SNR

The schedule therefore defines the time structure of diffusion.


================================================================================
CONNECTION TO NEXT LESSON
================================================================================

Lesson 18 will finally introduce a trainable neural network.

The network will receive:

    x_t
    t

and learn:

    epsilon_theta(x_t,t)

which estimates the noise epsilon that produced x_t.

Therefore:

    Lesson 16
        How to add noise.

    Lesson 17
        How to control the noise trajectory.

    Lesson 18
        How to learn the noise.


================================================================================
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

SEED = 42

T = 1000

BETA_START = 1e-4
BETA_END = 0.02

COSINE_S = 0.008

IMAGE_SIZE = 64

MAX_BETA = 0.999

EPS = 1e-12

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DEVICE = torch.device("cpu")

TEST_DTYPE = torch.float64


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int = SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# SYNTHETIC STRUCTURED IMAGE
# ==============================================================================

def create_structured_image(
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    y = torch.linspace(
        0.0,
        1.0,
        height,
        device=device,
        dtype=dtype,
    )

    x = torch.linspace(
        0.0,
        1.0,
        width,
        device=device,
        dtype=dtype,
    )

    yy, xx = torch.meshgrid(
        y,
        x,
        indexing="ij",
    )

    image = (
        0.15
        +
        0.20 * xx
        +
        0.10 * yy
    )

    rectangle = (
        (xx > 0.12)
        &
        (xx < 0.40)
        &
        (yy > 0.20)
        &
        (yy < 0.48)
    )

    image = (
        image
        +
        rectangle.to(dtype)
        * 0.45
    )

    circle = (
        (xx - 0.68) ** 2
        +
        (yy - 0.35) ** 2
        <
        0.12 ** 2
    )

    image = (
        image
        +
        circle.to(dtype)
        * 0.55
    )

    diagonal = (
        torch.abs(
            yy
            -
            (
                0.90
                -
                0.75 * xx
            )
        )
        < 0.035
    )

    image = (
        image
        +
        diagonal.to(dtype)
        * 0.30
    )

    texture = (
        0.035
        *
        torch.sin(
            2.0
            * math.pi
            * 8.0
            * xx
        )
        *
        torch.sin(
            2.0
            * math.pi
            * 7.0
            * yy
        )
    )

    image = image + texture

    image = image.clamp(
        0.0,
        1.0,
    )

    return image.unsqueeze(0)


# ==============================================================================
# LINEAR BETA SCHEDULE
# ==============================================================================

def make_linear_beta_schedule(
    timesteps: int = T,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    if timesteps < 2:
        raise ValueError(
            "timesteps must be >= 2"
        )

    if not (
        0.0 < beta_start < 1.0
    ):
        raise ValueError(
            "beta_start must be in (0,1)"
        )

    if not (
        0.0 < beta_end < 1.0
    ):
        raise ValueError(
            "beta_end must be in (0,1)"
        )

    if beta_start >= beta_end:
        raise ValueError(
            "beta_start must be < beta_end"
        )

    return torch.linspace(
        beta_start,
        beta_end,
        timesteps,
        device=device,
        dtype=dtype,
    )


# ==============================================================================
# COSINE ALPHA-BAR SCHEDULE
# ==============================================================================

def make_cosine_alpha_bar_schedule(
    timesteps: int = T,
    s: float = COSINE_S,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    if timesteps < 2:
        raise ValueError(
            "timesteps must be >= 2"
        )

    t = torch.linspace(
        0.0,
        1.0,
        timesteps + 1,
        device=device,
        dtype=dtype,
    )

    f = torch.cos(
        (
            (t + s)
            /
            (1.0 + s)
        )
        *
        (math.pi / 2.0)
    ) ** 2

    f = f / f[0]

    alpha_bar = f[1:]

    alpha_bar = alpha_bar.clamp(
        min=EPS,
        max=1.0 - EPS,
    )

    return alpha_bar


# ==============================================================================
# RAW BETA FROM ALPHA-BAR
# ==============================================================================

def raw_beta_from_alpha_bar(
    alpha_bar: torch.Tensor,
) -> torch.Tensor:

    alpha_bar_previous = torch.cat(
        [
            torch.ones(
                1,
                device=alpha_bar.device,
                dtype=alpha_bar.dtype,
            ),
            alpha_bar[:-1],
        ]
    )

    alpha = (
        alpha_bar
        /
        alpha_bar_previous
    )

    raw_beta = 1.0 - alpha

    return raw_beta


# ==============================================================================
# SAFE BETA
# ==============================================================================

def beta_from_alpha_bar(
    alpha_bar: torch.Tensor,
    max_beta: float = MAX_BETA,
) -> torch.Tensor:

    raw_beta = raw_beta_from_alpha_bar(
        alpha_bar
    )

    safe_beta = raw_beta.clamp(
        min=EPS,
        max=max_beta,
    )

    return safe_beta


# ==============================================================================
# COMPLETE SCHEDULE
# ==============================================================================

def prepare_schedule(
    beta: torch.Tensor,
) -> dict:

    alpha = 1.0 - beta

    alpha_bar = torch.cumprod(
        alpha,
        dim=0,
    )

    alpha_bar = alpha_bar.clamp(
        min=EPS,
        max=1.0,
    )

    sqrt_alpha_bar = torch.sqrt(
        alpha_bar
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        (
            1.0
            -
            alpha_bar
        ).clamp(min=EPS)
    )

    snr = (
        alpha_bar
        /
        (
            1.0
            -
            alpha_bar
        ).clamp(min=EPS)
    )

    log_snr = torch.log(
        snr.clamp(min=EPS)
    )

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar":
            sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar":
            sqrt_one_minus_alpha_bar,
        "snr": snr,
        "log_snr": log_snr,
    }


# ==============================================================================
# TIMESTEP INDEX
# ==============================================================================

def timestep_index(
    t,
    timesteps: int = T,
) -> torch.Tensor:

    if isinstance(t, int):

        if t < 1 or t > timesteps:
            raise ValueError(
                f"t must be in [1,{timesteps}]"
            )

        return torch.tensor(
            t - 1,
            dtype=torch.long,
        )

    t = torch.as_tensor(
        t,
        dtype=torch.long,
    )

    if torch.any(t < 1):
        raise ValueError(
            "Timestep must be >= 1"
        )

    if torch.any(t > timesteps):
        raise ValueError(
            "Timestep exceeds schedule"
        )

    return t - 1


# ==============================================================================
# CLOSED-FORM DIFFUSION
# ==============================================================================

def sample_xt_from_x0(
    x0: torch.Tensor,
    t,
    schedule: dict,
    noise: torch.Tensor | None = None,
):

    if noise is None:

        noise = torch.randn_like(
            x0
        )

    else:

        noise = noise.to(
            device=x0.device,
            dtype=x0.dtype,
        )

    indices = timestep_index(
        t,
        schedule["beta"].shape[0],
    ).to(
        device=x0.device
    )

    sqrt_alpha_bar = (
        schedule[
            "sqrt_alpha_bar"
        ][indices]
        .to(
            device=x0.device,
            dtype=x0.dtype,
        )
    )

    sqrt_one_minus_alpha_bar = (
        schedule[
            "sqrt_one_minus_alpha_bar"
        ][indices]
        .to(
            device=x0.device,
            dtype=x0.dtype,
        )
    )

    while (
        sqrt_alpha_bar.ndim
        <
        x0.ndim
    ):

        sqrt_alpha_bar = (
            sqrt_alpha_bar.unsqueeze(-1)
        )

        sqrt_one_minus_alpha_bar = (
            sqrt_one_minus_alpha_bar.unsqueeze(-1)
        )

    xt = (
        sqrt_alpha_bar * x0
        +
        sqrt_one_minus_alpha_bar
        * noise
    )

    return xt, noise


# ==============================================================================
# SCHEDULE VALUE
# ==============================================================================

def schedule_value(
    schedule: dict,
    key: str,
    t: int,
    device=None,
    dtype=None,
):

    index = timestep_index(
        t,
        schedule["beta"].shape[0],
    )

    value = schedule[key][index]

    if (
        device is not None
        or dtype is not None
    ):

        value = value.to(
            device=(
                device
                if device is not None
                else value.device
            ),
            dtype=(
                dtype
                if dtype is not None
                else value.dtype
            ),
        )

    return value


# ==============================================================================
# TEST 1 — SCHEDULE STRUCTURE
# ==============================================================================

def verify_schedule_structure(
    schedule: dict,
    name: str,
):

    beta = schedule["beta"]

    alpha = schedule["alpha"]

    alpha_bar = schedule["alpha_bar"]

    snr = schedule["snr"]

    print()
    print("=" * 80)
    print(
        f"TEST — SCHEDULE STRUCTURE: {name}"
    )
    print("=" * 80)

    assert beta.ndim == 1
    assert alpha.ndim == 1
    assert alpha_bar.ndim == 1
    assert snr.ndim == 1

    assert len(beta) == T
    assert len(alpha) == T
    assert len(alpha_bar) == T
    assert len(snr) == T

    assert torch.all(
        beta > 0
    )

    assert torch.all(
        beta < 1
    )

    assert torch.all(
        alpha > 0
    )

    assert torch.all(
        alpha < 1
    )

    assert torch.all(
        alpha_bar[1:]
        <=
        alpha_bar[:-1]
    )

    assert torch.all(
        alpha_bar > 0
    )

    assert torch.all(
        snr[1:]
        <=
        snr[:-1]
    )

    reconstruction_error = torch.max(
        torch.abs(
            alpha
            -
            (1.0 - beta)
        )
    ).item()

    assert reconstruction_error < 1e-6

    print(
        f"Device          : {beta.device}"
    )

    print(
        f"Dtype           : {beta.dtype}"
    )

    print(
        f"beta[0]         : "
        f"{beta[0].item():.8f}"
    )

    print(
        f"beta[-1]        : "
        f"{beta[-1].item():.8f}"
    )

    print(
        f"alpha_bar[0]    : "
        f"{alpha_bar[0].item():.8f}"
    )

    print(
        f"alpha_bar[-1]   : "
        f"{alpha_bar[-1].item():.8e}"
    )

    print(
        f"SNR[0]          : "
        f"{snr[0].item():.8e}"
    )

    print(
        f"SNR[-1]         : "
        f"{snr[-1].item():.8e}"
    )

    print(
        f"Max alpha error : "
        f"{reconstruction_error:.3e}"
    )

    print(
        "[PASS] Schedule structure is valid."
    )


# ==============================================================================
# TEST 2 — SIGNAL + NOISE
# ==============================================================================

def verify_signal_noise_power(
    schedule: dict,
    name: str,
):

    alpha_bar = schedule["alpha_bar"]

    signal_power = alpha_bar

    noise_power = (
        1.0
        -
        alpha_bar
    )

    total = (
        signal_power
        +
        noise_power
    )

    max_error = torch.max(
        torch.abs(
            total - 1.0
        )
    ).item()

    print()
    print("=" * 80)
    print(
        f"TEST — SIGNAL + NOISE: {name}"
    )
    print("=" * 80)

    print(
        f"Maximum error : "
        f"{max_error:.3e}"
    )

    assert max_error < 1e-6

    print(
        "[PASS] Signal + noise power = 1."
    )


# ==============================================================================
# TEST 3 — COSINE DERIVATION
# ==============================================================================

def verify_cosine_derivation(
    cosine_alpha_bar: torch.Tensor,
    cosine_schedule: dict,
):

    raw_beta = raw_beta_from_alpha_bar(
        cosine_alpha_bar
    )

    expected_safe_beta = raw_beta.clamp(
        min=EPS,
        max=MAX_BETA,
    )

    actual_beta = cosine_schedule[
        "beta"
    ]

    safe_error = torch.max(
        torch.abs(
            actual_beta
            -
            expected_safe_beta
        )
    ).item()

    print()
    print("=" * 80)
    print(
        "TEST — COSINE BETA DERIVATION"
    )
    print("=" * 80)

    print(
        f"Maximum safe derivation error : "
        f"{safe_error:.3e}"
    )

    assert safe_error < 1e-6

    assert torch.all(
        actual_beta <= MAX_BETA
    )

    print(
        "[PASS] Cosine derivation is correct."
    )


# ==============================================================================
# TEST 4 — COSINE CLIPPING
# ==============================================================================

def verify_cosine_clipping_behavior(
    cosine_alpha_bar: torch.Tensor,
    cosine_schedule: dict,
):

    raw_beta = raw_beta_from_alpha_bar(
        cosine_alpha_bar
    )

    actual_beta = cosine_schedule[
        "beta"
    ]

    clipping_mask = (
        raw_beta > MAX_BETA
    )

    number_clipped = int(
        clipping_mask.sum().item()
    )

    print()
    print("=" * 80)
    print(
        "TEST — COSINE NUMERICAL CLIPPING"
    )
    print("=" * 80)

    print(
        f"Maximum allowed beta : "
        f"{MAX_BETA}"
    )

    print(
        f"Clipped timesteps    : "
        f"{number_clipped}"
    )

    if number_clipped > 0:

        indices = torch.nonzero(
            clipping_mask,
            as_tuple=False,
        ).flatten()

        last_index = indices[-1].item()

        print(
            f"Last clipped timestep : "
            f"{last_index + 1}"
        )

        print(
            f"Raw beta              : "
            f"{raw_beta[last_index].item():.8f}"
        )

        print(
            f"Safe beta             : "
            f"{actual_beta[last_index].item():.8f}"
        )

    assert torch.all(
        actual_beta <= MAX_BETA
    )

    print(
        "[PASS] Numerical clipping is safe."
    )


# ==============================================================================
# TEST 5 — SAMPLING STATISTICS
# ==============================================================================

def verify_sampling_statistics(
    schedule: dict,
    t: int,
    num_samples: int = 100000,
):

    schedule_cpu = {
        key:
            value.detach().to(
                device=TEST_DEVICE,
                dtype=TEST_DTYPE,
            )
        for key, value
        in schedule.items()
    }

    index = timestep_index(
        t,
        T,
    )

    alpha_bar = (
        schedule_cpu[
            "alpha_bar"
        ][index]
    )

    x0_value = torch.tensor(
        0.35,
        device=TEST_DEVICE,
        dtype=TEST_DTYPE,
    )

    expected_mean = (
        torch.sqrt(alpha_bar)
        *
        x0_value
    )

    expected_variance = (
        1.0
        -
        alpha_bar
    )

    noise = torch.randn(
        num_samples,
        device=TEST_DEVICE,
        dtype=TEST_DTYPE,
    )

    samples = (
        expected_mean
        +
        torch.sqrt(
            expected_variance
        )
        *
        noise
    )

    empirical_mean = (
        samples.mean()
    )

    empirical_variance = (
        samples.var(
            unbiased=False
        )
    )

    mean_error = torch.abs(
        empirical_mean
        -
        expected_mean
    ).item()

    variance_error = torch.abs(
        empirical_variance
        -
        expected_variance
    ).item()

    print()
    print("=" * 80)
    print(
        f"TEST — SAMPLING STATISTICS t={t}"
    )
    print("=" * 80)

    print(
        f"Expected mean      : "
        f"{expected_mean.item():.8f}"
    )

    print(
        f"Empirical mean     : "
        f"{empirical_mean.item():.8f}"
    )

    print(
        f"Mean error         : "
        f"{mean_error:.3e}"
    )

    print(
        f"Expected variance  : "
        f"{expected_variance.item():.8f}"
    )

    print(
        f"Empirical variance : "
        f"{empirical_variance.item():.8f}"
    )

    print(
        f"Variance error     : "
        f"{variance_error:.3e}"
    )

    assert mean_error < 0.02

    assert variance_error < 0.02

    print(
        "[PASS] Sampling statistics."
    )


# ==============================================================================
# IMAGE METRICS
# ==============================================================================

def compute_image_metrics(
    x0: torch.Tensor,
    xt: torch.Tensor,
    noise: torch.Tensor,
):

    x0_flat = x0.flatten()

    xt_flat = xt.flatten()

    x0_centered = (
        x0_flat
        -
        x0_flat.mean()
    )

    xt_centered = (
        xt_flat
        -
        xt_flat.mean()
    )

    correlation = (
        torch.sum(
            x0_centered
            *
            xt_centered
        )
        /
        (
            torch.sqrt(
                torch.sum(
                    x0_centered ** 2
                )
                *
                torch.sum(
                    xt_centered ** 2
                )
            )
            +
            EPS
        )
    )

    mse = F.mse_loss(
        xt,
        x0,
    )

    return {
        "correlation":
            correlation.item(),

        "mse":
            mse.item(),

        "noise_std":
            noise.std().item(),
    }


# ==============================================================================
# GENERATE SCHEDULE COMPARISON
# ==============================================================================

def generate_schedule_comparison(
    x0: torch.Tensor,
    linear_schedule: dict,
    cosine_schedule: dict,
    timesteps_to_show: list[int],
):

    results = {
        "linear": {},
        "cosine": {},
    }

    for name, schedule in [
        ("linear", linear_schedule),
        ("cosine", cosine_schedule),
    ]:

        for t in timesteps_to_show:

            noise = torch.randn_like(
                x0
            )

            xt, _ = sample_xt_from_x0(
                x0,
                t,
                schedule,
                noise=noise,
            )

            results[name][t] = {
                "image":
                    xt.detach().cpu(),

                "noise":
                    noise.detach().cpu(),

                "metrics":
                    compute_image_metrics(
                        x0,
                        xt,
                        noise,
                    ),
            }

    return results


# ==============================================================================
# PRINT TABLE
# ==============================================================================

def print_schedule_table(
    linear_schedule: dict,
    cosine_schedule: dict,
    timesteps_to_show: list[int],
):

    print()
    print("=" * 110)
    print(
        "SCHEDULE COMPARISON"
    )
    print("=" * 110)

    print(
        f"{'t':>6} | "
        f"{'Linear beta':>12} | "
        f"{'Linear a_bar':>14} | "
        f"{'Linear SNR':>14} | "
        f"{'Cos beta':>12} | "
        f"{'Cos a_bar':>14} | "
        f"{'Cos SNR':>14}"
    )

    print("-" * 110)

    for t in timesteps_to_show:

        i = t - 1

        values = [
            linear_schedule[
                "beta"
            ][i].item(),

            linear_schedule[
                "alpha_bar"
            ][i].item(),

            linear_schedule[
                "snr"
            ][i].item(),

            cosine_schedule[
                "beta"
            ][i].item(),

            cosine_schedule[
                "alpha_bar"
            ][i].item(),

            cosine_schedule[
                "snr"
            ][i].item(),
        ]

        print(
            f"{t:6d} | "
            f"{values[0]:12.6f} | "
            f"{values[1]:14.6e} | "
            f"{values[2]:14.6e} | "
            f"{values[3]:12.6f} | "
            f"{values[4]:14.6e} | "
            f"{values[5]:14.6e}"
        )


# ==============================================================================
# VISUALIZATION — SCHEDULES
# ==============================================================================

def plot_schedule_curves(
    linear_schedule: dict,
    cosine_schedule: dict,
):

    t = np.arange(
        1,
        T + 1,
    )

    # --------------------------------------------------------------------------
    # BETA
    # --------------------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        t,
        linear_schedule[
            "beta"
        ].detach().cpu().numpy(),
        label="Linear beta",
    )

    plt.plot(
        t,
        cosine_schedule[
            "beta"
        ].detach().cpu().numpy(),
        label="Cosine beta",
    )

    plt.xlabel(
        "Diffusion timestep t"
    )

    plt.ylabel(
        "beta_t"
    )

    plt.title(
        "Noise Schedules"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    # --------------------------------------------------------------------------
    # ALPHA-BAR
    # --------------------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        t,
        linear_schedule[
            "alpha_bar"
        ].detach().cpu().numpy(),
        label="Linear beta → alpha_bar",
    )

    plt.plot(
        t,
        cosine_schedule[
            "alpha_bar"
        ].detach().cpu().numpy(),
        label="Cosine alpha_bar",
    )

    plt.xlabel(
        "Diffusion timestep t"
    )

    plt.ylabel(
        "alpha_bar_t"
    )

    plt.title(
        "Cumulative Signal Retention"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    # --------------------------------------------------------------------------
    # SIGNAL / NOISE
    # --------------------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        t,
        linear_schedule[
            "alpha_bar"
        ].detach().cpu().numpy(),
        label="Signal power",
    )

    plt.plot(
        t,
        (
            1.0
            -
            linear_schedule[
                "alpha_bar"
            ]
        ).detach().cpu().numpy(),
        label="Noise power",
    )

    plt.xlabel(
        "Diffusion timestep t"
    )

    plt.ylabel(
        "Power"
    )

    plt.title(
        "Linear Schedule: Signal vs Noise"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    # --------------------------------------------------------------------------
    # SNR
    # --------------------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.semilogy(
        t,
        linear_schedule[
            "snr"
        ].detach().cpu().numpy(),
        label="Linear SNR",
    )

    plt.semilogy(
        t,
        cosine_schedule[
            "snr"
        ].detach().cpu().numpy(),
        label="Cosine SNR",
    )

    plt.xlabel(
        "Diffusion timestep t"
    )

    plt.ylabel(
        "SNR"
    )

    plt.title(
        "Signal-to-Noise Ratio"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    # --------------------------------------------------------------------------
    # LOG-SNR
    # --------------------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        t,
        linear_schedule[
            "log_snr"
        ].detach().cpu().numpy(),
        label="Linear log-SNR",
    )

    plt.plot(
        t,
        cosine_schedule[
            "log_snr"
        ].detach().cpu().numpy(),
        label="Cosine log-SNR",
    )

    plt.xlabel(
        "Diffusion timestep t"
    )

    plt.ylabel(
        "log(SNR)"
    )

    plt.title(
        "Log Signal-to-Noise Ratio"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — IMAGES
# ==============================================================================

def plot_corrupted_images(
    x0: torch.Tensor,
    comparison: dict,
    timesteps_to_show: list[int],
):

    columns = (
        len(timesteps_to_show)
        +
        1
    )

    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(
            3.0 * columns,
            6.0,
        ),
    )

    clean = (
        x0[0]
        .detach()
        .cpu()
        .numpy()
    )

    axes[0, 0].imshow(
        clean,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    axes[0, 0].set_title(
        "Clean"
    )

    axes[1, 0].imshow(
        clean,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    axes[1, 0].set_title(
        "Clean"
    )

    for column, t in enumerate(
        timesteps_to_show,
        start=1,
    ):

        linear_image = (
            comparison[
                "linear"
            ][t]["image"][0]
        )

        cosine_image = (
            comparison[
                "cosine"
            ][t]["image"][0]
        )

        axes[0, column].imshow(
            linear_image,
            cmap="gray",
            vmin=-2.5,
            vmax=2.5,
        )

        axes[0, column].set_title(
            f"Linear t={t}"
        )

        axes[1, column].imshow(
            cosine_image,
            cmap="gray",
            vmin=-2.5,
            vmax=2.5,
        )

        axes[1, column].set_title(
            f"Cosine t={t}"
        )

    for row in range(2):

        for column in range(
            columns
        ):

            axes[
                row,
                column
            ].axis("off")

    plt.suptitle(
        "Diffusion Corruption Comparison"
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — PIXEL TRAJECTORY
# ==============================================================================

def plot_pixel_trajectory(
    x0: torch.Tensor,
    linear_schedule: dict,
    cosine_schedule: dict,
):

    pixel_y = IMAGE_SIZE // 2

    pixel_x = IMAGE_SIZE // 2

    x0_pixel = x0[
        0,
        pixel_y,
        pixel_x,
    ]

    t_values = torch.arange(
        1,
        T + 1,
        device=x0.device,
    )

    generator = torch.Generator(
        device=x0.device
    )

    generator.manual_seed(
        SEED + 100
    )

    noise = torch.randn(
        T,
        device=x0.device,
        dtype=x0.dtype,
        generator=generator,
    )

    linear_values = (
        linear_schedule[
            "sqrt_alpha_bar"
        ]
        * x0_pixel
        +
        linear_schedule[
            "sqrt_one_minus_alpha_bar"
        ]
        * noise
    )

    cosine_values = (
        cosine_schedule[
            "sqrt_alpha_bar"
        ]
        * x0_pixel
        +
        cosine_schedule[
            "sqrt_one_minus_alpha_bar"
        ]
        * noise
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        t_values.detach().cpu().numpy(),
        linear_values.detach().cpu().numpy(),
        label="Linear",
    )

    plt.plot(
        t_values.detach().cpu().numpy(),
        cosine_values.detach().cpu().numpy(),
        label="Cosine",
    )

    plt.axhline(
        float(x0_pixel.item()),
        linestyle="--",
        label="Original pixel",
    )

    plt.xlabel(
        "Diffusion timestep t"
    )

    plt.ylabel(
        "Pixel value"
    )

    plt.title(
        "Pixel Diffusion Trajectory"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — DISTRIBUTIONS
# ==============================================================================

def plot_noise_distribution(
    comparison: dict,
    timesteps_to_show: list[int],
):

    columns = len(
        timesteps_to_show
    )

    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(
            4.0 * columns,
            7.0,
        ),
        squeeze=False,
    )

    for column, t in enumerate(
        timesteps_to_show
    ):

        linear_image = (
            comparison[
                "linear"
            ][t]["image"]
        )

        cosine_image = (
            comparison[
                "cosine"
            ][t]["image"]
        )

        axes[
            0,
            column
        ].hist(
            linear_image.flatten().numpy(),
            bins=60,
            density=True,
        )

        axes[
            0,
            column
        ].set_title(
            f"Linear t={t}"
        )

        axes[
            1,
            column
        ].hist(
            cosine_image.flatten().numpy(),
            bins=60,
            density=True,
        )

        axes[
            1,
            column
        ].set_title(
            f"Cosine t={t}"
        )

        axes[
            0,
            column
        ].grid(alpha=0.3)

        axes[
            1,
            column
        ].grid(alpha=0.3)

    plt.suptitle(
        "Corrupted Image Value Distributions"
    )

    plt.tight_layout()


# ==============================================================================
# REAL-WORLD EXPERIMENT
# ==============================================================================

def real_world_schedule_experiment(
    x0: torch.Tensor,
    linear_schedule: dict,
    cosine_schedule: dict,
):

    selected_timesteps = [
        1,
        50,
        100,
        250,
        500,
        750,
        900,
        999,
        1000,
    ]

    print()
    print("=" * 110)
    print(
        "REAL-WORLD EXPERIMENT — "
        "CAMERA OBSERVATION CORRUPTION"
    )
    print("=" * 110)

    for schedule_name, schedule in [
        ("Linear", linear_schedule),
        ("Cosine", cosine_schedule),
    ]:

        print()
        print(
            f"Schedule: {schedule_name}"
        )

        print("-" * 110)

        print(
            f"{'t':>6} | "
            f"{'signal':>10} | "
            f"{'noise':>10} | "
            f"{'SNR':>14} | "
            f"{'corr':>10} | "
            f"{'MSE':>10}"
        )

        for t in selected_timesteps:

            noise = torch.randn_like(
                x0
            )

            xt, _ = sample_xt_from_x0(
                x0,
                t,
                schedule,
                noise=noise,
            )

            metrics = compute_image_metrics(
                x0,
                xt,
                noise,
            )

            signal = schedule_value(
                schedule,
                "sqrt_alpha_bar",
                t,
                device=x0.device,
                dtype=x0.dtype,
            ).item()

            noise_coeff = schedule_value(
                schedule,
                "sqrt_one_minus_alpha_bar",
                t,
                device=x0.device,
                dtype=x0.dtype,
            ).item()

            snr = schedule_value(
                schedule,
                "snr",
                t,
                device=x0.device,
                dtype=x0.dtype,
            ).item()

            print(
                f"{t:6d} | "
                f"{signal:10.6f} | "
                f"{noise_coeff:10.6f} | "
                f"{snr:14.6e} | "
                f"{metrics['correlation']:10.6f} | "
                f"{metrics['mse']:10.6f}"
            )


# ==============================================================================
# FINAL STATE TEST
# ==============================================================================

def compare_final_state_statistics(
    x0: torch.Tensor,
    linear_schedule: dict,
    cosine_schedule: dict,
    num_samples: int = 32,
):

    print()
    print("=" * 80)
    print(
        "TEST — FINAL DIFFUSION STATE"
    )
    print("=" * 80)

    for name, schedule in [
        ("Linear", linear_schedule),
        ("Cosine", cosine_schedule),
    ]:

        samples = []

        for _ in range(
            num_samples
        ):

            noise = torch.randn_like(
                x0
            )

            xt, _ = sample_xt_from_x0(
                x0,
                T,
                schedule,
                noise=noise,
            )

            samples.append(xt)

        batch = torch.cat(
            samples,
            dim=0,
        )

        mean = batch.mean().item()

        std = batch.std().item()

        alpha_bar_final = (
            schedule[
                "alpha_bar"
            ][-1].item()
        )

        print(
            f"{name:10s} | "
            f"alpha_bar_T="
            f"{alpha_bar_final:.3e} | "
            f"mean={mean:+.5f} | "
            f"std={std:.5f}"
        )

        assert abs(mean) < 0.20

        assert (
            0.75
            <
            std
            <
            1.25
        )

    print(
        "[PASS] Final state is noise dominated."
    )


# ==============================================================================
# ROBUST DEVICE DESCRIPTION
# ==============================================================================

def device_description(
    device: torch.device,
) -> str:

    device_type = device.type

    if device.index is None:

        device_index = (
            torch.cuda.current_device()
            if device_type == "cuda"
            and torch.cuda.is_available()
            else None
        )

    else:

        device_index = device.index

    if device_index is None:

        return device_type

    return (
        f"{device_type}:{device_index}"
    )


# ==============================================================================
# ROBUST DEVICE MATCH TEST
# ==============================================================================

def devices_match(
    a: torch.device,
    b: torch.device,
) -> bool:
    """
    Robust device comparison.

    We compare:

        device type

    and, for CUDA:

        actual CUDA device index.

    This avoids brittle direct object comparison.

    For CPU:

        cpu == cpu

    For CUDA:

        cuda:0 == cuda:0

    regardless of how the device object was originally constructed.
    """

    if a.type != b.type:
        return False

    if a.type == "cuda":

        a_index = (
            torch.cuda.current_device()
            if a.index is None
            else a.index
        )

        b_index = (
            torch.cuda.current_device()
            if b.index is None
            else b.index
        )

        return a_index == b_index

    return True


# ==============================================================================
# TEST 10 — DEVICE CONSISTENCY
# ==============================================================================

def verify_device_consistency(
    x0: torch.Tensor,
    linear_schedule: dict,
):

    t = 500

    xt, noise = sample_xt_from_x0(
        x0,
        t,
        linear_schedule,
    )

    beta_device = (
        linear_schedule[
            "beta"
        ].device
    )

    alpha_bar_device = (
        linear_schedule[
            "alpha_bar"
        ].device
    )

    print()
    print("=" * 80)
    print(
        "TEST — DEVICE CONSISTENCY"
    )
    print("=" * 80)

    print(
        f"Configured DEVICE : "
        f"{device_description(DEVICE)}"
    )

    print(
        f"x0 device         : "
        f"{device_description(x0.device)}"
    )

    print(
        f"xt device         : "
        f"{device_description(xt.device)}"
    )

    print(
        f"noise device      : "
        f"{device_description(noise.device)}"
    )

    print(
        f"beta device       : "
        f"{device_description(beta_device)}"
    )

    print(
        f"alpha_bar device  : "
        f"{device_description(alpha_bar_device)}"
    )

    # --------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT use:
    #
    #     assert x0.device == DEVICE
    #
    # as the only test.
    #
    # Instead verify the actual runtime devices.
    # --------------------------------------------------------------------------

    assert devices_match(
        x0.device,
        DEVICE,
    ), (
        "x0 is not on the configured device: "
        f"x0={x0.device}, DEVICE={DEVICE}"
    )

    assert devices_match(
        xt.device,
        x0.device,
    ), (
        "xt and x0 are on different devices: "
        f"xt={xt.device}, x0={x0.device}"
    )

    assert devices_match(
        noise.device,
        x0.device,
    ), (
        "noise and x0 are on different devices: "
        f"noise={noise.device}, x0={x0.device}"
    )

    assert devices_match(
        beta_device,
        x0.device,
    ), (
        "beta and x0 are on different devices: "
        f"beta={beta_device}, x0={x0.device}"
    )

    assert devices_match(
        alpha_bar_device,
        x0.device,
    ), (
        "alpha_bar and x0 are on different devices: "
        f"alpha_bar={alpha_bar_device}, "
        f"x0={x0.device}"
    )

    print(
        "[PASS] All diffusion tensors use compatible devices."
    )


# ==============================================================================
# TIMING EXPERIMENT
# ==============================================================================

def timing_experiment(
    x0: torch.Tensor,
    linear_schedule: dict,
):

    num_iterations = 100

    print()
    print("=" * 80)
    print(
        "EXPERIMENT — CLOSED-FORM SAMPLING"
    )
    print("=" * 80)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(
        num_iterations
    ):

        noise = torch.randn_like(
            x0
        )

        sample_xt_from_x0(
            x0,
            750,
            linear_schedule,
            noise=noise,
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    print(
        f"{num_iterations} samples: "
        f"{elapsed:.4f} seconds"
    )

    print(
        "Closed-form sampling jumps directly "
        "to any timestep."
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    set_seed()

    print()
    print("=" * 80)
    print(
        "LESSON 17 — NOISE SCHEDULES"
    )
    print("=" * 80)

    print(
        f"PyTorch version : "
        f"{torch.__version__}"
    )

    print(
        f"Device          : "
        f"{device_description(DEVICE)}"
    )

    print(
        f"Test device     : "
        f"{device_description(TEST_DEVICE)}"
    )

    print(
        f"T               : "
        f"{T}"
    )

    # ==========================================================================
    # CLEAN IMAGE
    # ==========================================================================

    x0 = create_structured_image(
        device=DEVICE,
        dtype=torch.float32,
    )

    print()
    print(
        "Clean image:"
    )

    print(
        f"shape  : {tuple(x0.shape)}"
    )

    print(
        f"device : {device_description(x0.device)}"
    )

    print(
        f"dtype  : {x0.dtype}"
    )

    print(
        f"min    : {x0.min().item():.4f}"
    )

    print(
        f"max    : {x0.max().item():.4f}"
    )

    print(
        f"mean   : {x0.mean().item():.4f}"
    )

    print(
        f"std    : {x0.std().item():.4f}"
    )

    # ==========================================================================
    # LINEAR SCHEDULE
    # ==========================================================================

    linear_beta = (
        make_linear_beta_schedule(
            timesteps=T,
            beta_start=BETA_START,
            beta_end=BETA_END,
            device=DEVICE,
            dtype=torch.float32,
        )
    )

    linear_schedule = (
        prepare_schedule(
            linear_beta
        )
    )

    # ==========================================================================
    # COSINE SCHEDULE
    # ==========================================================================

    cosine_alpha_bar = (
        make_cosine_alpha_bar_schedule(
            timesteps=T,
            s=COSINE_S,
            device=DEVICE,
            dtype=torch.float32,
        )
    )

    cosine_beta = (
        beta_from_alpha_bar(
            cosine_alpha_bar
        )
    )

    cosine_schedule = (
        prepare_schedule(
            cosine_beta
        )
    )

    # ==========================================================================
    # TESTS
    # ==========================================================================

    verify_schedule_structure(
        linear_schedule,
        "Linear",
    )

    verify_schedule_structure(
        cosine_schedule,
        "Cosine",
    )

    verify_signal_noise_power(
        linear_schedule,
        "Linear",
    )

    verify_signal_noise_power(
        cosine_schedule,
        "Cosine",
    )

    verify_cosine_derivation(
        cosine_alpha_bar,
        cosine_schedule,
    )

    verify_cosine_clipping_behavior(
        cosine_alpha_bar,
        cosine_schedule,
    )

    verify_sampling_statistics(
        linear_schedule,
        t=500,
        num_samples=100000,
    )

    verify_sampling_statistics(
        cosine_schedule,
        t=500,
        num_samples=100000,
    )

    verify_device_consistency(
        x0,
        linear_schedule,
    )

    # ==========================================================================
    # TABLE
    # ==========================================================================

    selected_timesteps = [
        1,
        100,
        250,
        500,
        750,
        900,
        999,
        1000,
    ]

    print_schedule_table(
        linear_schedule,
        cosine_schedule,
        selected_timesteps,
    )

    # ==========================================================================
    # REAL-WORLD EXPERIMENT
    # ==========================================================================

    real_world_schedule_experiment(
        x0,
        linear_schedule,
        cosine_schedule,
    )

    # ==========================================================================
    # FINAL STATE
    # ==========================================================================

    compare_final_state_statistics(
        x0,
        linear_schedule,
        cosine_schedule,
        num_samples=32,
    )

    # ==========================================================================
    # IMAGE COMPARISON
    # ==========================================================================

    comparison_timesteps = [
        1,
        100,
        250,
        500,
        750,
        1000,
    ]

    comparison = (
        generate_schedule_comparison(
            x0,
            linear_schedule,
            cosine_schedule,
            comparison_timesteps,
        )
    )

    # ==========================================================================
    # TIMING
    # ==========================================================================

    timing_experiment(
        x0,
        linear_schedule,
    )

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================

    print()
    print("=" * 80)
    print(
        "GENERATING VISUALIZATIONS"
    )
    print("=" * 80)

    plot_schedule_curves(
        linear_schedule,
        cosine_schedule,
    )

    plot_corrupted_images(
        x0,
        comparison,
        comparison_timesteps,
    )

    plot_pixel_trajectory(
        x0,
        linear_schedule,
        cosine_schedule,
    )

    plot_noise_distribution(
        comparison,
        [
            100,
            500,
            1000,
        ],
    )

    # ==========================================================================
    # FINAL RESULT
    # ==========================================================================

    print()
    print("=" * 80)
    print(
        "LESSON 17 RESULT"
    )
    print("=" * 80)

    print(
        """
Noise scheduling is the time structure of diffusion.

The complete chain is:

    beta_t
        |
        v
    alpha_t = 1-beta_t
        |
        v
    alpha_bar_t
        |
        +----------------------+
        |                      |
        v                      v
    signal power           noise power
        |                      |
        +----------+-----------+
                   |
                   v
                  SNR

We implemented both:

    Linear beta scheduling

and:

    Cosine alpha_bar scheduling.

We also separated:

    raw mathematical beta

from:

    numerically safe beta.

The device verification now checks actual runtime device
compatibility instead of relying on a brittle direct device-object
comparison.

Most importantly:

    Lesson 16 taught HOW to add noise.

    Lesson 17 taught HOW TO CONTROL the noise trajectory.

    Lesson 18 will teach HOW TO LEARN the noise.
"""
    )

    print()
    print("=" * 80)
    print(
        "ALL LESSON 17 TESTS PASSED"
    )
    print("=" * 80)

    print()
    print(
        "Next: Lesson 18 — Learning to Predict Noise"
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()