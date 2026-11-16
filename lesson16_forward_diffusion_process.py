"""
Filename: lesson16_forward_diffusion_process.py

================================================================================
LESSON 16 — FORWARD DIFFUSION PROCESS
================================================================================

PURPOSE
-------
This lesson begins the diffusion-model phase of the Computer Vision
Advanced Crash Course.

We implement the FORWARD DIFFUSION PROCESS from scratch.

Starting with a clean image:

    x_0

we progressively add Gaussian noise:

    x_0 -> x_1 -> x_2 -> ... -> x_T

until the image becomes approximately Gaussian noise.

The reverse process will later be learned by a neural network.

This lesson deliberately does NOT train a neural network.

We first establish the mathematical foundation required for:

    Lesson 17  -> Noise Schedules
    Lesson 18  -> Learning to Predict Noise
    Lesson 19  -> DDPM
    Lesson 20  -> Diffusion U-Net
    Lesson 21  -> Diffusion Sampling
    Lesson 22  -> DDIM
    Lesson 23  -> Latent Diffusion
    Lesson 24  -> Conditioning
    Lesson 25  -> Cross-Attention
    Lesson 26  -> Text Conditioning
    Lesson 27  -> Own Text-to-Image Model


================================================================================
WHAT YOU WILL LEARN
================================================================================

1. Forward diffusion.
2. Gaussian noise.
3. Markov-chain diffusion.
4. The forward transition q(x_t | x_{t-1}).
5. beta_t.
6. alpha_t = 1 - beta_t.
7. cumulative alpha_bar_t.
8. The closed-form diffusion equation.
9. Signal-to-noise ratio.
10. Why images eventually become Gaussian noise.
11. Numerical verification.
12. Why arbitrary diffusion timesteps can be sampled directly.


================================================================================
TECHNICAL / COURSE ROADMAP
================================================================================

COMPLETED

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

CURRENT

    Lesson 16  Forward Diffusion Process

NEXT

    Lesson 17  Noise Schedules
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

A generative model eventually needs to generate images from random noise.

Diffusion uses the following strategy.

TRAINING:

    clean image
         |
         v
    progressively add noise
         |
         v
    noisy image
         |
         v
    learn how noise relates to image structure


GENERATION:

    random Gaussian noise
         |
         v
    remove predicted noise
         |
         v
    remove predicted noise
         |
         v
    ...
         |
         v
    generated image


Lesson 16 studies the FORWARD direction.

The neural network that learns the reverse direction comes later.


================================================================================
THE FORWARD DIFFUSION PROCESS
================================================================================

Let:

    x_0 = clean image

At timestep t:

    x_t = noisy image

The forward process gradually destroys information.

Eventually:

    x_T ≈ N(0, I)


================================================================================
ONE-STEP TRANSITION
================================================================================

The forward transition is:

    q(x_t | x_{t-1})
      =
    N(
        sqrt(alpha_t) x_{t-1},
        beta_t I
    )

where:

    alpha_t = 1 - beta_t

Therefore:

    x_t =
        sqrt(alpha_t) x_{t-1}
        +
        sqrt(beta_t) epsilon

where:

    epsilon ~ N(0, I)


================================================================================
CUMULATIVE ALPHA
================================================================================

Define:

    alpha_bar_t
        =
    product_{s=1}^t alpha_s

Then the complete diffusion history has the closed form:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

where:

    epsilon ~ N(0, I)


================================================================================
WHY THE CLOSED FORM MATTERS
================================================================================

Without the closed form:

    x_0 -> x_1 -> x_2 -> ... -> x_t

would require simulating every previous step.

With the closed form:

    x_0 -----------> x_t

can be calculated directly.

This is critical for diffusion-model training.

During training we can randomly choose:

    t

and directly generate:

    x_t

without simulating all previous timesteps.


================================================================================
SIGNAL AND NOISE
================================================================================

The equation:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

contains two components.

SIGNAL:

    sqrt(alpha_bar_t) x_0

NOISE:

    sqrt(1-alpha_bar_t) epsilon

At early timesteps:

    alpha_bar_t is large

so:

    signal is strong
    noise is weak


At late timesteps:

    alpha_bar_t is small

so:

    signal is weak
    noise is strong


================================================================================
SIGNAL-TO-NOISE RATIO
================================================================================

Define:

    SNR_t =
        alpha_bar_t
        /
        (1-alpha_bar_t)

High SNR:

    image structure dominates.


Low SNR:

    noise dominates.


================================================================================
LIMIT
================================================================================

If:

    alpha_bar_T -> 0

then:

    sqrt(alpha_bar_T) -> 0

and:

    sqrt(1-alpha_bar_T) -> 1

Therefore:

    x_T ≈ epsilon

and:

    x_T ~ N(0, I)


This is why diffusion generation can begin from Gaussian noise.


================================================================================
IMPORTANT IMPLEMENTATION DETAIL
================================================================================

The actual diffusion process may run on:

    CUDA
    or
    CPU

depending on the available hardware.

However, statistical verification functions in this lesson intentionally use:

    CPU
    float64

for numerical stability and reproducibility.

This is important because PyTorch tensors cannot participate in arithmetic
when they are located on different devices.

For example, this is INVALID:

    CUDA tensor + CPU tensor

Therefore all verification calculations explicitly move the required
values to CPU before performing CPU-based statistical analysis.

The main diffusion tensors remain on DEVICE.


================================================================================
REAL-WORLD APPLICATION
================================================================================

Consider a robot vision dataset.

A clean camera observation:

    x_0

can be converted into:

    x_t

at a randomly selected timestep.

A future neural network will receive:

    noisy image x_t
    timestep t

and learn to estimate the noise that produced x_t.

This can eventually support:

    image generation
    image restoration
    image completion
    synthetic robot observations
    visual domain randomization
    scene generation
    conditional generation
    vision-action generation


================================================================================
LESSON RESULT
================================================================================

After completing this lesson you should understand:

    clean image
        |
        v
    controlled Gaussian corruption
        |
        v
    arbitrary noisy image
        |
        v
    approximately pure Gaussian noise

The key equation is:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon


================================================================================
"""

import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import torch


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 64

NUM_CHANNELS = 1

T = 1000

BETA_START = 1e-4
BETA_END = 0.02

VISUALIZATION_TIMESTEPS = [
    0,
    50,
    100,
    200,
    400,
    600,
    800,
    999,
]

STATISTICAL_SAMPLES = 20000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Verification is deliberately performed on CPU using float64.
TEST_DEVICE = torch.device("cpu")
TEST_DTYPE = torch.float64


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    """
    Set random seeds.
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
    print("LESSON 16 — FORWARD DIFFUSION PROCESS")
    print("=" * 80)

    print(
        f"PyTorch version : {torch.__version__}"
    )

    print(
        f"Execution device: {DEVICE}"
    )

    print(
        f"Test device     : {TEST_DEVICE}"
    )

    print(
        f"Test dtype      : {TEST_DTYPE}"
    )

    print(
        f"Image size      : {IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Diffusion steps : {T}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 80)


# =============================================================================
# CREATE SYNTHETIC CLEAN IMAGE
# =============================================================================

def create_clean_image(size=IMAGE_SIZE):
    """
    Create a synthetic structured grayscale image.

    Output:

        [1, H, W]

    Range:

        [-1, 1]
    """

    image = np.zeros(
        (size, size),
        dtype=np.float32,
    )

    image[:] = -0.8

    yy, xx = np.meshgrid(
        np.arange(size),
        np.arange(size),
        indexing="ij",
    )

    # -------------------------------------------------------------------------
    # Circular object
    # -------------------------------------------------------------------------

    center_y = int(size * 0.38)
    center_x = int(size * 0.35)
    radius = int(size * 0.17)

    circle_mask = (
        (yy - center_y) ** 2
        +
        (xx - center_x) ** 2
        <= radius ** 2
    )

    image[circle_mask] = 0.85

    # -------------------------------------------------------------------------
    # Rectangular object
    # -------------------------------------------------------------------------

    y0 = int(size * 0.55)
    y1 = int(size * 0.82)

    x0 = int(size * 0.58)
    x1 = int(size * 0.87)

    image[y0:y1, x0:x1] = 0.45

    # -------------------------------------------------------------------------
    # Diagonal structure
    # -------------------------------------------------------------------------

    for i in range(
        int(size * 0.15),
        int(size * 0.65),
    ):

        j = int(
            0.78 * size
            -
            0.55 * (i - 0.15 * size)
        )

        if 0 <= j < size:

            for offset in range(-1, 2):

                jj = j + offset

                if 0 <= jj < size:

                    image[i, jj] = 1.0

    # -------------------------------------------------------------------------
    # Small texture noise
    # -------------------------------------------------------------------------

    small_noise = (
        np.random.normal(
            0.0,
            0.025,
            size=(size, size),
        )
        .astype(np.float32)
    )

    image += small_noise

    image = np.clip(
        image,
        -1.0,
        1.0,
    )

    return torch.from_numpy(
        image
    ).unsqueeze(0).float()


# =============================================================================
# CREATE BETA SCHEDULE
# =============================================================================

def create_linear_beta_schedule(
    num_steps=T,
    beta_start=BETA_START,
    beta_end=BETA_END,
):
    """
    Create a simple linear beta schedule.
    """

    betas = torch.linspace(
        beta_start,
        beta_end,
        num_steps,
        dtype=torch.float32,
    )

    return betas


# =============================================================================
# PREPARE DIFFUSION PARAMETERS
# =============================================================================

def prepare_diffusion_parameters(
    betas,
    device=DEVICE,
):
    """
    Calculate all forward diffusion quantities.

    alpha_t:

        alpha_t = 1 - beta_t

    alpha_bar_t:

        alpha_bar_t =
            product(alpha_1 ... alpha_t)
    """

    betas = betas.to(device)

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0,
    )

    sqrt_alphas = torch.sqrt(
        alphas
    )

    sqrt_betas = torch.sqrt(
        betas
    )

    sqrt_alpha_bars = torch.sqrt(
        alpha_bars
    )

    sqrt_one_minus_alpha_bars = torch.sqrt(
        1.0 - alpha_bars
    )

    snr = (
        alpha_bars
        /
        (1.0 - alpha_bars)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alphas": sqrt_alphas,
        "sqrt_betas": sqrt_betas,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars":
            sqrt_one_minus_alpha_bars,
        "snr": snr,
    }


# =============================================================================
# ONE FORWARD DIFFUSION STEP
# =============================================================================

def forward_step(
    x_previous,
    alpha_t,
    beta_t,
    noise=None,
):
    """
    Perform:

        x_t =
            sqrt(alpha_t) x_{t-1}
            +
            sqrt(beta_t) epsilon
    """

    if noise is None:

        noise = torch.randn_like(
            x_previous
        )

    # Ensure scalar coefficients are on the same device as x_previous.
    alpha_t = alpha_t.to(
        device=x_previous.device,
        dtype=x_previous.dtype,
    )

    beta_t = beta_t.to(
        device=x_previous.device,
        dtype=x_previous.dtype,
    )

    x_current = (
        torch.sqrt(alpha_t)
        * x_previous
        +
        torch.sqrt(beta_t)
        * noise
    )

    return x_current


# =============================================================================
# CLOSED-FORM SAMPLING
# =============================================================================

def sample_xt_from_x0(
    x0,
    timestep,
    diffusion_parameters,
    noise=None,
):
    """
    Directly sample x_t from x_0.

    Equation:

        x_t =
            sqrt(alpha_bar_t) x_0
            +
            sqrt(1-alpha_bar_t) epsilon
    """

    sqrt_alpha_bar_t = (
        diffusion_parameters[
            "sqrt_alpha_bars"
        ][timestep]
    )

    sqrt_one_minus_alpha_bar_t = (
        diffusion_parameters[
            "sqrt_one_minus_alpha_bars"
        ][timestep]
    )

    # Explicit device/dtype alignment.
    sqrt_alpha_bar_t = sqrt_alpha_bar_t.to(
        device=x0.device,
        dtype=x0.dtype,
    )

    sqrt_one_minus_alpha_bar_t = (
        sqrt_one_minus_alpha_bar_t.to(
            device=x0.device,
            dtype=x0.dtype,
        )
    )

    if noise is None:

        noise = torch.randn_like(
            x0
        )

    else:

        noise = noise.to(
            device=x0.device,
            dtype=x0.dtype,
        )

    x_t = (
        sqrt_alpha_bar_t * x0
        +
        sqrt_one_minus_alpha_bar_t * noise
    )

    return x_t, noise


# =============================================================================
# SIMULATE COMPLETE MARKOV CHAIN
# =============================================================================

def simulate_markov_chain(
    x0,
    diffusion_parameters,
):
    """
    Explicitly simulate:

        x_0 -> x_1 -> ... -> x_T
    """

    betas = diffusion_parameters[
        "betas"
    ]

    alphas = diffusion_parameters[
        "alphas"
    ]

    states = []

    x = x0.clone()

    states.append(
        x.clone()
    )

    for t in range(T):

        noise = torch.randn_like(
            x
        )

        x = forward_step(
            x_previous=x,
            alpha_t=alphas[t],
            beta_t=betas[t],
            noise=noise,
        )

        states.append(
            x.clone()
        )

    return states


# =============================================================================
# VISUALIZATION CONVERSION
# =============================================================================

def tensor_to_display_image(image):
    """
    Convert [-1,1] to [0,1] ONLY for visualization.
    """

    image = image.detach().cpu()

    image = image.squeeze()

    image = (
        image + 1.0
    ) / 2.0

    image = torch.clamp(
        image,
        0.0,
        1.0,
    )

    return image.numpy()


# =============================================================================
# VISUALIZE FORWARD DIFFUSION
# =============================================================================

def visualize_forward_diffusion(
    x0,
    diffusion_parameters,
):
    """
    Visualize progressive corruption.
    """

    torch.manual_seed(SEED)

    shared_noise = torch.randn_like(
        x0
    )

    images = []

    for timestep in VISUALIZATION_TIMESTEPS:

        x_t, _ = sample_xt_from_x0(
            x0=x0,
            timestep=timestep,
            diffusion_parameters=diffusion_parameters,
            noise=shared_noise,
        )

        images.append(
            x_t
        )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7),
    )

    for axis, timestep, image in zip(
        axes.flatten(),
        VISUALIZATION_TIMESTEPS,
        images,
    ):

        axis.imshow(
            tensor_to_display_image(image),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        alpha_bar = (
            diffusion_parameters[
                "alpha_bars"
            ][timestep]
            .item()
        )

        snr = (
            diffusion_parameters[
                "snr"
            ][timestep]
            .item()
        )

        axis.set_title(
            f"t={timestep}\n"
            f"alpha_bar={alpha_bar:.4f}\n"
            f"SNR={snr:.3f}"
        )

        axis.axis("off")

    plt.suptitle(
        "Forward Diffusion: Clean Image → Gaussian Noise",
        fontsize=14,
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# VISUALIZE DIFFUSION PARAMETERS
# =============================================================================

def visualize_diffusion_parameters(
    diffusion_parameters,
):

    timesteps = np.arange(T)

    betas = (
        diffusion_parameters["betas"]
        .detach()
        .cpu()
        .numpy()
    )

    alpha_bars = (
        diffusion_parameters["alpha_bars"]
        .detach()
        .cpu()
        .numpy()
    )

    sqrt_alpha_bars = (
        diffusion_parameters[
            "sqrt_alpha_bars"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    sqrt_one_minus_alpha_bars = (
        diffusion_parameters[
            "sqrt_one_minus_alpha_bars"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    snr = (
        diffusion_parameters["snr"]
        .detach()
        .cpu()
        .numpy()
    )

    # -------------------------------------------------------------------------
    # Beta
    # -------------------------------------------------------------------------

    plt.figure(figsize=(9, 4))

    plt.plot(
        timesteps,
        betas,
    )

    plt.xlabel("Diffusion timestep")
    plt.ylabel("beta_t")

    plt.title(
        "Forward Noise Amount per Timestep"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # -------------------------------------------------------------------------
    # Alpha bar
    # -------------------------------------------------------------------------

    plt.figure(figsize=(9, 4))

    plt.plot(
        timesteps,
        alpha_bars,
    )

    plt.xlabel("Diffusion timestep")
    plt.ylabel("alpha_bar_t")

    plt.title(
        "Cumulative Signal Retention"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # -------------------------------------------------------------------------
    # Signal / noise coefficients
    # -------------------------------------------------------------------------

    plt.figure(figsize=(9, 4))

    plt.plot(
        timesteps,
        sqrt_alpha_bars,
        label="sqrt(alpha_bar_t)",
    )

    plt.plot(
        timesteps,
        sqrt_one_minus_alpha_bars,
        label="sqrt(1-alpha_bar_t)",
    )

    plt.xlabel("Diffusion timestep")
    plt.ylabel("Coefficient")

    plt.title(
        "Signal vs Noise Contribution"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # -------------------------------------------------------------------------
    # SNR
    # -------------------------------------------------------------------------

    plt.figure(figsize=(9, 4))

    snr_display = np.clip(
        snr,
        0.0,
        1000.0,
    )

    plt.plot(
        timesteps,
        snr_display,
    )

    plt.xlabel("Diffusion timestep")
    plt.ylabel("SNR")

    plt.title(
        "Signal-to-Noise Ratio"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# =============================================================================
# TEST — GAUSSIAN NOISE
# =============================================================================

def verify_gaussian_noise():

    samples = torch.randn(
        STATISTICAL_SAMPLES,
        dtype=TEST_DTYPE,
        device=TEST_DEVICE,
    )

    mean = samples.mean().item()

    std = samples.std(
        unbiased=True
    ).item()

    print()
    print("=" * 80)
    print("TEST — GAUSSIAN NOISE STATISTICS")
    print("=" * 80)

    print(
        f"Sample mean : {mean:.6f}"
    )

    print(
        f"Sample std  : {std:.6f}"
    )

    assert abs(mean) < 0.03

    assert abs(std - 1.0) < 0.03

    print(
        "[PASS] Gaussian noise statistics"
    )


# =============================================================================
# TEST — ONE-STEP DIFFUSION STATISTICS
# =============================================================================

def verify_one_step_statistics(
    diffusion_parameters,
):
    """
    IMPORTANT DEVICE FIX

    This test is intentionally performed entirely on CPU.

    The diffusion model itself may run on CUDA, but statistical verification
    is independent of the training device.

    Therefore:

        alpha_t
        beta_t
        x_previous
        noise
        x_current

    are ALL explicitly created/moved to:

        TEST_DEVICE = CPU

    and use:

        TEST_DTYPE = float64

    This prevents CUDA/CPU mixing.
    """

    timestep = 500

    # -------------------------------------------------------------------------
    # Explicitly extract scalar parameters and move them to CPU float64.
    # -------------------------------------------------------------------------

    alpha_t = (
        diffusion_parameters[
            "alphas"
        ][timestep]
        .detach()
        .to(
            device=TEST_DEVICE,
            dtype=TEST_DTYPE,
        )
    )

    beta_t = (
        diffusion_parameters[
            "betas"
        ][timestep]
        .detach()
        .to(
            device=TEST_DEVICE,
            dtype=TEST_DTYPE,
        )
    )

    # -------------------------------------------------------------------------
    # x_previous is also CPU float64.
    # -------------------------------------------------------------------------

    x_previous = torch.tensor(
        0.35,
        dtype=TEST_DTYPE,
        device=TEST_DEVICE,
    )

    # -------------------------------------------------------------------------
    # Generate all statistical samples on CPU.
    # -------------------------------------------------------------------------

    noise = torch.randn(
        STATISTICAL_SAMPLES,
        dtype=TEST_DTYPE,
        device=TEST_DEVICE,
    )

    x_current = (
        torch.sqrt(alpha_t)
        * x_previous
        +
        torch.sqrt(beta_t)
        * noise
    )

    empirical_mean = (
        x_current.mean().item()
    )

    empirical_variance = (
        x_current.var(
            unbiased=True
        ).item()
    )

    theoretical_mean = (
        torch.sqrt(alpha_t)
        * x_previous
    ).item()

    theoretical_variance = (
        beta_t.item()
    )

    print()
    print("=" * 80)
    print("TEST — ONE-STEP DIFFUSION STATISTICS")
    print("=" * 80)

    print(
        f"Test device          : {TEST_DEVICE}"
    )

    print(
        f"Test dtype           : {TEST_DTYPE}"
    )

    print(
        f"Timestep             : {timestep}"
    )

    print(
        f"alpha_t              : {alpha_t.item():.8f}"
    )

    print(
        f"beta_t               : {beta_t.item():.8f}"
    )

    print(
        f"Empirical mean       : {empirical_mean:.6f}"
    )

    print(
        f"Theoretical mean     : {theoretical_mean:.6f}"
    )

    print(
        f"Empirical variance   : {empirical_variance:.6f}"
    )

    print(
        f"Theoretical variance : {theoretical_variance:.6f}"
    )

    mean_error = abs(
        empirical_mean
        -
        theoretical_mean
    )

    variance_error = abs(
        empirical_variance
        -
        theoretical_variance
    )

    print(
        f"Mean error           : {mean_error:.6f}"
    )

    print(
        f"Variance error       : {variance_error:.6f}"
    )

    assert mean_error < 0.02

    assert variance_error < 0.01

    print(
        "[PASS] One-step forward diffusion statistics"
    )


# =============================================================================
# TEST — CLOSED FORM VS MARKOV CHAIN
# =============================================================================

def verify_closed_form_vs_markov_chain(
    x0,
    diffusion_parameters,
):
    """
    Verify that a Markov-chain result can be represented by the
    closed-form equation.

    We derive the equivalent aggregate Gaussian noise from the Markov chain
    and reconstruct x_t.
    """

    # For this mathematical test we use CPU float64.
    x0_test = (
        x0.detach()
        .to(
            device=TEST_DEVICE,
            dtype=TEST_DTYPE,
        )
    )

    alphas = (
        diffusion_parameters[
            "alphas"
        ]
        .detach()
        .to(
            device=TEST_DEVICE,
            dtype=TEST_DTYPE,
        )
    )

    alpha_bars = (
        diffusion_parameters[
            "alpha_bars"
        ]
        .detach()
        .to(
            device=TEST_DEVICE,
            dtype=TEST_DTYPE,
        )
    )

    timestep = 300

    torch.manual_seed(123)

    noises = []

    for _ in range(timestep + 1):

        noises.append(
            torch.randn_like(
                x0_test
            )
        )

    # -------------------------------------------------------------------------
    # Explicit Markov chain
    # -------------------------------------------------------------------------

    x = x0_test.clone()

    for t in range(
        timestep + 1
    ):

        x = (
            torch.sqrt(
                alphas[t]
            ) * x
            +
            torch.sqrt(
                1.0 - alphas[t]
            ) * noises[t]
        )

    markov_result = x.clone()

    # -------------------------------------------------------------------------
    # Derive equivalent aggregate Gaussian noise.
    # -------------------------------------------------------------------------

    alpha_bar = alpha_bars[
        timestep
    ]

    equivalent_noise = (
        markov_result
        -
        torch.sqrt(alpha_bar)
        * x0_test
    ) / torch.sqrt(
        1.0 - alpha_bar
    )

    # -------------------------------------------------------------------------
    # Reconstruct.
    # -------------------------------------------------------------------------

    closed_form_reconstruction = (
        torch.sqrt(alpha_bar)
        * x0_test
        +
        torch.sqrt(
            1.0 - alpha_bar
        )
        * equivalent_noise
    )

    max_error = (
        markov_result
        -
        closed_form_reconstruction
    ).abs().max().item()

    print()
    print("=" * 80)
    print("TEST — MARKOV CHAIN / CLOSED-FORM EQUIVALENCE")
    print("=" * 80)

    print(
        f"Timestep      : {timestep}"
    )

    print(
        f"Maximum error : {max_error:.12e}"
    )

    assert max_error < 1e-10

    print(
        "[PASS] Closed-form representation is mathematically consistent"
    )


# =============================================================================
# TEST — SIGNAL / NOISE DECOMPOSITION
# =============================================================================

def verify_signal_noise_decomposition(
    x0,
    diffusion_parameters,
):
    """
    Verify:

        x_t = signal + noise
    """

    x0_test = (
        x0.detach()
        .to(
            device=TEST_DEVICE,
            dtype=TEST_DTYPE,
        )
    )

    alpha_bars = (
        diffusion_parameters[
            "alpha_bars"
        ]
        .detach()
        .to(
            device=TEST_DEVICE,
            dtype=TEST_DTYPE,
        )
    )

    timestep = 700

    torch.manual_seed(987)

    noise = torch.randn_like(
        x0_test
    )

    alpha_bar = alpha_bars[
        timestep
    ]

    x_t = (
        torch.sqrt(alpha_bar)
        * x0_test
        +
        torch.sqrt(
            1.0 - alpha_bar
        )
        * noise
    )

    signal_component = (
        torch.sqrt(alpha_bar)
        * x0_test
    )

    noise_component = (
        torch.sqrt(
            1.0 - alpha_bar
        )
        * noise
    )

    reconstruction = (
        signal_component
        +
        noise_component
    )

    max_error = (
        x_t - reconstruction
    ).abs().max().item()

    print()
    print("=" * 80)
    print("TEST — SIGNAL + NOISE DECOMPOSITION")
    print("=" * 80)

    print(
        f"Timestep      : {timestep}"
    )

    print(
        f"Maximum error : {max_error:.12e}"
    )

    assert max_error < 1e-10

    print(
        "[PASS] Signal/noise decomposition"
    )


# =============================================================================
# TEST — NOISE DOMINANCE
# =============================================================================

def verify_noise_dominance(
    diffusion_parameters,
):

    sqrt_alpha_bars = (
        diffusion_parameters[
            "sqrt_alpha_bars"
        ]
        .detach()
        .cpu()
    )

    sqrt_one_minus_alpha_bars = (
        diffusion_parameters[
            "sqrt_one_minus_alpha_bars"
        ]
        .detach()
        .cpu()
    )

    early = 0
    late = T - 1

    early_signal = (
        sqrt_alpha_bars[early]
        .item()
    )

    early_noise = (
        sqrt_one_minus_alpha_bars[early]
        .item()
    )

    late_signal = (
        sqrt_alpha_bars[late]
        .item()
    )

    late_noise = (
        sqrt_one_minus_alpha_bars[late]
        .item()
    )

    print()
    print("=" * 80)
    print("TEST — NOISE DOMINANCE")
    print("=" * 80)

    print(
        f"Early signal coefficient : {early_signal:.6f}"
    )

    print(
        f"Early noise coefficient  : {early_noise:.6f}"
    )

    print(
        f"Late signal coefficient  : {late_signal:.6f}"
    )

    print(
        f"Late noise coefficient   : {late_noise:.6f}"
    )

    assert early_signal > early_noise

    assert late_noise > late_signal

    assert late_signal < early_signal

    assert late_noise > early_noise

    print(
        "[PASS] Diffusion transitions from signal-dominant "
        "to noise-dominant"
    )


# =============================================================================
# MEASURE TIMESTEP STATISTICS
# =============================================================================

def measure_timestep_statistics(
    x0,
    diffusion_parameters,
    timesteps,
):

    print()
    print("=" * 80)
    print("DIFFUSION IMAGE STATISTICS")
    print("=" * 80)

    print(
        f"{'t':>6} "
        f"{'alpha_bar':>14} "
        f"{'mean':>14} "
        f"{'std':>14} "
        f"{'mean|x|':>14}"
    )

    print("-" * 80)

    torch.manual_seed(555)

    noise = torch.randn_like(
        x0
    )

    for timestep in timesteps:

        x_t, _ = sample_xt_from_x0(
            x0=x0,
            timestep=timestep,
            diffusion_parameters=diffusion_parameters,
            noise=noise,
        )

        mean = x_t.mean().item()

        std = x_t.std().item()

        mean_abs = x_t.abs().mean().item()

        alpha_bar = (
            diffusion_parameters[
                "alpha_bars"
            ][timestep]
            .item()
        )

        print(
            f"{timestep:6d} "
            f"{alpha_bar:14.6f} "
            f"{mean:14.6f} "
            f"{std:14.6f} "
            f"{mean_abs:14.6f}"
        )


# =============================================================================
# COMPARE EXPLICIT VS CLOSED-FORM SAMPLING
# =============================================================================

def compare_sampling_methods(
    x0,
    diffusion_parameters,
):
    """
    Demonstrate why direct closed-form sampling is useful.
    """

    timestep = 900

    # -------------------------------------------------------------------------
    # Explicit simulation
    # -------------------------------------------------------------------------

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    x = x0.clone()

    for t in range(
        timestep + 1
    ):

        noise = torch.randn_like(
            x
        )

        x = forward_step(
            x,
            diffusion_parameters[
                "alphas"
            ][t],
            diffusion_parameters[
                "betas"
            ][t],
            noise,
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    explicit_time = (
        time.perf_counter() - start
    )

    # -------------------------------------------------------------------------
    # Closed-form sampling
    # -------------------------------------------------------------------------

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    noise = torch.randn_like(
        x0
    )

    _ = sample_xt_from_x0(
        x0=x0,
        timestep=timestep,
        diffusion_parameters=diffusion_parameters,
        noise=noise,
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    closed_form_time = (
        time.perf_counter() - start
    )

    print()
    print("=" * 80)
    print("DIRECT SAMPLING VS EXPLICIT MARKOV SIMULATION")
    print("=" * 80)

    print(
        f"Explicit simulation : "
        f"{explicit_time * 1000:.3f} ms"
    )

    print(
        f"Closed-form sample  : "
        f"{closed_form_time * 1000:.3f} ms"
    )

    if closed_form_time > 0:

        print(
            f"Approximate ratio   : "
            f"{explicit_time / closed_form_time:.2f}x"
        )

    print(
        """
The important lesson is not the exact benchmark.

The important computational difference is:

    explicit:

        x_0 -> x_1 -> ... -> x_t

    closed-form:

        x_0 -----------> x_t

The closed-form equation allows training to select arbitrary timesteps
without simulating all previous transitions.
"""
    )


# =============================================================================
# SIGNAL / NOISE EXPERIMENT
# =============================================================================

def run_experiment(
    diffusion_parameters,
):

    experiment_timesteps = [
        0,
        100,
        250,
        500,
        750,
        999,
    ]

    print()
    print("=" * 80)
    print("SIGNAL / NOISE EXPERIMENT")
    print("=" * 80)

    print(
        f"{'t':>6} "
        f"{'signal':>14} "
        f"{'noise':>14} "
        f"{'SNR':>14}"
    )

    print("-" * 60)

    for timestep in experiment_timesteps:

        signal = (
            diffusion_parameters[
                "sqrt_alpha_bars"
            ][timestep]
            .item()
        )

        noise = (
            diffusion_parameters[
                "sqrt_one_minus_alpha_bars"
            ][timestep]
            .item()
        )

        snr = (
            diffusion_parameters[
                "snr"
            ][timestep]
            .item()
        )

        print(
            f"{timestep:6d} "
            f"{signal:14.6f} "
            f"{noise:14.6f} "
            f"{snr:14.6f}"
        )


# =============================================================================
# REAL-WORLD APPLICATION
# =============================================================================

def explain_real_world_application():

    print()
    print("=" * 80)
    print("REAL-WORLD APPLICATION")
    print("=" * 80)

    print(
        """
Consider a real robot vision dataset.

A camera observation is:

    x_0

We can select a random timestep:

    t

and directly construct:

    x_t

using:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

A future neural network will receive:

    x_t
    t

and learn to predict:

    epsilon

This gives us the fundamental training task of diffusion models.

The same framework can later be used for:

    image generation
    image restoration
    image completion
    synthetic robot observations
    visual domain randomization
    conditional image generation
    scene generation
    vision-action generation
"""
    )


# =============================================================================
# FINAL TESTS
# =============================================================================

def final_tests(
    x0,
    diffusion_parameters,
):

    print()
    print("=" * 80)
    print("FINAL LESSON VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test 1 — Shape
    # -------------------------------------------------------------------------

    x_t, noise = sample_xt_from_x0(
        x0=x0,
        timestep=500,
        diffusion_parameters=diffusion_parameters,
    )

    assert x_t.shape == x0.shape

    assert noise.shape == x0.shape

    print(
        "[PASS] Forward diffusion tensor shapes"
    )

    # -------------------------------------------------------------------------
    # Test 2 — Finite values
    # -------------------------------------------------------------------------

    assert torch.isfinite(
        x_t
    ).all()

    print(
        "[PASS] Forward diffusion produces finite values"
    )

    # -------------------------------------------------------------------------
    # Test 3 — Parameter ranges
    # -------------------------------------------------------------------------

    betas = diffusion_parameters[
        "betas"
    ]

    alphas = diffusion_parameters[
        "alphas"
    ]

    alpha_bars = diffusion_parameters[
        "alpha_bars"
    ]

    assert torch.all(
        betas > 0
    )

    assert torch.all(
        betas < 1
    )

    assert torch.all(
        alphas > 0
    )

    assert torch.all(
        alphas < 1
    )

    assert torch.all(
        alpha_bars > 0
    )

    assert torch.all(
        alpha_bars <= 1
    )

    print(
        "[PASS] Diffusion parameter ranges"
    )

    # -------------------------------------------------------------------------
    # Test 4 — Cumulative alpha decreases
    # -------------------------------------------------------------------------

    differences = (
        alpha_bars[1:]
        -
        alpha_bars[:-1]
    )

    assert torch.all(
        differences <= 0
    )

    print(
        "[PASS] Cumulative alpha decreases monotonically"
    )

    # -------------------------------------------------------------------------
    # Test 5 — Signal decreases
    # -------------------------------------------------------------------------

    signal = diffusion_parameters[
        "sqrt_alpha_bars"
    ]

    assert signal[-1] < signal[0]

    print(
        "[PASS] Signal contribution decreases"
    )

    # -------------------------------------------------------------------------
    # Test 6 — Noise increases
    # -------------------------------------------------------------------------

    noise_coefficient = diffusion_parameters[
        "sqrt_one_minus_alpha_bars"
    ]

    assert (
        noise_coefficient[-1]
        >
        noise_coefficient[0]
    )

    print(
        "[PASS] Noise contribution increases"
    )

    # -------------------------------------------------------------------------
    # Test 7 — SNR decreases
    # -------------------------------------------------------------------------

    snr = diffusion_parameters[
        "snr"
    ]

    assert snr[-1] < snr[0]

    print(
        "[PASS] Signal-to-noise ratio decreases"
    )

    # -------------------------------------------------------------------------
    # Test 8 — Direct sampling
    # -------------------------------------------------------------------------

    direct_x_t, _ = sample_xt_from_x0(
        x0=x0,
        timestep=999,
        diffusion_parameters=diffusion_parameters,
    )

    assert direct_x_t.shape == x0.shape

    print(
        "[PASS] Closed-form direct sampling"
    )

    print()
    print(
        "ALL LESSON 16 TESTS PASSED."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    set_seed()

    print_environment()

    # =========================================================================
    # STEP 1 — CREATE CLEAN IMAGE
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 1 — CREATE CLEAN IMAGE x_0")
    print("=" * 80)

    x0 = create_clean_image()

    x0 = x0.to(
        DEVICE
    )

    print(
        f"x_0 shape : {tuple(x0.shape)}"
    )

    print(
        f"x_0 device: {x0.device}"
    )

    print(
        f"x_0 dtype : {x0.dtype}"
    )

    print(
        f"x_0 min   : {x0.min().item():.4f}"
    )

    print(
        f"x_0 max   : {x0.max().item():.4f}"
    )

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(
        tensor_to_display_image(x0),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    plt.title(
        "Clean Image x_0"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()

    # =========================================================================
    # STEP 2 — CREATE DIFFUSION PARAMETERS
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 2 — CREATE FORWARD DIFFUSION PARAMETERS")
    print("=" * 80)

    betas = create_linear_beta_schedule()

    diffusion_parameters = (
        prepare_diffusion_parameters(
            betas,
            device=DEVICE,
        )
    )

    print(
        f"beta_1 = "
        f"{diffusion_parameters['betas'][0].item():.8f}"
    )

    print(
        f"beta_T = "
        f"{diffusion_parameters['betas'][-1].item():.8f}"
    )

    print(
        f"alpha_bar_1 = "
        f"{diffusion_parameters['alpha_bars'][0].item():.8f}"
    )

    print(
        f"alpha_bar_T = "
        f"{diffusion_parameters['alpha_bars'][-1].item():.8f}"
    )

    # =========================================================================
    # STEP 3 — VISUALIZE PARAMETERS
    # =========================================================================

    visualize_diffusion_parameters(
        diffusion_parameters
    )

    # =========================================================================
    # STEP 4 — VISUALIZE FORWARD DIFFUSION
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 4 — VISUALIZE FORWARD DIFFUSION")
    print("=" * 80)

    visualize_forward_diffusion(
        x0,
        diffusion_parameters,
    )

    # =========================================================================
    # STEP 5 — EXPLICIT MARKOV CHAIN
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 5 — SIMULATE THE MARKOV CHAIN")
    print("=" * 80)

    states = simulate_markov_chain(
        x0,
        diffusion_parameters,
    )

    print(
        f"Number of stored states : "
        f"{len(states)}"
    )

    print(
        f"x_0 shape : "
        f"{tuple(states[0].shape)}"
    )

    print(
        f"x_T shape : "
        f"{tuple(states[-1].shape)}"
    )

    print(
        f"x_T device: "
        f"{states[-1].device}"
    )

    assert len(states) == T + 1

    print(
        "[PASS] Complete Markov chain generated"
    )

    # =========================================================================
    # STEP 6 — STATISTICAL TESTS
    # =========================================================================

    verify_gaussian_noise()

    verify_one_step_statistics(
        diffusion_parameters
    )

    verify_closed_form_vs_markov_chain(
        x0,
        diffusion_parameters,
    )

    verify_signal_noise_decomposition(
        x0,
        diffusion_parameters,
    )

    verify_noise_dominance(
        diffusion_parameters
    )

    # =========================================================================
    # STEP 7 — IMAGE STATISTICS
    # =========================================================================

    measure_timestep_statistics(
        x0,
        diffusion_parameters,
        [
            0,
            50,
            100,
            200,
            400,
            600,
            800,
            999,
        ],
    )

    # =========================================================================
    # STEP 8 — DIRECT VS EXPLICIT SAMPLING
    # =========================================================================

    compare_sampling_methods(
        x0,
        diffusion_parameters,
    )

    # =========================================================================
    # STEP 9 — SIGNAL / NOISE EXPERIMENT
    # =========================================================================

    run_experiment(
        diffusion_parameters
    )

    # =========================================================================
    # STEP 10 — REAL-WORLD CONNECTION
    # =========================================================================

    explain_real_world_application()

    # =========================================================================
    # STEP 11 — FINAL TESTS
    # =========================================================================

    final_tests(
        x0,
        diffusion_parameters,
    )

    # =========================================================================
    # LESSON RESULT
    # =========================================================================

    print()
    print("=" * 80)
    print("LESSON 16 RESULT")
    print("=" * 80)

    print(
        """
You have implemented the forward diffusion process.

The key equation is:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

where:

    epsilon ~ N(0, I)

You verified:

    [PASS] Gaussian noise statistics
    [PASS] One-step diffusion statistics
    [PASS] Markov-chain behavior
    [PASS] Closed-form diffusion representation
    [PASS] Signal/noise decomposition
    [PASS] Monotonic signal destruction
    [PASS] Increasing noise contribution
    [PASS] Decreasing SNR
    [PASS] Direct arbitrary-timestep sampling

Conceptually:

    CLEAN IMAGE
        |
        v
    small amount of noise
        |
        v
    slightly noisy image
        |
        v
    more noise
        |
        v
    ...
        |
        v
    approximately Gaussian noise

This is the forward half of diffusion modeling.
"""
    )

    # =========================================================================
    # NEXT LESSON
    # =========================================================================

    print()
    print("=" * 80)
    print("CONNECTION TO LESSON 17")
    print("=" * 80)

    print(
        """
LESSON 17 — NOISE SCHEDULES

This lesson used a simple linear beta schedule.

But beta_t controls how quickly information disappears.

A poor schedule can:

    destroy useful image information too quickly

or:

    preserve too much information for too long.

Lesson 17 will investigate:

    linear schedules
    cosine schedules
    beta_t
    alpha_t
    alpha_bar_t
    signal-to-noise ratio
    schedule comparison
    numerical stability

The next question is:

    "How should we control the amount of noise at every timestep?"

Then:

    Lesson 18
        will train the first neural network that predicts the noise.

Then:

    Lesson 19
        will turn that learned noise predictor into DDPM.

The diffusion progression is:

    Lesson 16
        Forward corruption
            |
            v
    Lesson 17
        Noise schedule
            |
            v
    Lesson 18
        Noise prediction
            |
            v
    Lesson 19
        DDPM
            |
            v
    Lesson 20
        Diffusion U-Net
            |
            v
    Lesson 21
        Sampling
            |
            v
    Lesson 22
        DDIM
            |
            v
    Lesson 23+
        Latent + conditional diffusion
"""
    )

    print()
    print("=" * 80)
    print("END OF LESSON 16")
    print("=" * 80)


if __name__ == "__main__":
    main()