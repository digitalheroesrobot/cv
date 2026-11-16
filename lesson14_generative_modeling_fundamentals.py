"""
================================================================================
LESSON 14 — GENERATIVE MODELING FUNDAMENTALS
================================================================================

PURPOSE
-------
Move from discriminative vision models to generative models.

Up to Lesson 13, our models answered questions such as:

    "Which class does this image belong to?"

Generative modeling asks:

    "What distribution of data does this dataset come from?"

This lesson builds the mathematical and practical foundation required before
implementing diffusion models.

NO pretrained model is used.
NO pretrained weights are used.
The generator is initialized randomly and trained from scratch.


WHAT YOU WILL LEARN
-------------------
1. Discriminative vs generative modeling.
2. Probability distributions over data.
3. p(x) and the data distribution.
4. Sampling from a learned distribution.
5. Maximum-likelihood intuition.
6. Negative log-likelihood.
7. Latent variables.
8. Random latent vectors.
9. Distribution matching.
10. Why noise is useful for generation.
11. How a neural network can transform noise into samples.
12. Why generative modeling is different from classification.
13. The conceptual bridge from generation to diffusion.


TECHNICAL / COURSE ROADMAP
--------------------------
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
    Lesson 12  Generative / representation foundations
    Lesson 13  Vision Transformer

Current:

    Lesson 14  Generative Modeling Fundamentals

Next:

    Lesson 15  Probability Distributions for Image Generation
    Lesson 16  Forward Diffusion Process
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


REAL-WORLD PROBLEM
------------------
A classifier learns:

    image -> class

A generative model learns the structure of the data itself.

Conceptually:

    examples -> p_data(x)

and then:

    sample x ~ p_theta(x)

Applications include:

    synthetic data
    data augmentation
    simulation
    visual inspection
    rare-event generation
    robotics simulation
    image restoration
    image synthesis
    text-to-image generation


THEORY — DISCRIMINATIVE VS GENERATIVE
--------------------------------------

Discriminative modeling:

    p(y | x)

Question:

    "Given image x, what is label y?"

Generative modeling:

    p(x)

Question:

    "What samples are likely to occur?"

Conditional generative modeling:

    p(x | c)

Question:

    "What samples are likely under condition c?"

Text-to-image generation can conceptually be written:

    p(image | text)


THE DATA DISTRIBUTION
---------------------
Suppose the dataset contains:

    x_1, x_2, ..., x_N

These are observations from an unknown distribution:

    p_data(x)

A generative model attempts to learn:

    p_theta(x) ~= p_data(x)


SAMPLING
--------
Once a distribution has been learned, we want to draw new samples:

    x_new ~ p_theta(x)


MAXIMUM LIKELIHOOD
------------------
A classical objective is:

    maximize product_i p_theta(x_i)

Because products can become numerically inconvenient, we use log likelihood:

    maximize sum_i log p_theta(x_i)

Equivalent optimization:

    minimize -sum_i log p_theta(x_i)


LATENT VARIABLES
----------------
A latent variable is an internal variable that is not directly observed.

Let:

    z = latent variable
    x = generated sample

A generator can implement:

    x = G_theta(z)

where:

    z ~ N(0, I)

Different z values produce different samples.


WHY RANDOMNESS?
---------------
If the generator always receives the same input, it will normally produce the
same output.

Random latent variables provide a source of variation:

    z_1 -> G(z_1) -> x_1
    z_2 -> G(z_2) -> x_2
    z_3 -> G(z_3) -> x_3


DISTRIBUTION MATCHING
---------------------
For this educational lesson we use a simple synthetic one-dimensional
distribution:

    0.5 N(-2, 0.35^2)
    +
    0.5 N(+2, 0.35^2)

This produces two probability modes.

The generator receives random Gaussian vectors and learns to produce samples
whose statistical properties resemble the real distribution.

This is deliberately simpler than a production generative objective.

Its purpose is to expose the fundamental loop:

    random input
          |
          v
      generator
          |
          v
    generated data
          |
          v
      comparison
          |
          v
         loss
          |
          v
       gradient
          |
          v
    parameter update


DIFFUSION CONNECTION
-------------------
The generator idea is:

    noise -> data

Diffusion introduces a structured sequence of transformations.

Forward:

    x_0 -> x_1 -> x_2 -> ... -> x_T

where increasingly more noise is added.

Eventually:

    x_T ~= Gaussian noise

The model then learns the reverse process:

    x_T -> x_{T-1} -> ... -> x_0

This is the conceptual foundation of diffusion models.


REAL-WORLD APPLICATION
----------------------
In robotics, generative models can help create additional visual observations
for training and simulation.

For example:

    robot demonstrations
            |
            v
    learned visual distribution
            |
            v
    synthetic observations
            |
            v
    additional training data

Possible applications:

    manipulation
    object recognition
    visual inspection
    simulation
    domain randomization
    rare-event augmentation


REAL-WORLD RESULT
-----------------
After this lesson you should understand:

    what p(x) means
    what sampling means
    what latent variables are
    why noise is useful
    how a neural network can generate samples
    how distribution matching works
    why diffusion starts from noise


LESSON RESULT
-------------
The conceptual progression is:

    CLASSIFICATION
          |
          v
    REPRESENTATION
          |
          v
    GENERATION
          |
          v
    DISTRIBUTION MODELING
          |
          v
       NOISE
          |
          v
      DIFFUSION


================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import math
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

NUM_REAL_SAMPLES = 6000
NUM_GENERATED_SAMPLES = 6000

BATCH_SIZE = 256
EPOCHS = 1500

LEARNING_RATE = 2e-3

LATENT_DIM = 2
HIDDEN_DIM = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DEVICE UTILITIES
# =============================================================================

def get_model_device(model):
    """
    Return the actual device of the model parameters.

    This is preferable to comparing a model device directly against a global
    torch.device object.

    Example:

        model = model.to("cuda")

        get_model_device(model)

    may return:

        cuda:0
    """

    return next(
        model.parameters()
    ).device


def same_device(
    tensor_a,
    tensor_b,
):
    """
    Return True if two tensors are on the same device.
    """

    return (
        tensor_a.device
        == tensor_b.device
    )


def print_device_info(
    model,
    name="model",
):
    """
    Print robust device information without making fragile device equality
    assumptions.
    """

    device = get_model_device(
        model
    )

    print(
        f"{name} device: {device}"
    )

    print(
        f"{name} device type: "
        f"{device.type}"
    )

    print(
        f"{name} device index: "
        f"{device.index}"
    )


# =============================================================================
# TRUE DATA DISTRIBUTION
# =============================================================================

def sample_true_distribution(
    num_samples,
    device=None,
):
    """
    Sample from:

        0.5 N(-2, 0.35^2)
        +
        0.5 N(+2, 0.35^2)
    """

    if device is None:
        device = DEVICE

    sigma = 0.35

    mode = torch.randint(
        low=0,
        high=2,
        size=(num_samples, 1),
        device=device,
    )

    left_mean = torch.tensor(
        -2.0,
        device=device,
    )

    right_mean = torch.tensor(
        2.0,
        device=device,
    )

    means = torch.where(
        mode == 0,
        left_mean,
        right_mean,
    )

    noise = torch.randn(
        num_samples,
        1,
        device=device,
    )

    samples = (
        means
        + sigma * noise
    )

    return samples


# =============================================================================
# TRUE DENSITY
# =============================================================================

def gaussian_pdf(
    x,
    mean,
    sigma,
):
    coefficient = (
        1.0
        / (
            sigma
            * math.sqrt(
                2.0 * math.pi
            )
        )
    )

    exponent = -0.5 * (
        (x - mean)
        / sigma
    ) ** 2

    return (
        coefficient
        * torch.exp(exponent)
    )


def true_density(x):
    """
    Analytical density of the synthetic two-mode distribution.
    """

    sigma = 0.35

    density_left = gaussian_pdf(
        x,
        mean=-2.0,
        sigma=sigma,
    )

    density_right = gaussian_pdf(
        x,
        mean=2.0,
        sigma=sigma,
    )

    return 0.5 * (
        density_left
        + density_right
    )


# =============================================================================
# GENERATOR
# =============================================================================

class Generator(nn.Module):
    """
    Noise -> generated sample.

    Architecture:

        z
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
        x
    """

    def __init__(
        self,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, z):
        return self.network(z)


# =============================================================================
# STATISTICAL MOMENTS
# =============================================================================

def calculate_moments(samples):

    mean = samples.mean()

    centered = (
        samples
        - mean
    )

    variance = (
        centered.pow(2).mean()
        + 1e-8
    )

    std = torch.sqrt(
        variance
    )

    skewness = (
        centered.pow(3).mean()
        / std.pow(3)
    )

    kurtosis = (
        centered.pow(4).mean()
        / std.pow(4)
    )

    return (
        mean,
        variance,
        skewness,
        kurtosis,
    )


# =============================================================================
# MOMENT-MATCHING LOSS
# =============================================================================

def moment_matching_loss(
    generated,
    real,
):
    """
    Educational distribution-matching objective.

    Matches:

        mean
        variance
        skewness
        kurtosis
    """

    (
        real_mean,
        real_variance,
        real_skewness,
        real_kurtosis,
    ) = calculate_moments(
        real
    )

    (
        generated_mean,
        generated_variance,
        generated_skewness,
        generated_kurtosis,
    ) = calculate_moments(
        generated
    )

    mean_loss = (
        generated_mean
        - real_mean
    ).pow(2)

    variance_loss = (
        generated_variance
        - real_variance
    ).pow(2)

    skewness_loss = (
        generated_skewness
        - real_skewness
    ).pow(2)

    kurtosis_loss = (
        generated_kurtosis
        - real_kurtosis
    ).pow(2)

    return (
        mean_loss
        + variance_loss
        + skewness_loss
        + kurtosis_loss
    )


# =============================================================================
# DATA BATCH
# =============================================================================

def sample_real_batch(
    batch_size,
    device=None,
):

    if device is None:
        device = DEVICE

    return sample_true_distribution(
        batch_size,
        device=device,
    )


def sample_latent(
    batch_size,
    latent_dim=LATENT_DIM,
    device=None,
):
    """
    Sample z ~ N(0, I).

    Device is explicit so that z can always be placed on the same device as
    the generator.
    """

    if device is None:
        device = DEVICE

    return torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_generator(
    generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=None,
):
    """
    Train the generator.

    The model itself is moved to the requested device.
    """

    if device is None:
        device = get_model_device(
            generator
        )

    generator = generator.to(
        device
    )

    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    for epoch in range(
        epochs
    ):

        real_samples = sample_real_batch(
            batch_size=batch_size,
            device=device,
        )

        latent = sample_latent(
            batch_size=batch_size,
            device=device,
        )

        generated_samples = generator(
            latent
        )

        loss = moment_matching_loss(
            generated=generated_samples,
            real=real_samples,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        history.append(
            loss.item()
        )

        if (
            epoch == 0
            or (epoch + 1) % 100 == 0
        ):
            print(
                f"Epoch "
                f"{epoch + 1:04d}/"
                f"{epochs} | "
                f"loss = "
                f"{loss.item():.6f}"
            )

    return history


# =============================================================================
# GENERATION
# =============================================================================

@torch.no_grad()
def generate_samples(
    generator,
    num_samples,
    device=None,
):
    """
    Generate samples using the generator's actual device unless an explicit
    device is supplied.
    """

    generator.eval()

    if device is None:
        device = get_model_device(
            generator
        )

    latent = sample_latent(
        batch_size=num_samples,
        device=device,
    )

    generated = generator(
        latent
    )

    return generated


# =============================================================================
# HISTOGRAM DENSITY
# =============================================================================

def histogram_density(
    samples,
    bins=100,
    value_range=(-4.0, 4.0),
):

    samples_np = (
        samples.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    density, edges = np.histogram(
        samples_np,
        bins=bins,
        range=value_range,
        density=True,
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    return centers, density


# =============================================================================
# VISUALIZATION — DISTRIBUTION
# =============================================================================

def plot_distribution_comparison(
    real_samples,
    generated_samples,
):

    real_x, real_density = (
        histogram_density(
            real_samples
        )
    )

    generated_x, generated_density = (
        histogram_density(
            generated_samples
        )
    )

    x = torch.linspace(
        -4.0,
        4.0,
        1000,
    )

    density = true_density(
        x
    ).numpy()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        x.numpy(),
        density,
        label="true density",
        linewidth=3,
    )

    ax.plot(
        real_x,
        real_density,
        label="real samples",
        linewidth=2,
    )

    ax.plot(
        generated_x,
        generated_density,
        label="generated samples",
        linewidth=2,
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "Probability density"
    )

    ax.set_title(
        "True Distribution vs Generated Distribution"
    )

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZATION — LATENT SPACE
# =============================================================================

@torch.no_grad()
def plot_latent_to_output(
    generator,
):

    generator.eval()

    device = get_model_device(
        generator
    )

    grid_size = 25

    z1_values = torch.linspace(
        -3,
        3,
        grid_size,
        device=device,
    )

    z2_values = torch.linspace(
        -3,
        3,
        grid_size,
        device=device,
    )

    outputs = []

    coordinates_z1 = []
    coordinates_z2 = []

    for z1 in z1_values:

        for z2 in z2_values:

            z = torch.stack(
                [
                    z1,
                    z2,
                ]
            ).reshape(
                1,
                2,
            )

            output = generator(
                z
            ).item()

            outputs.append(
                output
            )

            coordinates_z1.append(
                z1.item()
            )

            coordinates_z2.append(
                z2.item()
            )

    outputs = np.asarray(
        outputs
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    scatter = ax.scatter(
        coordinates_z1,
        coordinates_z2,
        c=outputs,
        s=35,
    )

    ax.set_xlabel(
        "Latent z1"
    )

    ax.set_ylabel(
        "Latent z2"
    )

    ax.set_title(
        "Latent Space -> Generated Sample"
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Generated x",
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZATION — TRAINING
# =============================================================================

def plot_training_loss(
    history,
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history
    )

    ax.set_xlabel(
        "Training iteration"
    )

    ax.set_ylabel(
        "Moment matching loss"
    )

    ax.set_title(
        "Generative Model Training"
    )

    ax.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# STATISTICAL REPORT
# =============================================================================

def print_distribution_statistics(
    name,
    samples,
):

    (
        mean,
        variance,
        skewness,
        kurtosis,
    ) = calculate_moments(
        samples
    )

    print()
    print(name)
    print("-" * 60)

    print(
        f"Mean     : "
        f"{mean.item(): .6f}"
    )

    print(
        f"Variance : "
        f"{variance.item(): .6f}"
    )

    print(
        f"Skewness : "
        f"{skewness.item(): .6f}"
    )

    print(
        f"Kurtosis : "
        f"{kurtosis.item(): .6f}"
    )


# =============================================================================
# TEST 1 — TRUE DISTRIBUTION
# =============================================================================

def test_true_distribution():

    print()
    print(
        "TEST 1 — True Distribution"
    )

    test_device = torch.device(
        "cpu"
    )

    samples = sample_true_distribution(
        10000,
        device=test_device,
    )

    assert samples.shape == (
        10000,
        1,
    )

    assert torch.isfinite(
        samples
    ).all()

    mean = samples.mean().item()

    assert abs(mean) < 0.10

    print(
        f"PASS: generated "
        f"{len(samples)} valid samples "
        f"with mean={mean:.4f}"
    )


# =============================================================================
# TEST 2 — TRUE DENSITY
# =============================================================================

def test_true_density():

    print()
    print(
        "TEST 2 — Analytical Density"
    )

    x = torch.linspace(
        -8,
        8,
        10000,
    )

    density = true_density(
        x
    )

    assert torch.isfinite(
        density
    ).all()

    assert torch.all(
        density >= 0
    )

    integral = torch.trapezoid(
        density,
        x,
    ).item()

    assert abs(
        integral - 1.0
    ) < 0.01

    print(
        f"PASS: density integral = "
        f"{integral:.6f}"
    )


# =============================================================================
# TEST 3 — GENERATOR SHAPE
# =============================================================================

def test_generator_shape():

    print()
    print(
        "TEST 3 — Generator Shape"
    )

    test_device = torch.device(
        "cpu"
    )

    generator = Generator().to(
        test_device
    )

    z = torch.randn(
        32,
        LATENT_DIM,
        device=test_device,
    )

    output = generator(
        z
    )

    assert output.shape == (
        32,
        1,
    )

    assert same_device(
        z,
        next(
            generator.parameters()
        ),
    )

    assert torch.isfinite(
        output
    ).all()

    print(
        f"PASS: z "
        f"{tuple(z.shape)} -> "
        f"x "
        f"{tuple(output.shape)}"
    )


# =============================================================================
# TEST 4 — GENERATOR GRADIENT FLOW
# =============================================================================

def test_generator_gradient_flow():

    print()
    print(
        "TEST 4 — Generator Gradient Flow"
    )

    # IMPORTANT:
    # This test deliberately uses CPU.
    #
    # It therefore does not depend on the machine having CUDA.
    #
    # Every tensor and every model parameter is explicitly placed on CPU.

    test_device = torch.device(
        "cpu"
    )

    generator = Generator().to(
        test_device
    )

    real = sample_real_batch(
        batch_size=64,
        device=test_device,
    )

    z = sample_latent(
        batch_size=64,
        device=test_device,
    )

    model_parameter = next(
        generator.parameters()
    )

    assert (
        z.device
        == model_parameter.device
    )

    assert (
        real.device
        == model_parameter.device
    )

    generated = generator(
        z
    )

    assert (
        generated.device
        == model_parameter.device
    )

    loss = moment_matching_loss(
        generated,
        real,
    )

    assert torch.isfinite(
        loss
    )

    loss.backward()

    total_gradient_values = 0
    nonzero_gradient_values = 0

    for parameter in (
        generator.parameters()
    ):

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all()

            total_gradient_values += (
                parameter.grad.numel()
            )

            nonzero_gradient_values += (
                parameter.grad.abs()
                .gt(1e-12)
                .sum()
                .item()
            )

    assert (
        total_gradient_values > 0
    )

    assert (
        nonzero_gradient_values > 0
    )

    print(
        "PASS: generator receives "
        "non-zero gradients."
    )


# =============================================================================
# TEST 5 — MOMENT LOSS
# =============================================================================

def test_moment_loss():

    print()
    print(
        "TEST 5 — Moment Matching"
    )

    test_device = torch.device(
        "cpu"
    )

    samples = sample_true_distribution(
        1000,
        device=test_device,
    )

    loss = moment_matching_loss(
        samples,
        samples.clone(),
    )

    assert torch.isfinite(
        loss
    )

    assert loss.item() < 1e-12

    print(
        f"PASS: identical distributions "
        f"produce loss={loss.item():.3e}"
    )


# =============================================================================
# TEST 6 — GENERATED SAMPLE VALIDITY
# =============================================================================

def test_generated_samples(
    generator,
):

    print()
    print(
        "TEST 6 — Generated Samples"
    )

    generator_device = (
        get_model_device(
            generator
        )
    )

    samples = generate_samples(
        generator,
        2000,
        device=generator_device,
    )

    assert samples.shape == (
        2000,
        1,
    )

    assert torch.isfinite(
        samples
    ).all()

    assert (
        samples.abs().mean()
        < 5.0
    )

    print(
        "PASS: generated samples "
        "have valid shape and finite values."
    )


# =============================================================================
# TEST 7 — MODEL DEVICE CONSISTENCY
# =============================================================================

def test_model_device_consistency(
    generator,
):

    print()
    print(
        "TEST 7 — Model Device Consistency"
    )

    model_device = get_model_device(
        generator
    )

    z = sample_latent(
        batch_size=8,
        device=model_device,
    )

    parameter = next(
        generator.parameters()
    )

    assert (
        z.device
        == parameter.device
    )

    generated = generator(
        z
    )

    assert (
        generated.device
        == parameter.device
    )

    print(
        "PASS: model, latent input, "
        "and generated output use the same device."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    print("=" * 80)
    print(
        "LESSON 14 — GENERATIVE MODELING FUNDAMENTALS"
    )
    print("=" * 80)

    print()
    print(
        f"Global preferred device: "
        f"{DEVICE}"
    )

    # =========================================================================
    # UNIT TESTS
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "RUNNING UNIT TESTS"
    )
    print("=" * 80)

    test_true_distribution()

    test_true_density()

    test_generator_shape()

    test_generator_gradient_flow()

    test_moment_loss()

    print()
    print("=" * 80)
    print(
        "INITIAL TESTS PASSED"
    )
    print("=" * 80)

    # =========================================================================
    # REAL DATA
    # =========================================================================

    real_samples = sample_true_distribution(
        NUM_REAL_SAMPLES,
        device=DEVICE,
    )

    print_distribution_statistics(
        "REAL DATA DISTRIBUTION",
        real_samples,
    )

    # =========================================================================
    # TRUE DISTRIBUTION VISUALIZATION
    # =========================================================================

    x = torch.linspace(
        -4,
        4,
        1000,
    )

    density = true_density(
        x
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        x.numpy(),
        density.numpy(),
        linewidth=3,
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "Probability density"
    )

    ax.set_title(
        "True Data Distribution"
    )

    ax.grid(True)

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # CREATE GENERATOR
    # =========================================================================

    generator = Generator().to(
        DEVICE
    )

    print()
    print("=" * 80)
    print(
        "GENERATOR"
    )
    print("=" * 80)

    print(generator)

    parameter_count = sum(
        p.numel()
        for p in generator.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # =========================================================================
    # MODEL DEVICE
    # =========================================================================

    model_device = get_model_device(
        generator
    )

    print()
    print_device_info(
        generator,
        name="Generator",
    )

    # =========================================================================
    # IMPORTANT DEVICE NOTE
    # =========================================================================
    #
    # Do NOT use:
    #
    #     assert model_device == DEVICE
    #
    # as a general correctness test.
    #
    # torch.device objects can differ in representation, for example:
    #
    #     cuda
    #
    # versus:
    #
    #     cuda:0
    #
    # while referring to the same usable CUDA device.
    #
    # Instead, all actual computation below follows model_device.

    assert model_device.type == DEVICE.type

    if model_device.type == "cuda":

        assert torch.cuda.is_available()

        assert model_device.index in (
            None,
            torch.cuda.current_device(),
        )

    print(
        "PASS: generator is on a valid "
        "device for training."
    )

    # =========================================================================
    # DEVICE CONSISTENCY TEST
    # =========================================================================

    test_model_device_consistency(
        generator
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "TRAINING GENERATIVE MODEL"
    )
    print("=" * 80)

    history = train_generator(
        generator=generator,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=model_device,
    )

    # =========================================================================
    # GENERATE
    # =========================================================================

    generated_samples = generate_samples(
        generator,
        NUM_GENERATED_SAMPLES,
        device=model_device,
    )

    print_distribution_statistics(
        "GENERATED DISTRIBUTION",
        generated_samples,
    )

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    plot_distribution_comparison(
        real_samples,
        generated_samples,
    )

    plot_training_loss(
        history
    )

    plot_latent_to_output(
        generator
    )

    # =========================================================================
    # POST-TRAINING TEST
    # =========================================================================

    test_generated_samples(
        generator
    )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    (
        real_mean,
        real_variance,
        real_skewness,
        real_kurtosis,
    ) = calculate_moments(
        real_samples
    )

    (
        generated_mean,
        generated_variance,
        generated_skewness,
        generated_kurtosis,
    ) = calculate_moments(
        generated_samples
    )

    print()
    print("=" * 80)
    print(
        "DISTRIBUTION MATCHING RESULT"
    )
    print("=" * 80)

    print(
        f"Real mean       : "
        f"{real_mean.item(): .4f}"
    )

    print(
        f"Generated mean  : "
        f"{generated_mean.item(): .4f}"
    )

    print()

    print(
        f"Real variance       : "
        f"{real_variance.item(): .4f}"
    )

    print(
        f"Generated variance  : "
        f"{generated_variance.item(): .4f}"
    )

    print()

    print(
        f"Real skewness       : "
        f"{real_skewness.item(): .4f}"
    )

    print(
        f"Generated skewness  : "
        f"{generated_skewness.item(): .4f}"
    )

    print()

    print(
        f"Real kurtosis       : "
        f"{real_kurtosis.item(): .4f}"
    )

    print(
        f"Generated kurtosis  : "
        f"{generated_kurtosis.item(): .4f}"
    )

    # =========================================================================
    # FINAL NUMERICAL VERIFICATION
    # =========================================================================

    assert len(history) == EPOCHS

    assert np.isfinite(
        np.asarray(history)
    ).all()

    assert np.isfinite(
        generated_samples.detach()
        .cpu()
        .numpy()
    ).all()

    assert (
        abs(
            generated_mean.item()
        )
        < 0.35
    )

    assert (
        generated_variance.item()
        > 1.0
    )

    # =========================================================================
    # REAL-WORLD APPLICATION
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "REAL-WORLD APPLICATION"
    )
    print("=" * 80)

    print(
        """
Consider a robot-learning dataset.

Suppose a robot has collected:

    500 demonstrations

and the visual observations contain limited variation.

A generative model can potentially learn:

    p(image)

and produce additional plausible observations.

The same general concept can be used for:

    synthetic robotics data
    visual inspection
    simulation
    rare-event augmentation
    object appearance generation
    image restoration
    image synthesis

The core loop is:

    DATA
      |
      v
    DISTRIBUTION
      |
      v
    GENERATIVE MODEL
      |
      v
    NEW SAMPLES
"""
    )

    # =========================================================================
    # DIFFUSION CONNECTION
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "CONNECTION TO DIFFUSION"
    )
    print("=" * 80)

    print(
        """
This lesson used:

    random latent z
          |
          v
       Generator
          |
          v
       sample x


Diffusion will introduce a structured sequence.

FORWARD PROCESS:

    clean image x_0
          |
          v
    slightly noisy x_1
          |
          v
    more noisy x_2
          |
          v
          ...
          |
          v
    almost pure noise x_T


REVERSE PROCESS:

    x_T
     |
     v
    x_{T-1}
     |
     v
    x_{T-2}
     |
     v
    ...
     |
     v
    x_0


This is the central bridge to diffusion.

The next lesson studies the probability mathematics needed to derive that
process.
"""
    )

    # =========================================================================
    # LESSON RESULT
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "LESSON 14 RESULT"
    )
    print("=" * 80)

    print(
        """
SUCCESS.

You have moved from discriminative vision to generative modeling.

You now have a concrete understanding of:

    p(x)
    sampling
    latent variables
    learned distributions
    distribution matching
    generative neural networks
    noise as a source of randomness
    the conceptual basis of diffusion

Mental model:

    SIMPLE RANDOM SOURCE
             |
             v
      LEARNED TRANSFORM
             |
             v
       DATA DISTRIBUTION
             |
             v
        NEW SAMPLE


NEXT:

    LESSON 15 — PROBABILITY DISTRIBUTIONS FOR IMAGE GENERATION

We will study:

    Gaussian distributions
    multivariate Gaussian noise
    covariance
    conditional distributions
    reparameterization
    KL divergence
    entropy
    score intuition
    why Gaussian noise is central to diffusion

Then:

    LESSON 16 — FORWARD DIFFUSION PROCESS
"""
    )

    print()
    print("=" * 80)
    print(
        "LESSON 14 COMPLETE"
    )
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()