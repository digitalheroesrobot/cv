"""
================================================================================
LESSON 15 — PROBABILITY DISTRIBUTIONS FOR IMAGE GENERATION
================================================================================

PURPOSE
-------
Build the probability foundations required for diffusion models.

Lesson 14 introduced the idea:

    noise -> generator -> data

Now we need to understand the probability distributions used to construct
diffusion mathematically.

This lesson focuses heavily on Gaussian distributions because Gaussian noise
is the fundamental building block of the diffusion process.


WHAT YOU WILL LEARN
-------------------
1. Probability density functions.
2. Gaussian / Normal distributions.
3. Mean and variance.
4. Standard deviation.
5. Sampling from Gaussian distributions.
6. Multivariate Gaussian distributions.
7. Covariance.
8. Independent vs correlated dimensions.
9. Standard Gaussian noise.
10. Reparameterization.
11. Entropy.
12. KL divergence.
13. Log probability.
14. Score intuition.
15. Why Gaussian noise is central to diffusion.
16. How these concepts connect to image tensors.


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
    Lesson 12  Generative modeling foundations
    Lesson 13  Vision Transformer
    Lesson 14  Generative modeling fundamentals
    Lesson 15  Probability distributions

NEXT:

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


================================================================================
REAL-WORLD PROBLEM
================================================================================

A camera does not produce perfectly deterministic observations.

For example, a robot camera observing the same object may produce different
pixel values because of:

    sensor noise
    lighting
    exposure
    reflections
    motion
    compression
    environmental changes

Therefore, instead of thinking only about one exact image:

    x

we can think probabilistically:

    x ~ p(x)


For image generation we ultimately want to model distributions over extremely
high-dimensional data.

For an RGB image:

    x in R^(H x W x 3)

For example:

    256 x 256 x 3 = 196,608 values

Diffusion models construct a tractable probabilistic path between:

    real image distribution

and

    Gaussian noise distribution.


================================================================================
1. PROBABILITY DENSITY
================================================================================

For a continuous variable x, a probability density function is written:

    p(x)

Important:

    p(x) is a density, not directly the probability of one exact point.

Probability over an interval is:

    P(a <= x <= b)
        =
    integral_a^b p(x) dx

A valid probability density satisfies:

    p(x) >= 0

and:

    integral p(x) dx = 1


================================================================================
2. GAUSSIAN DISTRIBUTION
================================================================================

The Gaussian distribution is:

                    1
    p(x) = ---------------------- exp(
            sqrt(2*pi)*sigma

                 -(x-mu)^2
                 -----------
                   2 sigma^2
    )


where:

    mu     = mean
    sigma  = standard deviation
    sigma² = variance


Notation:

    x ~ N(mu, sigma²)


The standard Gaussian is:

    N(0, 1)

Therefore:

    mu = 0
    sigma = 1


This distribution is especially important because diffusion starts from
Gaussian noise.


================================================================================
3. MEAN
================================================================================

The expected value of a random variable is:

    E[x]

For a Gaussian:

    E[x] = mu


================================================================================
4. VARIANCE
================================================================================

Variance measures spread:

    Var(x)
        =
    E[(x - mu)^2]

For a Gaussian:

    Var(x) = sigma²


================================================================================
5. STANDARD DEVIATION
================================================================================

Standard deviation is:

    sigma = sqrt(variance)

A useful interpretation:

    small sigma
        -> concentrated distribution

    large sigma
        -> wide distribution


================================================================================
6. MULTIVARIATE GAUSSIAN
================================================================================

For vectors:

    x in R^D

we can use a multivariate Gaussian:

    x ~ N(mu, Sigma)

where:

    mu     = mean vector
    Sigma  = covariance matrix


The covariance matrix describes relationships between dimensions.

For two variables:

    Sigma =
        [ sigma_x²       cov_xy ]
        [ cov_xy         sigma_y² ]


If:

    cov_xy = 0

the dimensions are uncorrelated.

If:

    cov_xy > 0

they tend to increase together.

If:

    cov_xy < 0

one tends to increase while the other decreases.


================================================================================
7. STANDARD MULTIVARIATE GAUSSIAN
================================================================================

The most important special case for diffusion is:

    x ~ N(0, I)

where:

    0 = zero vector
    I = identity matrix


This means:

    each dimension has mean 0
    each dimension has variance 1
    dimensions are uncorrelated


For an image tensor, we can generate Gaussian noise using:

    epsilon ~ N(0, I)


================================================================================
8. SAMPLING
================================================================================

Sampling means drawing random values from a probability distribution.

For example:

    epsilon = torch.randn(...)

produces approximately:

    epsilon ~ N(0, I)


The distribution is random, but its statistical behavior is predictable.


================================================================================
9. REPARAMETERIZATION
================================================================================

Suppose:

    z ~ N(mu, sigma²)

We can express this as:

    epsilon ~ N(0, 1)

and:

    z = mu + sigma * epsilon


This is the reparameterization trick.

For vectors:

    z = mu + sigma * epsilon

For diagonal covariance:

    epsilon ~ N(0, I)

    z = mu + sigma * epsilon


This is important in VAEs because it allows gradients to flow through the
sampling operation.


================================================================================
10. ENTROPY
================================================================================

Entropy measures uncertainty.

For a one-dimensional Gaussian:

    H(X)
      =
    1/2 log(2*pi*e*sigma²)

Higher variance means higher entropy.

For the standard Gaussian:

    sigma² = 1

so:

    H(N(0,1))
      =
    1/2 log(2*pi*e)


Entropy becomes important when reasoning about how much information is
contained in a distribution.


================================================================================
11. KL DIVERGENCE
================================================================================

KL divergence measures how different one probability distribution is from
another.

For distributions P and Q:

             P(x)
    KL(P||Q) = E_P [ log ----- ]
             Q(x)


Properties:

    KL(P||Q) >= 0

and:

    KL(P||Q) = 0

only when the distributions are identical almost everywhere.


Important:

    KL(P||Q) is not generally symmetric.

Therefore:

    KL(P||Q) != KL(Q||P)


For two one-dimensional Gaussians:

    P = N(mu1, sigma1²)
    Q = N(mu2, sigma2²)

the KL divergence has a closed form.

This gives us an excellent numerical verification target.


================================================================================
12. LOG PROBABILITY
================================================================================

Instead of working directly with:

    p(x)

we often work with:

    log p(x)

Reasons include:

    numerical stability
    products become sums
    easier optimization


For independent observations:

    log product_i p(x_i)

becomes:

    sum_i log p(x_i)


This is one reason likelihood-based generative modeling is mathematically
convenient.


================================================================================
13. SCORE INTUITION
================================================================================

A very important quantity for diffusion is the score:

                 d
    score(x) = -------- log p(x)
                 dx


For a Gaussian:

    p(x) = N(mu, sigma²)

the score is:

              mu - x
    score = -----------
                sigma²


This vector points toward regions of higher probability density.

For a standard Gaussian:

    mu = 0
    sigma² = 1

therefore:

    score(x) = -x


This is a beautiful result:

    positive x -> score points left
    negative x -> score points right
    x near zero -> score near zero


The score therefore tells us the direction toward higher probability density.


================================================================================
14. WHY THIS MATTERS FOR DIFFUSION
================================================================================

Diffusion gradually transforms data into Gaussian noise.

At a high level:

    x_0
     |
     | add a little Gaussian noise
     v
    x_1
     |
     | add more Gaussian noise
     v
    x_2
     |
     v
    ...
     |
     v
    x_T ~ N(0, I)


The reverse process attempts to recover:

    x_T -> x_0


The model therefore needs to understand how probability changes through this
noise process.

The concepts introduced here:

    Gaussian
    covariance
    noise
    log probability
    KL divergence
    score

will appear repeatedly in diffusion mathematics.


================================================================================
15. IMAGE CONNECTION
================================================================================

A real RGB image can be represented as:

    x in R^(H x W x 3)

Gaussian noise has the same shape:

    epsilon in R^(H x W x 3)

Therefore we can write:

    epsilon ~ N(0, I)

and combine image and noise:

    x_noisy = alpha * x
              +
              sigma * epsilon


This is the core mathematical pattern behind diffusion.

Lesson 16 will turn this idea into an explicit forward diffusion process.


================================================================================
EXPERIMENT
================================================================================

This program performs several experiments:

    Experiment 1
        visualize Gaussian distributions with different variances

    Experiment 2
        verify sampled Gaussian mean and variance

    Experiment 3
        visualize a 2D multivariate Gaussian

    Experiment 4
        visualize correlated Gaussian samples

    Experiment 5
        verify the reparameterization trick

    Experiment 6
        numerically verify Gaussian entropy

    Experiment 7
        verify Gaussian KL divergence

    Experiment 8
        visualize the score of a Gaussian

    Experiment 9
        create synthetic image-like Gaussian noise

    Experiment 10
        demonstrate image + Gaussian noise


================================================================================
REAL-WORLD APPLICATION
================================================================================

Robotics example:

A robot camera observes a cube.

The ideal image is:

    x

Real sensors introduce variation:

    x_observed = x + sensor_noise

A probabilistic model represents this uncertainty using:

    epsilon ~ N(0, I)

Then:

    x_noisy = alpha*x + sigma*epsilon


This is useful for:

    sensor simulation
    robustness testing
    synthetic training data
    visual domain randomization
    image denoising
    diffusion-based image generation


================================================================================
REAL-WORLD RESULT
================================================================================

After this lesson you should be able to explain:

    Why Gaussian noise is useful.
    What N(0,I) means.
    What covariance represents.
    Why reparameterization is useful.
    What KL divergence measures.
    What entropy measures.
    What the score means.
    Why diffusion naturally uses Gaussian noise.


================================================================================
LESSON RESULT
================================================================================

The key mathematical object to remember is:

    epsilon ~ N(0, I)


and the key noisy-data equation is:

    x_noisy
        =
    alpha * x
        +
    sigma * epsilon


This equation is the doorway into diffusion.

NEXT:

    LESSON 16 — FORWARD DIFFUSION PROCESS

There we will construct:

    x_0 -> x_1 -> x_2 -> ... -> x_T

mathematically and visually.

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


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_SAMPLES = 100_000

IMAGE_SIZE = 32


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
# GAUSSIAN PDF
# =============================================================================

def gaussian_pdf(
    x,
    mean=0.0,
    variance=1.0,
):
    """
    Gaussian probability density:

                        1
        p(x) = ------------------- exp(
                sqrt(2*pi*variance)

                   -(x-mu)^2
                   ----------
                   2*variance
        )
    """

    x = torch.as_tensor(
        x,
        dtype=torch.float64,
    )

    mean = torch.as_tensor(
        mean,
        dtype=torch.float64,
        device=x.device,
    )

    variance = torch.as_tensor(
        variance,
        dtype=torch.float64,
        device=x.device,
    )

    coefficient = 1.0 / torch.sqrt(
        2.0
        * math.pi
        * variance
    )

    exponent = -(
        (x - mean) ** 2
    ) / (
        2.0 * variance
    )

    return coefficient * torch.exp(
        exponent
    )


# =============================================================================
# GAUSSIAN LOG PDF
# =============================================================================

def gaussian_log_pdf(
    x,
    mean=0.0,
    variance=1.0,
):
    """
    Log probability density of a Gaussian.

    log p(x)
        =
    -0.5 * [
        log(2*pi*variance)
        +
        (x-mean)^2 / variance
    ]
    """

    x = torch.as_tensor(
        x,
        dtype=torch.float64,
    )

    return -0.5 * (
        math.log(
            2.0
            * math.pi
            * variance
        )
        +
        (
            (x - mean) ** 2
            / variance
        )
    )


# =============================================================================
# SAMPLE GAUSSIAN
# =============================================================================

def sample_gaussian(
    num_samples,
    mean=0.0,
    variance=1.0,
    device=DEVICE,
):

    epsilon = torch.randn(
        num_samples,
        device=device,
        dtype=torch.float64,
    )

    return (
        mean
        + math.sqrt(variance)
        * epsilon
    )


# =============================================================================
# MULTIVARIATE GAUSSIAN SAMPLING
# =============================================================================

def sample_multivariate_gaussian(
    mean,
    covariance,
    num_samples,
    device=DEVICE,
):

    mean = torch.as_tensor(
        mean,
        dtype=torch.float64,
        device=device,
    )

    covariance = torch.as_tensor(
        covariance,
        dtype=torch.float64,
        device=device,
    )

    dimension = mean.numel()

    standard_normal = torch.randn(
        num_samples,
        dimension,
        dtype=torch.float64,
        device=device,
    )

    # Cholesky decomposition:
    #
    #     Sigma = L L^T
    #
    # If:
    #
    #     epsilon ~ N(0,I)
    #
    # then:
    #
    #     x = mu + L epsilon
    #
    # has covariance Sigma.

    L = torch.linalg.cholesky(
        covariance
    )

    samples = (
        mean
        + standard_normal @ L.T
    )

    return samples


# =============================================================================
# EMPIRICAL MOMENTS
# =============================================================================

def empirical_mean(x):

    return x.mean(
        dim=0
    )


def empirical_variance(x):

    return x.var(
        dim=0,
        unbiased=True,
    )


# =============================================================================
# COVARIANCE MATRIX
# =============================================================================

def empirical_covariance(x):

    centered = (
        x
        - x.mean(
            dim=0,
            keepdim=True,
        )
    )

    n = x.shape[0]

    return (
        centered.T
        @ centered
        / (n - 1)
    )


# =============================================================================
# REPARAMETERIZATION
# =============================================================================

def reparameterize(
    mean,
    std,
    epsilon=None,
):

    if epsilon is None:

        epsilon = torch.randn_like(
            mean
        )

    return (
        mean
        + std * epsilon
    )


# =============================================================================
# GAUSSIAN ENTROPY
# =============================================================================

def gaussian_entropy(
    variance,
):
    """
    Entropy of:

        N(mu, variance)

    H = 1/2 log(2*pi*e*variance)
    """

    return 0.5 * math.log(
        2.0
        * math.pi
        * math.e
        * variance
    )


# =============================================================================
# GAUSSIAN KL DIVERGENCE
# =============================================================================

def gaussian_kl_divergence(
    mean_p,
    variance_p,
    mean_q,
    variance_q,
):
    """
    KL(P || Q) for:

        P = N(mean_p, variance_p)
        Q = N(mean_q, variance_q)

    KL(P||Q)
        =
    1/2 [
        log(variance_q / variance_p)
        +
        (variance_p
         + (mean_p-mean_q)^2)
        / variance_q
        - 1
    ]
    """

    return 0.5 * (
        math.log(
            variance_q
            / variance_p
        )
        +
        (
            variance_p
            + (
                mean_p
                - mean_q
            ) ** 2
        )
        / variance_q
        - 1.0
    )


# =============================================================================
# MONTE CARLO KL ESTIMATION
# =============================================================================

def monte_carlo_gaussian_kl(
    mean_p,
    variance_p,
    mean_q,
    variance_q,
    num_samples=100_000,
):

    samples = sample_gaussian(
        num_samples,
        mean=mean_p,
        variance=variance_p,
        device=DEVICE,
    )

    log_p = gaussian_log_pdf(
        samples,
        mean=mean_p,
        variance=variance_p,
    )

    log_q = gaussian_log_pdf(
        samples,
        mean=mean_q,
        variance=variance_q,
    )

    return (
        log_p - log_q
    ).mean().item()


# =============================================================================
# GAUSSIAN SCORE
# =============================================================================

def gaussian_score(
    x,
    mean=0.0,
    variance=1.0,
):
    """
    Score:

        d/dx log p(x)

    For Gaussian:

        score = -(x - mean) / variance

    equivalently:

        score = (mean - x) / variance
    """

    return (
        mean - x
    ) / variance


# =============================================================================
# TEST 1 — PDF NORMALIZATION
# =============================================================================

def test_gaussian_pdf_normalization():

    print()
    print(
        "TEST 1 — Gaussian PDF Normalization"
    )

    x = torch.linspace(
        -10.0,
        10.0,
        200_000,
        dtype=torch.float64,
    )

    pdf = gaussian_pdf(
        x,
        mean=0.0,
        variance=1.0,
    )

    integral = torch.trapezoid(
        pdf,
        x,
    ).item()

    assert abs(
        integral - 1.0
    ) < 1e-5

    assert torch.all(
        pdf >= 0
    )

    print(
        f"PASS: integral = "
        f"{integral:.8f}"
    )


# =============================================================================
# TEST 2 — GAUSSIAN SAMPLING
# =============================================================================

def test_gaussian_sampling():

    print()
    print(
        "TEST 2 — Gaussian Sampling"
    )

    mean = 2.5
    variance = 1.7

    samples = sample_gaussian(
        NUM_SAMPLES,
        mean=mean,
        variance=variance,
        device=DEVICE,
    )

    empirical_mean = (
        samples.mean().item()
    )

    empirical_variance = (
        samples.var(
            unbiased=True
        ).item()
    )

    assert abs(
        empirical_mean - mean
    ) < 0.02

    assert abs(
        empirical_variance
        - variance
    ) < 0.03

    print(
        f"PASS: empirical mean = "
        f"{empirical_mean:.5f}"
    )

    print(
        f"PASS: empirical variance = "
        f"{empirical_variance:.5f}"
    )


# =============================================================================
# TEST 3 — MULTIVARIATE GAUSSIAN
# =============================================================================

def test_multivariate_gaussian():

    print()
    print(
        "TEST 3 — Multivariate Gaussian"
    )

    mean = torch.tensor(
        [1.0, -2.0],
        dtype=torch.float64,
    )

    covariance = torch.tensor(
        [
            [2.0, 0.7],
            [0.7, 1.5],
        ],
        dtype=torch.float64,
    )

    samples = sample_multivariate_gaussian(
        mean,
        covariance,
        100_000,
        device=DEVICE,
    )

    empirical_mean = empirical_mean_fn = (
        samples.mean(
            dim=0
        )
    )

    empirical_cov = (
        empirical_covariance(
            samples
        )
    )

    assert torch.allclose(
        empirical_mean,
        mean.to(DEVICE),
        atol=0.02,
    )

    assert torch.allclose(
        empirical_cov,
        covariance.to(DEVICE),
        atol=0.03,
    )

    print(
        "PASS: empirical mean "
        "matches target."
    )

    print(
        "PASS: empirical covariance "
        "matches target."
    )


# =============================================================================
# TEST 4 — STANDARD GAUSSIAN
# =============================================================================

def test_standard_gaussian():

    print()
    print(
        "TEST 4 — Standard Gaussian N(0,I)"
    )

    samples = torch.randn(
        100_000,
        8,
        device=DEVICE,
        dtype=torch.float64,
    )

    means = samples.mean(
        dim=0
    )

    variances = samples.var(
        dim=0,
        unbiased=True,
    )

    assert torch.all(
        means.abs() < 0.02
    )

    assert torch.all(
        (variances - 1.0).abs()
        < 0.03
    )

    print(
        "PASS: sampled standard "
        "Gaussian has approximately "
        "zero mean and unit variance."
    )


# =============================================================================
# TEST 5 — REPARAMETERIZATION
# =============================================================================

def test_reparameterization():

    print()
    print(
        "TEST 5 — Reparameterization"
    )

    mean = torch.tensor(
        [2.0, -1.0],
        dtype=torch.float64,
        device=DEVICE,
    )

    std = torch.tensor(
        [0.5, 2.0],
        dtype=torch.float64,
        device=DEVICE,
    )

    epsilon = torch.tensor(
        [1.0, -2.0],
        dtype=torch.float64,
        device=DEVICE,
    )

    z = reparameterize(
        mean,
        std,
        epsilon,
    )

    expected = torch.tensor(
        [2.5, -5.0],
        dtype=torch.float64,
        device=DEVICE,
    )

    assert torch.allclose(
        z,
        expected,
    )

    print(
        "PASS: z = mean + std * epsilon"
    )


# =============================================================================
# TEST 6 — ENTROPY
# =============================================================================

def test_entropy():

    print()
    print(
        "TEST 6 — Gaussian Entropy"
    )

    entropy_1 = gaussian_entropy(
        1.0
    )

    entropy_4 = gaussian_entropy(
        4.0
    )

    expected_1 = 0.5 * math.log(
        2.0
        * math.pi
        * math.e
    )

    assert abs(
        entropy_1
        - expected_1
    ) < 1e-12

    assert entropy_4 > entropy_1

    print(
        f"PASS: H(N(0,1)) = "
        f"{entropy_1:.6f}"
    )

    print(
        "PASS: larger variance "
        "produces larger entropy."
    )


# =============================================================================
# TEST 7 — KL DIVERGENCE
# =============================================================================

def test_kl_divergence():

    print()
    print(
        "TEST 7 — Gaussian KL Divergence"
    )

    p = (
        0.0,
        1.0,
    )

    q = (
        0.0,
        1.0,
    )

    kl_identical = (
        gaussian_kl_divergence(
            p[0],
            p[1],
            q[0],
            q[1],
        )
    )

    assert abs(
        kl_identical
    ) < 1e-12

    kl_pq = (
        gaussian_kl_divergence(
            0.0,
            1.0,
            1.0,
            2.0,
        )
    )

    kl_qp = (
        gaussian_kl_divergence(
            1.0,
            2.0,
            0.0,
            1.0,
        )
    )

    assert kl_pq >= 0.0
    assert kl_qp >= 0.0

    print(
        f"KL(P||P) = "
        f"{kl_identical:.6e}"
    )

    print(
        f"KL(P||Q) = "
        f"{kl_pq:.6f}"
    )

    print(
        f"KL(Q||P) = "
        f"{kl_qp:.6f}"
    )

    print(
        "PASS: KL divergence is "
        "non-negative and zero for "
        "identical distributions."
    )


# =============================================================================
# TEST 8 — SCORE
# =============================================================================

def test_score():

    print()
    print(
        "TEST 8 — Gaussian Score"
    )

    x = torch.tensor(
        [-2.0, 0.0, 2.0],
        dtype=torch.float64,
    )

    score = gaussian_score(
        x,
        mean=0.0,
        variance=1.0,
    )

    expected = torch.tensor(
        [2.0, 0.0, -2.0],
        dtype=torch.float64,
    )

    assert torch.allclose(
        score,
        expected,
    )

    print(
        "PASS: score of N(0,1) "
        "is -x."
    )


# =============================================================================
# TEST 9 — IMAGE-SHAPED GAUSSIAN NOISE
# =============================================================================

def test_image_noise():

    print()
    print(
        "TEST 9 — Image-Shaped Gaussian Noise"
    )

    noise = torch.randn(
        8,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    assert noise.shape == (
        8,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert torch.isfinite(
        noise
    ).all()

    print(
        f"PASS: Gaussian noise shape = "
        f"{tuple(noise.shape)}"
    )


# =============================================================================
# TEST 10 — NOISY IMAGE EQUATION
# =============================================================================

def test_noisy_image_equation():

    print()
    print(
        "TEST 10 — Image + Gaussian Noise"
    )

    image = torch.rand(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    epsilon = torch.randn_like(
        image
    )

    alpha = 0.8
    sigma = 0.6

    noisy = (
        alpha * image
        + sigma * epsilon
    )

    assert noisy.shape == image.shape

    assert torch.isfinite(
        noisy
    ).all()

    print(
        "PASS: noisy image has "
        "same shape as original."
    )


# =============================================================================
# VISUALIZATION 1 — GAUSSIAN DISTRIBUTIONS
# =============================================================================

def visualize_gaussian_distributions():

    x = torch.linspace(
        -5.0,
        5.0,
        1000,
        dtype=torch.float64,
    )

    variances = [
        0.25,
        1.0,
        4.0,
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for variance in variances:

        pdf = gaussian_pdf(
            x,
            mean=0.0,
            variance=variance,
        )

        ax.plot(
            x.numpy(),
            pdf.numpy(),
            linewidth=2,
            label=f"variance={variance}",
        )

    ax.set_title(
        "Gaussian Distributions with Different Variances"
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "p(x)"
    )

    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZATION 2 — 2D GAUSSIAN
# =============================================================================

def visualize_2d_gaussian():

    mean = torch.tensor(
        [0.0, 0.0],
        dtype=torch.float64,
    )

    covariance = torch.tensor(
        [
            [1.0, 0.75],
            [0.75, 1.0],
        ],
        dtype=torch.float64,
    )

    samples = sample_multivariate_gaussian(
        mean,
        covariance,
        20_000,
        device=DEVICE,
    )

    samples_cpu = (
        samples
        .detach()
        .cpu()
        .numpy()
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.scatter(
        samples_cpu[:, 0],
        samples_cpu[:, 1],
        s=4,
        alpha=0.25,
    )

    ax.set_xlabel(
        "x1"
    )

    ax.set_ylabel(
        "x2"
    )

    ax.set_title(
        "Correlated Multivariate Gaussian"
    )

    ax.axis(
        "equal"
    )

    ax.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZATION 3 — SCORE
# =============================================================================

def visualize_gaussian_score():

    x = torch.linspace(
        -4.0,
        4.0,
        1000,
        dtype=torch.float64,
    )

    score = gaussian_score(
        x,
        mean=0.0,
        variance=1.0,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        x.numpy(),
        score.numpy(),
        linewidth=3,
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.axvline(
        0.0,
        linewidth=1,
    )

    ax.set_title(
        "Score of Standard Gaussian: score(x) = -x"
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "d log p(x) / dx"
    )

    ax.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZATION 4 — IMAGE NOISE
# =============================================================================

def visualize_image_noise():

    image = torch.zeros(
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    # Synthetic RGB object.
    image[
        0,
        8:24,
        8:24,
    ] = 1.0

    image[
        1,
        12:20,
        12:20,
    ] = 0.5

    epsilon = torch.randn_like(
        image
    )

    alpha = 0.8
    sigma = 0.35

    noisy = (
        alpha * image
        + sigma * epsilon
    )

    image_display = (
        image
        .clamp(0, 1)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    noisy_display = (
        noisy
        .clamp(0, 1)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    axes[0].imshow(
        image_display
    )

    axes[0].set_title(
        "Original synthetic image"
    )

    axes[0].axis(
        "off"
    )

    axes[1].imshow(
        noisy_display
    )

    axes[1].set_title(
        "Image + Gaussian noise"
    )

    axes[1].axis(
        "off"
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZATION 5 — NOISE LEVELS
# =============================================================================

def visualize_noise_levels():

    image = torch.zeros(
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    image[
        0,
        7:25,
        7:25,
    ] = 1.0

    image[
        1,
        11:21,
        11:21,
    ] = 0.5

    noise = torch.randn_like(
        image
    )

    alpha_values = [
        1.0,
        0.85,
        0.65,
        0.40,
        0.10,
    ]

    sigma_values = [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    ]

    fig, axes = plt.subplots(
        1,
        len(alpha_values),
        figsize=(15, 3.5)
    )

    for i, (
        alpha,
        sigma,
    ) in enumerate(
        zip(
            alpha_values,
            sigma_values,
        )
    ):

        noisy = (
            alpha * image
            + sigma * noise
        )

        display = (
            noisy
            .clamp(0, 1)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        axes[i].imshow(
            display
        )

        axes[i].set_title(
            f"alpha={alpha:.2f}\n"
            f"sigma={sigma:.2f}"
        )

        axes[i].axis(
            "off"
        )

    fig.suptitle(
        "Increasing Gaussian Noise"
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    print("=" * 80)
    print(
        "LESSON 15 — PROBABILITY DISTRIBUTIONS "
        "FOR IMAGE GENERATION"
    )
    print("=" * 80)

    print()
    print(
        f"Device: {DEVICE}"
    )

    # =========================================================================
    # UNIT TESTS
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "RUNNING TESTS"
    )
    print("=" * 80)

    test_gaussian_pdf_normalization()

    test_gaussian_sampling()

    test_multivariate_gaussian()

    test_standard_gaussian()

    test_reparameterization()

    test_entropy()

    test_kl_divergence()

    test_score()

    test_image_noise()

    test_noisy_image_equation()

    print()
    print("=" * 80)
    print(
        "ALL TESTS PASSED"
    )
    print("=" * 80)

    # =========================================================================
    # EXPERIMENT 1 — GAUSSIAN DISTRIBUTIONS
    # =========================================================================

    print()
    print(
        "EXPERIMENT 1 — Gaussian variance"
    )

    visualize_gaussian_distributions()

    # =========================================================================
    # EXPERIMENT 2 — 2D GAUSSIAN
    # =========================================================================

    print()
    print(
        "EXPERIMENT 2 — Multivariate Gaussian"
    )

    visualize_2d_gaussian()

    # =========================================================================
    # EXPERIMENT 3 — SCORE
    # =========================================================================

    print()
    print(
        "EXPERIMENT 3 — Gaussian score"
    )

    visualize_gaussian_score()

    # =========================================================================
    # EXPERIMENT 4 — IMAGE NOISE
    # =========================================================================

    print()
    print(
        "EXPERIMENT 4 — Image + Gaussian noise"
    )

    visualize_image_noise()

    # =========================================================================
    # EXPERIMENT 5 — INCREASING NOISE
    # =========================================================================

    print()
    print(
        "EXPERIMENT 5 — Increasing noise"
    )

    visualize_noise_levels()

    # =========================================================================
    # KL MONTE CARLO VERIFICATION
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "MONTE CARLO KL VERIFICATION"
    )
    print("=" * 80)

    analytical_kl = (
        gaussian_kl_divergence(
            mean_p=0.0,
            variance_p=1.0,
            mean_q=1.0,
            variance_q=2.0,
        )
    )

    monte_carlo_kl = (
        monte_carlo_gaussian_kl(
            mean_p=0.0,
            variance_p=1.0,
            mean_q=1.0,
            variance_q=2.0,
            num_samples=100_000,
        )
    )

    print(
        f"Analytical KL(P||Q): "
        f"{analytical_kl:.6f}"
    )

    print(
        f"Monte Carlo KL(P||Q): "
        f"{monte_carlo_kl:.6f}"
    )

    assert abs(
        analytical_kl
        - monte_carlo_kl
    ) < 0.02

    print(
        "PASS: Monte Carlo KL "
        "agrees with analytical KL."
    )

    # =========================================================================
    # REPARAMETERIZATION DEMONSTRATION
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "REPARAMETERIZATION DEMONSTRATION"
    )
    print("=" * 80)

    mean = torch.tensor(
        2.0,
        device=DEVICE,
    )

    std = torch.tensor(
        0.5,
        device=DEVICE,
    )

    epsilon = torch.randn(
        10_000,
        device=DEVICE,
    )

    samples = (
        mean
        + std * epsilon
    )

    print(
        f"Target mean: "
        f"{mean.item():.4f}"
    )

    print(
        f"Sample mean: "
        f"{samples.mean().item():.4f}"
    )

    print(
        f"Target std: "
        f"{std.item():.4f}"
    )

    print(
        f"Sample std: "
        f"{samples.std().item():.4f}"
    )

    # =========================================================================
    # REAL-WORLD ROBOTICS CONNECTION
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "REAL-WORLD ROBOTICS APPLICATION"
    )
    print("=" * 80)

    print(
        """
Suppose an OpenArm robot observes a cube through an RGB camera.

The image can be represented as:

    x in R^(H x W x 3)

Sensor uncertainty can be modeled using:

    epsilon ~ N(0, I)

A synthetic noisy observation can then be constructed as:

    x_noisy
        =
    alpha*x
        +
    sigma*epsilon


This allows us to simulate different visual conditions:

    low sensor noise
    medium sensor noise
    strong sensor noise

Such techniques are useful for:

    robot perception robustness
    data augmentation
    simulation
    domain randomization
    visual policy training
    diffusion-based image generation
"""
    )

    # =========================================================================
    # DIFFUSION BRIDGE
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "THE DIFFUSION BRIDGE"
    )
    print("=" * 80)

    print(
        """
The most important equation from this lesson is:

    x_noisy
        =
    alpha*x
        +
    sigma*epsilon

where:

    epsilon ~ N(0, I)


Think about what happens when sigma increases.

sigma small:

    x_noisy ~= x


sigma medium:

    x_noisy = image + visible noise


sigma large:

    x_noisy ~= Gaussian noise


Therefore we can construct a path:

    CLEAN IMAGE
         |
         | small noise
         v
      SLIGHTLY NOISY
         |
         | more noise
         v
      VERY NOISY
         |
         | more noise
         v
      GAUSSIAN NOISE


This is exactly the direction we need for diffusion.

Lesson 16 will formalize this process using a sequence of noise levels.
"""
    )

    # =========================================================================
    # LESSON RESULT
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "LESSON 15 RESULT"
    )
    print("=" * 80)

    print(
        """
SUCCESS.

You now have the probability foundations required for diffusion.

Key concepts:

    Gaussian distribution
    N(mu, sigma^2)

    Standard Gaussian
    N(0, 1)

    Multivariate Gaussian
    N(mu, Sigma)

    Standard multivariate Gaussian
    N(0, I)

    Reparameterization
    z = mu + sigma*epsilon

    KL divergence
    KL(P||Q)

    Entropy
    H(X)

    Gaussian score
    score(x) = -(x-mu)/sigma^2

    Noisy data
    x_noisy = alpha*x + sigma*epsilon


The central mental model is now:

    DATA
      |
      | gradually add Gaussian noise
      v
    NOISE


The next lesson reverses the question:

    If we know how to destroy an image using noise,
    how can we mathematically learn to reverse that process?


NEXT LESSON:

    LESSON 16 — FORWARD DIFFUSION PROCESS

We will implement the complete forward process:

    x_0
     |
     v
    x_1
     |
     v
    x_2
     |
     v
    ...
     |
     v
    x_T


and derive the important closed-form equation:

    x_t
      =
    sqrt(alpha_bar_t) * x_0
      +
    sqrt(1-alpha_bar_t) * epsilon

where:

    epsilon ~ N(0,I)

That equation is one of the most important equations in the entire
diffusion-model course.
"""
    )

    print()
    print("=" * 80)
    print(
        "LESSON 15 COMPLETE"
    )
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()