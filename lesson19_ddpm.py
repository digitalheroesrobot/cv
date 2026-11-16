"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 19 — DDPM: DENOISING DIFFUSION PROBABILISTIC MODEL
================================================================================

Filename:
    lesson19_ddpm.py

================================================================================
PURPOSE
================================================================================

Lesson 18 introduced the first learned component of diffusion:

    epsilon_theta(x_t, t)

The network learned to predict the Gaussian noise that was used
to corrupt an image.

This lesson connects that learned noise predictor to the COMPLETE
REVERSE DIFFUSION PROCESS.

We will implement a small DDPM from scratch.

The complete generative pipeline becomes:

    random Gaussian noise
            |
            v
           x_T
            |
            v
       reverse step
            |
            v
       x_{T-1}
            |
            v
       reverse step
            |
            v
           ...
            |
            v
           x_1
            |
            v
           x_0

This is the first lesson where the system can generate an image
starting from pure noise.

This is a foundational diffusion model.

It is NOT Stable Diffusion.

It is a small educational DDPM implemented from random
initialization.


================================================================================
WHAT YOU WILL LEARN
================================================================================

By the end of this lesson you will understand:

    1. What DDPM means.
    2. The forward diffusion Markov chain.
    3. The reverse diffusion Markov chain.
    4. Why the reverse process requires a neural network.
    5. How predicted noise determines the reverse mean.
    6. Why the posterior variance is needed.
    7. How to sample x_{t-1} from x_t.
    8. Why noise is added during reverse sampling.
    9. Why the final timestep must not add additional noise.
   10. How training and sampling connect.
   11. How a model can generate images from random Gaussian noise.
   12. How DDPM relates to the noise-prediction objective from Lesson 18.


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
    Lesson 17  Noise schedules
    Lesson 18  Learning to predict noise

Current:

    Lesson 19  DDPM

Next:

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

Suppose a robot needs synthetic camera images of objects.

Instead of storing every possible camera image, we would like
a generative model that learns the visual distribution.

A diffusion generator learns:

    random noise
        |
        v
    reverse denoising
        |
        v
    structured visual scene

The generated image can eventually become:

    - synthetic robot training data
    - simulation imagery
    - object variations
    - visual augmentation
    - image restoration results
    - generative perception priors


================================================================================
DDPM — THE CORE IDEA
================================================================================

DDPM stands for:

    Denoising Diffusion Probabilistic Model

The model defines two processes.

FORWARD PROCESS:

    q(x_t | x_{t-1})

which gradually adds Gaussian noise.

REVERSE PROCESS:

    p_theta(x_{t-1} | x_t)

which learns to remove that noise.


================================================================================
FORWARD PROCESS
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


The complete forward process is:

    x_0 -> x_1 -> x_2 -> ... -> x_T


The closed-form equation is:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

where:

    alpha_bar_t =
        product_{s=1}^t alpha_s

and:

    epsilon ~ N(0,I)


================================================================================
REVERSE PROCESS
================================================================================

The reverse process is:

    x_T -> x_{T-1} -> ... -> x_1 -> x_0

The model approximates:

    p_theta(x_{t-1}|x_t)

with a Gaussian:

    p_theta(x_{t-1}|x_t)
        =
    N(
        mu_theta(x_t,t),
        Sigma_t
    )

The important problem is therefore:

    How do we calculate mu_theta?


================================================================================
FROM NOISE PREDICTION TO REVERSE MEAN
================================================================================

Lesson 18 gives:

    epsilon_theta(x_t,t)

The forward equation is:

    x_t =
        sqrt(alpha_bar_t)x_0
        +
        sqrt(1-alpha_bar_t)epsilon

Solving for x_0:

    x_0 =
        [
            x_t
            -
            sqrt(1-alpha_bar_t)epsilon
        ]
        /
        sqrt(alpha_bar_t)

Therefore the network gives an estimate:

    x0_theta =
        [
            x_t
            -
            sqrt(1-alpha_bar_t)
            epsilon_theta
        ]
        /
        sqrt(alpha_bar_t)


The DDPM reverse mean can be written directly in terms of
predicted noise:

    mu_theta(x_t,t)
        =
    1 / sqrt(alpha_t)
    [
        x_t
        -
        beta_t /
        sqrt(1-alpha_bar_t)
        *
        epsilon_theta(x_t,t)
    ]


This is one of the most important equations in the course.


================================================================================
DDPM POSTERIOR VARIANCE
================================================================================

The true forward-process posterior is:

    q(x_{t-1}|x_t,x_0)

and its variance is:

    beta_tilde_t
        =
    beta_t
    *
    (1-alpha_bar_{t-1})
    /
    (1-alpha_bar_t)

Therefore DDPM uses:

    sigma_t^2 = beta_tilde_t

for the reverse transition.


================================================================================
REVERSE SAMPLING EQUATION
================================================================================

For t > 1:

    x_{t-1}
        =
    mu_theta(x_t,t)
    +
    sqrt(beta_tilde_t) z

where:

    z ~ N(0,I)


At the final step:

    t = 1

we do NOT add additional random noise.

Therefore:

    x_0 = mu_theta(x_1,1)


This detail is important.

If random noise were added at the final step, the final image
would be unnecessarily corrupted.


================================================================================
THE COMPLETE DDPM SAMPLER
================================================================================

Start:

    x_T ~ N(0,I)

Then:

    for t = T,T-1,...,1:

        epsilon_pred =
            epsilon_theta(x_t,t)

        calculate reverse mean

        if t > 1:

            sample random z

            x_{t-1} =
                mean + sigma_t z

        else:

            x_0 =
                mean


This produces a sample from the learned distribution.


================================================================================
TRAINING OBJECTIVE
================================================================================

The simplified DDPM objective used here is:

    L_simple =
        E[
            ||epsilon
            -
            epsilon_theta(x_t,t)||^2
        ]

Training therefore remains exactly the noise-prediction
problem from Lesson 18.

The major change in this lesson is SAMPLING.

Lesson 18:

    learn epsilon_theta

Lesson 19:

    use epsilon_theta
    to reverse the diffusion chain


================================================================================
WHY DOES REVERSE SAMPLING NEED RANDOMNESS?
================================================================================

The reverse distribution is probabilistic.

Therefore for intermediate timesteps:

    x_{t-1}
        =
    mean
    +
    random_noise

This means two samples starting from different x_T values can
produce different images.

Even with the same trained model:

    random seed A -> image A

    random seed B -> image B


This is the fundamental source of generative diversity.


================================================================================
WHY DOES THE MODEL NEED t?
================================================================================

At:

    t = 900

the input is heavily corrupted.

At:

    t = 100

the input is much cleaner.

The reverse operation is therefore different at every timestep.

The network must know:

    "How far through the diffusion process are we?"


================================================================================
MODEL ARCHITECTURE
================================================================================

This lesson uses the small timestep-conditioned CNN introduced
in Lesson 18.

Input:

    x_t
        [B,1,H,W]

and:

    t
        [B]

Network:

    image
       |
       v
    Conv
       |
       v
    Conv
       |
       +------ timestep embedding
       |
       v
    Conv
       |
       v
    Conv
       |
       v
    epsilon prediction

This architecture is intentionally small.

Lesson 20 will replace this educational CNN with a more complete
diffusion U-Net.


================================================================================
DATA
================================================================================

We generate small structured grayscale images.

The images contain:

    - gradients
    - rectangles
    - circles
    - diagonal structures
    - texture

The goal is not photorealism.

The goal is to make the complete DDPM mathematics visible and
runnable on a normal GPU.


================================================================================
TRAINING STRATEGY
================================================================================

Each training iteration:

    1. Generate clean images.
    2. Randomly choose t.
    3. Sample Gaussian noise.
    4. Construct x_t.
    5. Predict epsilon.
    6. Compute MSE.
    7. Backpropagate.
    8. Update the network.


================================================================================
SAMPLING STRATEGY
================================================================================

After training:

    x_T = Gaussian noise

Then:

    t = T

    t = T-1

    ...

    t = 1

At every step:

    predict noise
    calculate reverse mean
    optionally add posterior noise


================================================================================
IMPORTANT NUMERICAL DETAIL
================================================================================

At very early timesteps:

    alpha_bar_t

can be very close to 1.

At very late timesteps:

    alpha_bar_t

can be very close to 0.

Therefore divisions such as:

    1 / sqrt(alpha_bar_t)

must be numerically protected.

The implementation uses small positive clamps where necessary.


================================================================================
VERIFICATION / TESTS
================================================================================

The script verifies:

    TEST 1
        Schedule structure.

    TEST 2
        Posterior variance is valid.

    TEST 3
        Reverse mean is finite.

    TEST 4
        Posterior sampling preserves shape.

    TEST 5
        Model output shape.

    TEST 6
        Gradient flow.

    TEST 7
        Training loss decreases.

    TEST 8
        Deterministic reverse mean at t=1.

    TEST 9
        Complete DDPM sampler produces finite images.

    TEST 10
        Device consistency.


================================================================================
VISUALIZATION
================================================================================

The script displays:

    1. Clean training examples.
    2. Forward diffusion examples.
    3. Training loss.
    4. Reverse diffusion trajectory.
    5. Generated images.
    6. Multiple generated samples.
    7. Pixel statistics of generated images.


================================================================================
REAL-WORLD APPLICATION
================================================================================

A learned diffusion model can eventually be used to generate
visual observations for robotics.

For example:

    random latent noise
            |
            v
       DDPM sampler
            |
            v
      synthetic image
            |
            v
      robot perception
            |
            v
      manipulation policy

This can help create visual variations that are difficult or
expensive to collect with physical robots.


================================================================================
REAL-WORLD RESULT
================================================================================

The major result of this lesson is architectural rather than
photorealistic.

We now have:

    forward diffusion
    +
    learned noise prediction
    +
    reverse probabilistic sampling

Together these form a complete DDPM.


================================================================================
LESSON RESULT
================================================================================

The conceptual chain is now:

    x_0
     |
     | forward diffusion
     v
    x_T
     |
     | learned reverse diffusion
     v
    x_0_generated

This is the first complete generative diffusion pipeline in
the course.


================================================================================
CONNECTION TO NEXT LESSON
================================================================================

Lesson 20 will introduce the Diffusion U-Net.

The current CNN is intentionally small.

A stronger architecture needs:

    - encoder / decoder structure
    - multi-scale features
    - residual blocks
    - skip connections
    - timestep conditioning at multiple depths

The DDPM mathematics will remain largely the same.

The major improvement will be the neural network architecture.

Therefore:

    Lesson 19:
        DDPM mathematics + complete reverse process

    Lesson 20:
        Diffusion U-Net architecture


================================================================================
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEED = 42

T = 1000

IMAGE_SIZE = 32

BETA_START = 1e-4
BETA_END = 0.02

BATCH_SIZE = 64

TRAIN_STEPS = 2200

LEARNING_RATE = 2e-3

TIME_EMBED_DIM = 64

BASE_CHANNELS = 32

SAMPLE_COUNT = 8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TEST_DEVICE = torch.device(
    "cpu"
)

TEST_DTYPE = torch.float64

EPS = 1e-8


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(
    seed: int = SEED,
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ==============================================================================
# DEVICE HELPERS
# ==============================================================================

def device_description(
    device: torch.device,
) -> str:

    if device.type == "cuda":

        index = (
            torch.cuda.current_device()
            if device.index is None
            else device.index
        )

        return f"cuda:{index}"

    return device.type


def devices_match(
    a: torch.device,
    b: torch.device,
) -> bool:

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
# SYNTHETIC IMAGE GENERATION
# ==============================================================================

def create_structured_image(
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate one structured grayscale image.

    Output:

        [1,H,W]

    The image contains several spatial structures so that the
    diffusion model learns something more meaningful than a
    constant pixel distribution.
    """

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
        0.10
        +
        0.12 * xx
        +
        0.08 * yy
    )

    # Random smooth background variation.
    background = torch.sin(
        2.0
        * math.pi
        * (
            1.0
            +
            torch.rand(
                (),
                device=device,
                dtype=dtype,
            )
        )
        * xx
    )

    image = (
        image
        +
        0.025 * background
    )

    # Rectangle.
    rx = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.50
        +
        0.05
    )

    ry = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.50
        +
        0.05
    )

    rw = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.25
        +
        0.12
    )

    rh = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.25
        +
        0.12
    )

    rectangle = (
        (xx > rx)
        &
        (xx < rx + rw)
        &
        (yy > ry)
        &
        (yy < ry + rh)
    )

    image = (
        image
        +
        rectangle.to(dtype)
        *
        0.45
    )

    # Circle.
    cx = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.70
        +
        0.15
    )

    cy = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.70
        +
        0.15
    )

    radius = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.10
        +
        0.06
    )

    circle = (
        (xx - cx) ** 2
        +
        (yy - cy) ** 2
        <
        radius ** 2
    )

    image = (
        image
        +
        circle.to(dtype)
        *
        0.50
    )

    # Diagonal structure.
    slope = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 1.2
        -
        0.6
    )

    intercept = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.8
        +
        0.1
    )

    diagonal = (
        torch.abs(
            yy
            -
            (
                slope * xx
                +
                intercept
            )
        )
        <
        0.025
    )

    image = (
        image
        +
        diagonal.to(dtype)
        *
        0.25
    )

    # Texture.
    texture = (
        0.025
        *
        torch.sin(
            2.0
            * math.pi
            * 4.0
            * xx
        )
        *
        torch.sin(
            2.0
            * math.pi
            * 3.0
            * yy
        )
    )

    image = image + texture

    return image.clamp(
        0.0,
        1.0,
    ).unsqueeze(0)


def create_image_batch(
    batch_size: int,
    device: torch.device = DEVICE,
) -> torch.Tensor:

    images = [
        create_structured_image(
            device=device,
        )
        for _ in range(
            batch_size
        )
    ]

    return torch.stack(
        images,
        dim=0,
    )


# ==============================================================================
# DDPM SCHEDULE
# ==============================================================================

def make_beta_schedule(
    timesteps: int = T,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
    device: torch.device = DEVICE,
) -> torch.Tensor:

    return torch.linspace(
        beta_start,
        beta_end,
        timesteps,
        device=device,
        dtype=torch.float32,
    )


def prepare_ddpm_schedule(
    beta: torch.Tensor,
) -> dict:
    """
    Prepare every quantity required by DDPM.

    beta_t:
        forward noise variance

    alpha_t:
        1 - beta_t

    alpha_bar_t:
        product of alpha values

    posterior_variance:
        beta_tilde_t
    """

    alpha = 1.0 - beta

    alpha_bar = torch.cumprod(
        alpha,
        dim=0,
    )

    alpha_bar = alpha_bar.clamp(
        min=EPS,
        max=1.0,
    )

    alpha_bar_prev = torch.cat(
        [
            torch.ones(
                1,
                device=beta.device,
                dtype=beta.dtype,
            ),
            alpha_bar[:-1],
        ],
        dim=0,
    )

    sqrt_alpha = torch.sqrt(
        alpha
    )

    sqrt_alpha_bar = torch.sqrt(
        alpha_bar
    )

    sqrt_one_minus_alpha_bar = torch.sqrt(
        (
            1.0
            -
            alpha_bar
        ).clamp(
            min=EPS
        )
    )

    posterior_variance = (
        beta
        *
        (
            1.0
            -
            alpha_bar_prev
        )
        /
        (
            1.0
            -
            alpha_bar
        ).clamp(
            min=EPS
        )
    )

    posterior_variance = posterior_variance.clamp(
        min=0.0
    )

    posterior_log_variance = torch.log(
        posterior_variance.clamp(
            min=EPS
        )
    )

    posterior_mean_coef1 = (
        beta
        *
        torch.sqrt(
            alpha_bar_prev
        )
        /
        (
            1.0
            -
            alpha_bar
        ).clamp(
            min=EPS
        )
    )

    posterior_mean_coef2 = (
        (
            1.0
            -
            alpha_bar_prev
        )
        *
        torch.sqrt(alpha)
        /
        (
            1.0
            -
            alpha_bar
        ).clamp(
            min=EPS
        )
    )

    return {
        "beta":
            beta,

        "alpha":
            alpha,

        "alpha_bar":
            alpha_bar,

        "alpha_bar_prev":
            alpha_bar_prev,

        "sqrt_alpha":
            sqrt_alpha,

        "sqrt_alpha_bar":
            sqrt_alpha_bar,

        "sqrt_one_minus_alpha_bar":
            sqrt_one_minus_alpha_bar,

        "posterior_variance":
            posterior_variance,

        "posterior_log_variance":
            posterior_log_variance,

        "posterior_mean_coef1":
            posterior_mean_coef1,

        "posterior_mean_coef2":
            posterior_mean_coef2,
    }


# ==============================================================================
# SCHEDULE EXTRACTION
# ==============================================================================

def extract(
    values: torch.Tensor,
    t: torch.Tensor,
    x_shape: torch.Size,
) -> torch.Tensor:

    batch_size = t.shape[0]

    selected = values[
        t
    ]

    return selected.reshape(
        batch_size,
        *(
            [1]
            *
            (
                len(x_shape)
                - 1
            )
        ),
    )


# ==============================================================================
# FORWARD DIFFUSION
# ==============================================================================

def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: dict,
    noise: torch.Tensor | None = None,
):
    """
    Closed-form forward diffusion:

        x_t =
            sqrt(alpha_bar_t) x_0
            +
            sqrt(1-alpha_bar_t) epsilon
    """

    if noise is None:

        noise = torch.randn_like(
            x0
        )

    sqrt_alpha_bar = extract(
        schedule[
            "sqrt_alpha_bar"
        ],
        t,
        x0.shape,
    ).to(
        device=x0.device,
        dtype=x0.dtype,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule[
            "sqrt_one_minus_alpha_bar"
        ],
        t,
        x0.shape,
    ).to(
        device=x0.device,
        dtype=x0.dtype,
    )

    xt = (
        sqrt_alpha_bar * x0
        +
        sqrt_one_minus_alpha_bar
        * noise
    )

    return xt, noise


# ==============================================================================
# TIMESTEP EMBEDDING
# ==============================================================================

def sinusoidal_timestep_embedding(
    t: torch.Tensor,
    embedding_dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:

    if embedding_dim % 2 != 0:

        raise ValueError(
            "embedding_dim must be even."
        )

    half_dim = embedding_dim // 2

    frequencies = torch.exp(
        -math.log(max_period)
        *
        torch.arange(
            half_dim,
            device=t.device,
            dtype=torch.float32,
        )
        /
        max(
            half_dim - 1,
            1,
        )
    )

    angles = (
        t.float().unsqueeze(1)
        *
        frequencies.unsqueeze(0)
    )

    return torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=1,
    )


# ==============================================================================
# NOISE PREDICTOR
# ==============================================================================

class NoisePredictor(nn.Module):
    """
    Small timestep-conditioned CNN.

    This is intentionally similar to Lesson 18.

    The important lesson is that the SAME learned noise predictor
    can now be used inside the DDPM reverse process.
    """

    def __init__(
        self,
        time_embed_dim: int = TIME_EMBED_DIM,
        base_channels: int = BASE_CHANNELS,
    ):

        super().__init__()

        self.time_embed_dim = (
            time_embed_dim
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(
                time_embed_dim,
                base_channels,
            ),
            nn.SiLU(),
            nn.Linear(
                base_channels,
                base_channels,
            ),
        )

        self.conv1 = nn.Conv2d(
            1,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv4 = nn.Conv2d(
            base_channels,
            1,
            kernel_size=3,
            padding=1,
        )

        self.activation = nn.SiLU()

        self.initialize_weights()

    def initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.Linear,
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        h = self.activation(
            self.conv1(x)
        )

        h = self.activation(
            self.conv2(h)
        )

        timestep_embedding = (
            sinusoidal_timestep_embedding(
                t,
                self.time_embed_dim,
            )
        )

        timestep_embedding = (
            self.time_mlp(
                timestep_embedding
            )
        )

        timestep_embedding = (
            timestep_embedding
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        h = h + timestep_embedding

        h = self.activation(
            self.conv3(h)
        )

        return self.conv4(h)


# ==============================================================================
# TRAINING LOSS
# ==============================================================================

def noise_prediction_loss(
    model: nn.Module,
    x0: torch.Tensor,
    schedule: dict,
):

    batch_size = x0.shape[0]

    t = torch.randint(
        0,
        T,
        (
            batch_size,
        ),
        device=x0.device,
        dtype=torch.long,
    )

    xt, noise = q_sample(
        x0,
        t,
        schedule,
    )

    predicted_noise = model(
        xt,
        t,
    )

    loss = F.mse_loss(
        predicted_noise,
        noise,
    )

    return (
        loss,
        xt,
        noise,
        predicted_noise,
        t,
    )


# ==============================================================================
# DDPM REVERSE MEAN
# ==============================================================================

def ddpm_reverse_mean(
    xt: torch.Tensor,
    t: torch.Tensor,
    predicted_noise: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    """
    Compute:

        mu_theta(x_t,t)
            =
        1/sqrt(alpha_t)
        [
            x_t
            -
            beta_t /
            sqrt(1-alpha_bar_t)
            *
            epsilon_theta
        ]

    This is the central DDPM reverse equation.
    """

    beta_t = extract(
        schedule["beta"],
        t,
        xt.shape,
    ).to(
        device=xt.device,
        dtype=xt.dtype,
    )

    sqrt_alpha_t = extract(
        schedule["sqrt_alpha"],
        t,
        xt.shape,
    ).to(
        device=xt.device,
        dtype=xt.dtype,
    )

    sqrt_one_minus_alpha_bar_t = extract(
        schedule[
            "sqrt_one_minus_alpha_bar"
        ],
        t,
        xt.shape,
    ).to(
        device=xt.device,
        dtype=xt.dtype,
    )

    coefficient = (
        beta_t
        /
        (
            sqrt_one_minus_alpha_bar_t
            +
            EPS
        )
    )

    mean = (
        xt
        -
        coefficient
        *
        predicted_noise
    ) / (
        sqrt_alpha_t
        +
        EPS
    )

    return mean


# ==============================================================================
# DDPM POSTERIOR SAMPLE
# ==============================================================================

def ddpm_posterior_sample(
    xt: torch.Tensor,
    t: torch.Tensor,
    predicted_noise: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    """
    Perform one reverse DDPM step.

    For t > 0:

        x_{t-1}
            =
        mu_theta
        +
        sqrt(beta_tilde_t) z

    For t = 0:

        x_0 = mu_theta

    The implementation uses zero-indexed t:

        t = 999 corresponds to diffusion timestep 1000.

        t = 0 corresponds to final x_0.
    """

    mean = ddpm_reverse_mean(
        xt,
        t,
        predicted_noise,
        schedule,
    )

    posterior_variance = extract(
        schedule[
            "posterior_variance"
        ],
        t,
        xt.shape,
    ).to(
        device=xt.device,
        dtype=xt.dtype,
    )

    # In zero-indexed notation:
    #
    # t > 0 means another reverse step exists.
    #
    # t == 0 means we have reached x_0.

    noise_mask = (
        (t > 0)
        .float()
        .reshape(
            t.shape[0],
            *(
                [1]
                *
                (
                    xt.ndim - 1
                )
            ),
        )
    )

    noise = torch.randn_like(
        xt
    )

    sample = (
        mean
        +
        noise_mask
        *
        torch.sqrt(
            posterior_variance.clamp(
                min=0.0
            )
        )
        *
        noise
    )

    return sample


# ==============================================================================
# ONE COMPLETE REVERSE STEP
# ==============================================================================

@torch.no_grad()
def ddpm_reverse_step(
    model: nn.Module,
    xt: torch.Tensor,
    timestep: int,
    schedule: dict,
):
    """
    timestep uses human-readable numbering:

        1 ... T

    The schedule itself uses zero-based indexing.
    """

    t_index = timestep - 1

    t = torch.full(
        (
            xt.shape[0],
        ),
        t_index,
        device=xt.device,
        dtype=torch.long,
    )

    predicted_noise = model(
        xt,
        t,
    )

    x_previous = (
        ddpm_posterior_sample(
            xt,
            t,
            predicted_noise,
            schedule,
        )
    )

    return (
        x_previous,
        predicted_noise,
    )


# ==============================================================================
# COMPLETE DDPM SAMPLER
# ==============================================================================

@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    schedule: dict,
    batch_size: int,
    image_size: int = IMAGE_SIZE,
    record_timesteps: list[int] | None = None,
):
    """
    Start from Gaussian noise and repeatedly apply the reverse
    DDPM transition.

    Initial state:

        x_T ~ N(0,I)

    Final state:

        x_0
    """

    model.eval()

    xt = torch.randn(
        (
            batch_size,
            1,
            image_size,
            image_size,
        ),
        device=DEVICE,
        dtype=torch.float32,
    )

    trajectory = {}

    if record_timesteps is None:

        record_timesteps = []

    if T in record_timesteps:

        trajectory[T] = (
            xt.detach().cpu()
        )

    for timestep in range(
        T,
        0,
        -1,
    ):

        xt, _ = ddpm_reverse_step(
            model,
            xt,
            timestep,
            schedule,
        )

        if timestep - 1 in record_timesteps:

            trajectory[
                timestep - 1
            ] = (
                xt.detach().cpu()
            )

    model.train()

    return (
        xt,
        trajectory,
    )


# ==============================================================================
# TEST 1 — SCHEDULE
# ==============================================================================

def test_schedule(
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 1 — DDPM SCHEDULE"
    )
    print("=" * 80)

    beta = schedule["beta"]

    alpha = schedule["alpha"]

    alpha_bar = schedule[
        "alpha_bar"
    ]

    print(
        f"beta[0]       : "
        f"{beta[0].item():.8f}"
    )

    print(
        f"beta[-1]      : "
        f"{beta[-1].item():.8f}"
    )

    print(
        f"alpha_bar[0]  : "
        f"{alpha_bar[0].item():.8f}"
    )

    print(
        f"alpha_bar[-1] : "
        f"{alpha_bar[-1].item():.8e}"
    )

    assert beta.shape == (
        T,
    )

    assert torch.all(
        beta > 0
    )

    assert torch.all(
        beta < 1
    )

    assert torch.allclose(
        alpha,
        1.0 - beta,
    )

    assert torch.all(
        alpha_bar[1:]
        <=
        alpha_bar[:-1]
    )

    assert torch.isfinite(
        alpha_bar
    ).all()

    print(
        "[PASS] DDPM schedule is valid."
    )


# ==============================================================================
# TEST 2 — POSTERIOR VARIANCE
# ==============================================================================

def test_posterior_variance(
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 2 — POSTERIOR VARIANCE"
    )
    print("=" * 80)

    variance = schedule[
        "posterior_variance"
    ]

    print(
        f"First variance : "
        f"{variance[0].item():.8e}"
    )

    print(
        f"Last variance  : "
        f"{variance[-1].item():.8e}"
    )

    assert variance.shape == (
        T,
    )

    assert torch.all(
        variance >= 0
    )

    assert torch.isfinite(
        variance
    ).all()

    # At the final clean step, the posterior variance is exactly
    # zero in the mathematical DDPM formulation.
    assert (
        variance[0].item()
        <
        1e-10
    )

    print(
        "[PASS] Posterior variance is valid."
    )


# ==============================================================================
# TEST 3 — REVERSE MEAN
# ==============================================================================

def test_reverse_mean(
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 3 — REVERSE MEAN"
    )
    print("=" * 80)

    x = torch.randn(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    predicted_noise = torch.randn_like(
        x
    )

    t = torch.tensor(
        [
            0,
            100,
            500,
            999,
        ],
        device=DEVICE,
        dtype=torch.long,
    )

    mean = ddpm_reverse_mean(
        x,
        t,
        predicted_noise,
        schedule,
    )

    print(
        f"Input shape : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Mean shape  : "
        f"{tuple(mean.shape)}"
    )

    print(
        f"Mean std    : "
        f"{mean.std().item():.6f}"
    )

    assert mean.shape == x.shape

    assert torch.isfinite(
        mean
    ).all()

    print(
        "[PASS] DDPM reverse mean is finite and correctly shaped."
    )


# ==============================================================================
# TEST 4 — POSTERIOR SAMPLE
# ==============================================================================

def test_posterior_sample(
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 4 — POSTERIOR SAMPLING"
    )
    print("=" * 80)

    x = torch.randn(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    predicted_noise = torch.randn_like(
        x
    )

    t = torch.tensor(
        [
            0,
            1,
            500,
            999,
        ],
        device=DEVICE,
        dtype=torch.long,
    )

    sample = ddpm_posterior_sample(
        x,
        t,
        predicted_noise,
        schedule,
    )

    print(
        f"Sample shape : "
        f"{tuple(sample.shape)}"
    )

    print(
        f"Sample mean   : "
        f"{sample.mean().item():.6f}"
    )

    print(
        f"Sample std    : "
        f"{sample.std().item():.6f}"
    )

    assert sample.shape == x.shape

    assert torch.isfinite(
        sample
    ).all()

    print(
        "[PASS] Posterior sampling works."
    )


# ==============================================================================
# TEST 5 — MODEL OUTPUT
# ==============================================================================

def test_model_output(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 5 — MODEL OUTPUT"
    )
    print("=" * 80)

    x0 = create_image_batch(
        4,
        DEVICE,
    )

    t = torch.tensor(
        [
            0,
            100,
            500,
            999,
        ],
        device=DEVICE,
    )

    xt, noise = q_sample(
        x0,
        t,
        schedule,
    )

    with torch.no_grad():

        prediction = model(
            xt,
            t,
        )

    print(
        f"x_t shape        : "
        f"{tuple(xt.shape)}"
    )

    print(
        f"noise shape       : "
        f"{tuple(noise.shape)}"
    )

    print(
        f"prediction shape  : "
        f"{tuple(prediction.shape)}"
    )

    assert prediction.shape == (
        x0.shape
    )

    assert torch.isfinite(
        prediction
    ).all()

    print(
        "[PASS] Model output is valid."
    )


# ==============================================================================
# TEST 6 — GRADIENT FLOW
# ==============================================================================

def test_gradient_flow(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 6 — GRADIENT FLOW"
    )
    print("=" * 80)

    model.zero_grad(
        set_to_none=True
    )

    x0 = create_image_batch(
        8,
        DEVICE,
    )

    (
        loss,
        _,
        _,
        _,
        _,
    ) = noise_prediction_loss(
        model,
        x0,
        schedule,
    )

    loss.backward()

    parameter_count = 0

    gradient_count = 0

    maximum_gradient = 0.0

    for parameter in model.parameters():

        if not parameter.requires_grad:

            continue

        parameter_count += 1

        if parameter.grad is not None:

            gradient_count += 1

            maximum_gradient = max(
                maximum_gradient,
                parameter.grad.abs().max().item(),
            )

    print(
        f"Trainable tensors      : "
        f"{parameter_count}"
    )

    print(
        f"Gradient tensors       : "
        f"{gradient_count}"
    )

    print(
        f"Maximum gradient       : "
        f"{maximum_gradient:.6e}"
    )

    assert (
        parameter_count
        ==
        gradient_count
    )

    assert maximum_gradient > 0

    print(
        "[PASS] Gradient flow is valid."
    )


# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(
    model: nn.Module,
    schedule: dict,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    history = []

    print()
    print("=" * 80)
    print(
        "TRAINING DDPM NOISE PREDICTOR"
    )
    print("=" * 80)

    print(
        f"Steps       : {TRAIN_STEPS}"
    )

    print(
        f"Batch size  : {BATCH_SIZE}"
    )

    print(
        f"LR          : {LEARNING_RATE}"
    )

    start = time.perf_counter()

    model.train()

    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):

        x0 = create_image_batch(
            BATCH_SIZE,
            DEVICE,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            loss,
            _,
            _,
            _,
            _,
        ) = noise_prediction_loss(
            model,
            x0,
            schedule,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        history.append(
            loss.item()
        )

        if (
            step == 1
            or step % 200 == 0
            or step == TRAIN_STEPS
        ):

            recent_mean = float(
                np.mean(
                    history[
                        -100:
                    ]
                )
            )

            print(
                f"Step "
                f"{step:5d}/{TRAIN_STEPS} | "
                f"loss={loss.item():.6f} | "
                f"100-step mean={recent_mean:.6f}"
            )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    print()
    print(
        f"Training time: "
        f"{elapsed:.2f} seconds"
    )

    return history


# ==============================================================================
# TEST 7 — TRAINING IMPROVEMENT
# ==============================================================================

def test_training_improvement(
    history: list[float],
):

    print()
    print("=" * 80)
    print(
        "TEST 7 — TRAINING IMPROVEMENT"
    )
    print("=" * 80)

    initial_window = history[
        :100
    ]

    final_window = history[
        -100:
    ]

    initial_loss = float(
        np.mean(
            initial_window
        )
    )

    final_loss = float(
        np.mean(
            final_window
        )
    )

    best_loss = float(
        np.min(history)
    )

    print(
        f"Initial 100-step mean : "
        f"{initial_loss:.6f}"
    )

    print(
        f"Final 100-step mean   : "
        f"{final_loss:.6f}"
    )

    print(
        f"Best training loss    : "
        f"{best_loss:.6f}"
    )

    assert np.isfinite(
        initial_loss
    )

    assert np.isfinite(
        final_loss
    )

    # Require actual learning but avoid an excessively strict
    # threshold because the task is stochastic.
    assert (
        best_loss
        <
        initial_loss
    )

    assert (
        final_loss
        <
        initial_loss
        *
        0.95
    )

    print(
        "[PASS] DDPM noise predictor learned."
    )


# ==============================================================================
# TEST 8 — FINAL STEP HAS NO EXTRA RANDOM NOISE
# ==============================================================================

def test_final_step_determinism(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 8 — FINAL STEP DETERMINISM"
    )
    print("=" * 80)

    x = torch.randn(
        2,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    t = torch.zeros(
        2,
        device=DEVICE,
        dtype=torch.long,
    )

    with torch.no_grad():

        predicted_noise = model(
            x,
            t,
        )

        mean = ddpm_reverse_mean(
            x,
            t,
            predicted_noise,
            schedule,
        )

        torch.manual_seed(123)

        sample_a = (
            ddpm_posterior_sample(
                x,
                t,
                predicted_noise,
                schedule,
            )
        )

        torch.manual_seed(456)

        sample_b = (
            ddpm_posterior_sample(
                x,
                t,
                predicted_noise,
                schedule,
            )
        )

    difference = (
        sample_a
        -
        sample_b
    ).abs().max().item()

    mean_difference = (
        sample_a
        -
        mean
    ).abs().max().item()

    print(
        f"Maximum difference between final samples: "
        f"{difference:.6e}"
    )

    print(
        f"Maximum difference from reverse mean: "
        f"{mean_difference:.6e}"
    )

    assert difference < 1e-7

    assert mean_difference < 1e-7

    print(
        "[PASS] Final x0 step adds no additional random noise."
    )


# ==============================================================================
# TEST 9 — COMPLETE SAMPLER
# ==============================================================================

def test_complete_sampler(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 9 — COMPLETE DDPM SAMPLER"
    )
    print("=" * 80)

    samples, trajectory = ddpm_sample(
        model,
        schedule,
        batch_size=4,
        image_size=IMAGE_SIZE,
        record_timesteps=[
            T,
            750,
            500,
            250,
            100,
            0,
        ],
    )

    print(
        f"Generated shape : "
        f"{tuple(samples.shape)}"
    )

    print(
        f"Generated mean  : "
        f"{samples.mean().item():.6f}"
    )

    print(
        f"Generated std   : "
        f"{samples.std().item():.6f}"
    )

    print(
        f"Generated min   : "
        f"{samples.min().item():.6f}"
    )

    print(
        f"Generated max   : "
        f"{samples.max().item():.6f}"
    )

    print(
        f"Recorded states : "
        f"{sorted(trajectory.keys(), reverse=True)}"
    )

    assert samples.shape == (
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert torch.isfinite(
        samples
    ).all()

    assert len(
        trajectory
    ) >= 5

    print(
        "[PASS] Complete DDPM reverse sampling works."
    )


# ==============================================================================
# TEST 10 — DEVICE CONSISTENCY
# ==============================================================================

def test_device_consistency(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 10 — DEVICE CONSISTENCY"
    )
    print("=" * 80)

    model_device = next(
        model.parameters()
    ).device

    schedule_device = (
        schedule["beta"].device
    )

    x = torch.randn(
        2,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    t = torch.tensor(
        [
            100,
            500,
        ],
        device=DEVICE,
    )

    prediction = model(
        x,
        t,
    )

    print(
        f"Configured device : "
        f"{device_description(DEVICE)}"
    )

    print(
        f"Model device      : "
        f"{device_description(model_device)}"
    )

    print(
        f"Schedule device   : "
        f"{device_description(schedule_device)}"
    )

    print(
        f"Input device      : "
        f"{device_description(x.device)}"
    )

    print(
        f"Prediction device : "
        f"{device_description(prediction.device)}"
    )

    assert devices_match(
        model_device,
        DEVICE,
    )

    assert devices_match(
        schedule_device,
        DEVICE,
    )

    assert devices_match(
        x.device,
        DEVICE,
    )

    assert devices_match(
        prediction.device,
        DEVICE,
    )

    print(
        "[PASS] Device consistency is valid."
    )


# ==============================================================================
# VISUALIZATION — CLEAN / NOISY
# ==============================================================================

@torch.no_grad()
def visualize_forward_process(
    schedule: dict,
):

    x0 = create_image_batch(
        1,
        DEVICE,
    )

    timesteps = [
        1,
        100,
        250,
        500,
        750,
        1000,
    ]

    fig, axes = plt.subplots(
        1,
        len(timesteps),
        figsize=(15, 3),
    )

    for axis, timestep in zip(
        axes,
        timesteps,
    ):

        t = torch.tensor(
            [
                timestep - 1,
            ],
            device=DEVICE,
        )

        xt, _ = q_sample(
            x0,
            t,
            schedule,
        )

        axis.imshow(
            xt[0, 0].cpu().numpy(),
            cmap="gray",
            vmin=-2,
            vmax=2,
        )

        axis.set_title(
            f"t={timestep}"
        )

        axis.axis("off")

    plt.suptitle(
        "Forward Diffusion"
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — TRAINING LOSS
# ==============================================================================

def visualize_training_loss(
    history: list[float],
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history,
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "DDPM Noise-Prediction Training"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — REVERSE TRAJECTORY
# ==============================================================================

def visualize_reverse_trajectory(
    trajectory: dict,
):

    ordered_timesteps = [
        timestep
        for timestep in [
            T,
            750,
            500,
            250,
            100,
            0,
        ]
        if timestep in trajectory
    ]

    fig, axes = plt.subplots(
        1,
        len(ordered_timesteps),
        figsize=(
            3.0
            *
            len(ordered_timesteps),
            3.5,
        ),
    )

    if len(ordered_timesteps) == 1:

        axes = [
            axes
        ]

    for axis, timestep in zip(
        axes,
        ordered_timesteps,
    ):

        image = trajectory[
            timestep
        ][0, 0].numpy()

        axis.imshow(
            image,
            cmap="gray",
        )

        if timestep == 0:

            axis.set_title(
                "Generated x0"
            )

        else:

            axis.set_title(
                f"x{timestep}"
            )

        axis.axis("off")

    plt.suptitle(
        "Reverse DDPM Trajectory"
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — GENERATED SAMPLES
# ==============================================================================

def visualize_generated_samples(
    samples: torch.Tensor,
):

    count = samples.shape[0]

    columns = 4

    rows = math.ceil(
        count / columns
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            10,
            2.7 * rows,
        ),
        squeeze=False,
    )

    for index in range(
        rows * columns
    ):

        axis = axes[
            index // columns
        ][
            index % columns
        ]

        axis.axis(
            "off"
        )

        if index >= count:

            continue

        image = (
            samples[
                index,
                0,
            ]
            .cpu()
            .numpy()
        )

        axis.imshow(
            image,
            cmap="gray",
        )

        axis.set_title(
            f"Sample {index + 1}"
        )

    plt.suptitle(
        "DDPM Generated Samples"
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — PIXEL STATISTICS
# ==============================================================================

def visualize_generated_statistics(
    samples: torch.Tensor,
):

    values = (
        samples
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        values,
        bins=50,
    )

    plt.xlabel(
        "Pixel value"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Generated Image Pixel Distribution"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


# ==============================================================================
# REAL-WORLD EXPERIMENT
# ==============================================================================

def real_world_experiment():

    print()
    print("=" * 100)
    print(
        "REAL-WORLD EXPERIMENT — SYNTHETIC ROBOT CAMERA GENERATION"
    )
    print("=" * 100)

    print(
        """
The DDPM can now be viewed as a synthetic camera-image generator.

Training:

    structured camera-like images
            |
            v
    progressively add noise
            |
            v
    learn noise prediction

Generation:

    random Gaussian image
            |
            v
    reverse DDPM
            |
            v
    structured image

In a robotics system this could eventually provide synthetic
visual observations for:

    - manipulation training
    - perception augmentation
    - simulation
    - rare-object generation
    - visual robustness experiments
"""
    )


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

def print_final_summary(
    history: list[float],
    samples: torch.Tensor,
):

    initial_loss = float(
        np.mean(
            history[:100]
        )
    )

    final_loss = float(
        np.mean(
            history[-100:]
        )
    )

    print()
    print("=" * 80)
    print(
        "LESSON 19 RESULT"
    )
    print("=" * 80)

    print(
        f"""
The complete DDPM pipeline is now implemented.

TRAINING:

    x_0
      |
      v
    choose t
      |
      v
    sample epsilon
      |
      v
    create x_t
      |
      v
    epsilon_theta(x_t,t)
      |
      v
    MSE
      |
      v
    gradient descent


REVERSE GENERATION:

    x_T ~ N(0,I)
      |
      v
    predict epsilon
      |
      v
    calculate reverse mean
      |
      v
    add posterior noise
      |
      v
    x_(T-1)
      |
      v
      ...
      |
      v
    x_0


Training loss:

    Initial 100-step mean:
        {initial_loss:.6f}

    Final 100-step mean:
        {final_loss:.6f}


Generated batch:

    shape:
        {tuple(samples.shape)}

    mean:
        {samples.mean().item():.6f}

    std:
        {samples.std().item():.6f}

The important conceptual transition is:

    Lesson 18:
        "Can the network predict the noise?"

    Lesson 19:
        "Can predicted noise drive the reverse process?"

We now have a complete DDPM.

The current network is intentionally small.

Lesson 20 will improve the neural architecture using a
Diffusion U-Net.
"""
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    set_seed()

    print()
    print("=" * 80)
    print(
        "LESSON 19 — DDPM"
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
        f"T               : "
        f"{T}"
    )

    print(
        f"Image size      : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Training steps  : "
        f"{TRAIN_STEPS}"
    )

    # ==========================================================================
    # PREPARE DDPM SCHEDULE
    # ==========================================================================

    beta = make_beta_schedule(
        timesteps=T,
        device=DEVICE,
    )

    schedule = prepare_ddpm_schedule(
        beta
    )

    # ==========================================================================
    # CREATE MODEL
    # ==========================================================================

    model = NoisePredictor(
        time_embed_dim=TIME_EMBED_DIM,
        base_channels=BASE_CHANNELS,
    ).to(
        DEVICE
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # ==========================================================================
    # TESTS BEFORE TRAINING
    # ==========================================================================

    test_schedule(
        schedule
    )

    test_posterior_variance(
        schedule
    )

    test_reverse_mean(
        schedule
    )

    test_posterior_sample(
        schedule
    )

    test_model_output(
        model,
        schedule,
    )

    test_gradient_flow(
        model,
        schedule,
    )

    test_device_consistency(
        model,
        schedule,
    )

    # ==========================================================================
    # FORWARD PROCESS VISUALIZATION
    # ==========================================================================

    visualize_forward_process(
        schedule
    )

    # ==========================================================================
    # TRAIN
    # ==========================================================================

    history = train_model(
        model,
        schedule,
    )

    # ==========================================================================
    # TRAINING TEST
    # ==========================================================================

    test_training_improvement(
        history
    )

    # ==========================================================================
    # FINAL STEP TEST
    # ==========================================================================

    test_final_step_determinism(
        model,
        schedule,
    )

    # ==========================================================================
    # COMPLETE DDPM SAMPLING
    # ==========================================================================

    print()
    print("=" * 80)
    print(
        "GENERATING DDPM SAMPLES"
    )
    print("=" * 80)

    start = time.perf_counter()

    samples, trajectory = (
        ddpm_sample(
            model,
            schedule,
            batch_size=SAMPLE_COUNT,
            image_size=IMAGE_SIZE,
            record_timesteps=[
                T,
                750,
                500,
                250,
                100,
                0,
            ],
        )
    )

    sampling_time = (
        time.perf_counter()
        -
        start
    )

    print(
        f"Sampling time : "
        f"{sampling_time:.2f} seconds"
    )

    print(
        f"Samples shape : "
        f"{tuple(samples.shape)}"
    )

    # ==========================================================================
    # COMPLETE SAMPLER TEST
    # ==========================================================================

    test_complete_sampler(
        model,
        schedule,
    )

    # ==========================================================================
    # REAL-WORLD EXPERIMENT
    # ==========================================================================

    real_world_experiment()

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================

    print()
    print("=" * 80)
    print(
        "GENERATING VISUALIZATIONS"
    )
    print("=" * 80)

    visualize_training_loss(
        history
    )

    visualize_reverse_trajectory(
        trajectory
    )

    visualize_generated_samples(
        samples
    )

    visualize_generated_statistics(
        samples
    )

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================

    print_final_summary(
        history,
        samples,
    )

    print()
    print("=" * 80)
    print(
        "ALL LESSON 19 TESTS PASSED"
    )
    print("=" * 80)

    print()
    print(
        "Next: Lesson 20 — Diffusion U-Net"
    )

    plt.show()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()