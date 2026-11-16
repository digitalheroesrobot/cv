"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 18 — LEARNING TO PREDICT NOISE
================================================================================

Filename:
    lesson18_learning_to_predict_noise.py

================================================================================
PURPOSE
================================================================================

Lessons 16 and 17 implemented the FORWARD diffusion process.

We learned how to transform:

    clean image x_0

into:

    noisy image x_t

using:

    x_t =
        sqrt(alpha_bar_t) * x_0
        +
        sqrt(1-alpha_bar_t) * epsilon

where:

    epsilon ~ N(0, I)

But this only gives us the forward process.

A generative diffusion model needs to solve the reverse problem:

    Given x_t and t:

        "What noise was added?"

The neural network therefore learns:

    epsilon_theta(x_t, t)

which estimates the original noise:

    epsilon

This lesson implements the FIRST TRAINABLE DENOISING NETWORK
in the course.

The model is intentionally small and educational.

It is NOT a pretrained model.

It is NOT Stable Diffusion.

It is a diffusion noise-prediction network built from random
initialization specifically to understand the mechanism.


================================================================================
WHAT YOU WILL LEARN
================================================================================

By the end of this lesson you will understand:

    1. Why diffusion needs a neural network.
    2. What the training target is.
    3. Why the target is the actual sampled Gaussian noise.
    4. Why timestep t must be provided to the network.
    5. What timestep embedding means.
    6. How sinusoidal timestep embeddings work.
    7. How timestep information enters a CNN.
    8. Why the network predicts noise rather than directly predicting x_0.
    9. The diffusion noise-prediction loss.
   10. How to train the model.
   11. How to evaluate noise prediction.
   12. How noise-prediction quality changes with timestep.
   13. How predicted noise can be converted into an x_0 estimate.
   14. Why this prediction becomes the foundation of DDPM sampling.


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

Current:

    Lesson 18  Learning to Predict Noise

Next:

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

Imagine a robot camera looking at a waste object.

The clean camera image is:

    x_0

The diffusion process intentionally corrupts it:

    x_0 -> x_t

At some timestep t, the network receives:

    x_t
    t

The network must estimate:

    epsilon

that was used to create x_t.

The important point is that the network does NOT receive the
original clean image.

It only sees the corrupted image and the timestep.

Therefore the task is:

    noisy image + amount of noise
                 |
                 v
          neural network
                 |
                 v
          predicted noise


================================================================================
THEORY — FORWARD PROCESS
================================================================================

From Lesson 16:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

where:

    epsilon ~ N(0, I)


================================================================================
THEORY — TRAINING TARGET
================================================================================

During training we know exactly which noise was sampled.

For example:

    epsilon = torch.randn_like(x_0)

Then:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

Therefore we have a supervised learning pair:

    INPUT:

        x_t
        t

    TARGET:

        epsilon

The neural network learns:

    epsilon_theta(x_t,t)


================================================================================
MATHEMATICS — NOISE PREDICTION LOSS
================================================================================

The basic DDPM training objective is:

    L =
        E[
            ||epsilon
            -
            epsilon_theta(x_t,t)||^2
        ]

This is mean squared error:

    MSE =
        mean(
            (epsilon - epsilon_pred)^2
        )

The model is therefore trained using ordinary gradient descent.

This is an important conceptual point:

    DIFFUSION TRAINING IS SUPERVISED LEARNING

because we construct the target noise ourselves.


================================================================================
WHY PREDICT NOISE?
================================================================================

We could theoretically train a network to predict x_0.

But noise prediction has several useful properties.

The forward process always uses:

    epsilon ~ N(0,I)

Therefore the target distribution is standardized.

The network learns the transformation:

    x_t + t
        |
        v
    epsilon

Once epsilon is predicted, we can estimate the original image.

Starting from:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

solve for x_0:

    x_0 =
        [
            x_t
            -
            sqrt(1-alpha_bar_t) epsilon
        ]
        /
        sqrt(alpha_bar_t)

Therefore:

    x_0_pred =
        [
            x_t
            -
            sqrt(1-alpha_bar_t)
            epsilon_theta(x_t,t)
        ]
        /
        sqrt(alpha_bar_t)


================================================================================
WHY DOES THE NETWORK NEED t?
================================================================================

The same noisy-looking image can mean very different things
at different timesteps.

For example:

    t = 50

means:

    the image is only mildly corrupted.

But:

    t = 900

means:

    the image is heavily corrupted.

The network therefore needs to know:

    HOW MUCH DIFFUSION HAS OCCURRED?


================================================================================
SINUSOIDAL TIMESTEP EMBEDDING
================================================================================

A common method is to transform the scalar timestep into a vector.

Instead of giving the network only:

    t = 500

we construct:

    embedding(t)

using sine and cosine functions.

For dimension i:

    PE(t,2i) =
        sin(
            t / 10000^(2i/d)
        )

    PE(t,2i+1) =
        cos(
            t / 10000^(2i/d)
        )

This produces a smooth representation of time.

The network can then learn different behavior for different
diffusion stages.


================================================================================
NETWORK ARCHITECTURE
================================================================================

This lesson intentionally uses a small CNN.

Input:

    x_t
        shape:
        [B, 1, H, W]

Timestep:

    t
        shape:
        [B]

Timestep embedding:

    [B, TIME_EMBED_DIM]

Network:

    x_t
       |
       v
    Conv
       |
       v
    ReLU
       |
       v
    Conv
       |
       +------ timestep embedding
       |
       v
    ReLU
       |
       v
    Conv
       |
       v
    predicted epsilon

The timestep embedding is projected into feature channels and
added to the intermediate feature map.


================================================================================
WHY NOT A FULL U-NET YET?
================================================================================

A full diffusion U-Net contains:

    - encoder blocks
    - decoder blocks
    - skip connections
    - residual blocks
    - timestep embeddings
    - attention

That architecture will be introduced later.

This lesson deliberately isolates the most important concept:

    LEARN TO PREDICT THE NOISE.

Lesson 20 will introduce the more powerful diffusion U-Net.


================================================================================
DATA GENERATION
================================================================================

We do not need a large stored dataset for this lesson.

Instead we generate structured synthetic images representing
simple camera observations.

Each training example:

    1. Create clean image x_0.
    2. Randomly choose timestep t.
    3. Sample epsilon.
    4. Construct x_t.
    5. Train the model to predict epsilon.

This allows us to focus on diffusion mathematics rather than
dataset management.


================================================================================
REAL-WORLD ANALOGY
================================================================================

Imagine deliberately adding known sensor corruption to a robot
camera image.

You know exactly which random disturbance was added.

Therefore:

    corrupted image
        +
    corruption level
        |
        v
    neural network
        |
        v
    estimated corruption


This is analogous to learning how to identify and remove
structured uncertainty from visual observations.


================================================================================
IMPLEMENTATION
================================================================================

This file implements:

    - linear diffusion schedule
    - sinusoidal timestep embeddings
    - small CNN noise predictor
    - random timestep training
    - epsilon prediction
    - MSE loss
    - training loop
    - validation
    - gradient verification
    - timestep-wise evaluation
    - x_0 reconstruction from predicted noise
    - visualization
    - real-world experiment


================================================================================
VERIFICATION / TESTS
================================================================================

The script verifies:

    TEST 1
        Timestep embedding shape.

    TEST 2
        Timestep embedding is finite.

    TEST 3
        Model output shape matches input.

    TEST 4
        Model parameters receive gradients.

    TEST 5
        Loss is finite.

    TEST 6
        Training reduces validation noise-prediction error.

    TEST 7
        Predicted noise can reconstruct x_0.

    TEST 8
        Different timesteps produce valid predictions.

    TEST 9
        Model and tensors remain on the selected device.


================================================================================
VISUALIZATION
================================================================================

The script visualizes:

    - training loss
    - validation loss
    - true noise
    - predicted noise
    - prediction error
    - reconstructed x_0
    - clean x_0
    - noisy x_t
    - timestep-dependent MSE


================================================================================
EXPERIMENT
================================================================================

After training, the model is evaluated at:

    t = 50
    t = 100
    t = 250
    t = 500
    t = 750
    t = 900

We compare:

    true epsilon

against:

    predicted epsilon


================================================================================
REAL-WORLD APPLICATION
================================================================================

For a robot vision system, this mechanism can eventually be used
as a learned visual prior.

For example:

    camera image
        |
        v
    diffusion representation
        |
        v
    learned denoising
        |
        v
    recovered visual structure

This is relevant to:

    - image restoration
    - visual generation
    - inpainting
    - synthetic data generation
    - visual perception under uncertainty
    - robot scene generation


================================================================================
REAL-WORLD RESULT
================================================================================

The immediate result is not yet image generation.

The immediate result is:

    A neural network that has learned the direction of the
    corruption process.

That is the critical missing component between:

    forward diffusion

and:

    reverse diffusion.


================================================================================
LESSON RESULT
================================================================================

We have now implemented the first trainable diffusion model.

The complete learning loop is:

    clean image x_0
          |
          v
    choose timestep t
          |
          v
    sample epsilon
          |
          v
    construct x_t
          |
          v
    network(x_t,t)
          |
          v
    predicted epsilon
          |
          v
    MSE(predicted epsilon, epsilon)
          |
          v
    gradient descent


================================================================================
CONNECTION TO NEXT LESSON
================================================================================

Lesson 19 will use the learned noise prediction to implement
the DDPM reverse process.

The progression is now:

    Lesson 16
        Forward diffusion.

    Lesson 17
        Noise schedules.

    Lesson 18
        Learn epsilon_theta(x_t,t).

    Lesson 19
        Use epsilon_theta to perform reverse diffusion.

The key transition is:

    FORWARD:

        x_0 -> x_t

    LEARN:

        x_t,t -> epsilon

    REVERSE:

        x_t -> x_{t-1}

Lesson 19 will connect these three pieces into an actual
generative diffusion process.


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
# GLOBAL CONFIGURATION
# ==============================================================================

SEED = 42

T = 1000

IMAGE_SIZE = 32

BETA_START = 1e-4
BETA_END = 0.02

BATCH_SIZE = 64

TRAIN_STEPS = 1800

VALIDATION_STEPS = 150

LEARNING_RATE = 2e-3

TIME_EMBED_DIM = 64

BASE_CHANNELS = 32

TEST_DEVICE = torch.device("cpu")

TEST_DTYPE = torch.float64

EPS = 1e-8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
# SYNTHETIC CAMERA IMAGE GENERATION
# ==============================================================================

def create_structured_image(
    height: int = IMAGE_SIZE,
    width: int = IMAGE_SIZE,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create one structured grayscale image.

    The image contains:

        - smooth background
        - rectangle
        - circle
        - diagonal structure
        - sinusoidal texture

    Output:

        [1, H, W]
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

    # Randomly vary the background slightly so the model cannot
    # simply memorize one exact image.
    background_bias = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.06
    )

    image = (
        0.12
        +
        background_bias
        +
        0.12 * xx
        +
        0.08 * yy
    )

    # Random rectangle.
    rx1 = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.45
        + 0.05
    )

    ry1 = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.45
        + 0.05
    )

    rw = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.25
        + 0.15
    )

    rh = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.25
        + 0.15
    )

    rectangle = (
        (xx > rx1)
        &
        (xx < rx1 + rw)
        &
        (yy > ry1)
        &
        (yy < ry1 + rh)
    )

    image = (
        image
        +
        rectangle.to(dtype)
        *
        0.45
    )

    # Random circle.
    cx = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.70
        + 0.15
    )

    cy = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.70
        + 0.15
    )

    radius = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.10
        + 0.07
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

    # Random diagonal structure.
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

    # Low-amplitude texture.
    texture = (
        0.025
        *
        torch.sin(
            2.0
            * math.pi
            * 5.0
            * xx
        )
        *
        torch.sin(
            2.0
            * math.pi
            * 4.0
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
            dtype=torch.float32,
        )
        for _ in range(batch_size)
    ]

    return torch.stack(
        images,
        dim=0,
    )


# ==============================================================================
# DIFFUSION SCHEDULE
# ==============================================================================

def make_linear_beta_schedule(
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

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar":
            sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar":
            sqrt_one_minus_alpha_bar,
        "snr": snr,
    }


# ==============================================================================
# FORWARD DIFFUSION
# ==============================================================================

def extract_schedule_values(
    values: torch.Tensor,
    t: torch.Tensor,
    x_shape: torch.Size,
) -> torch.Tensor:
    """
    Select one schedule value for every batch item.

    Input:

        values: [T]
        t:      [B]

    Output:

        [B, 1, 1, 1]
    """

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


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: dict,
    noise: torch.Tensor | None = None,
):
    """
    Construct x_t directly.

        x_t =
            sqrt(alpha_bar_t) x_0
            +
            sqrt(1-alpha_bar_t) epsilon
    """

    if noise is None:

        noise = torch.randn_like(
            x0
        )

    sqrt_alpha_bar = (
        extract_schedule_values(
            schedule[
                "sqrt_alpha_bar"
            ],
            t,
            x0.shape,
        )
        .to(
            device=x0.device,
            dtype=x0.dtype,
        )
    )

    sqrt_one_minus_alpha_bar = (
        extract_schedule_values(
            schedule[
                "sqrt_one_minus_alpha_bar"
            ],
            t,
            x0.shape,
        )
        .to(
            device=x0.device,
            dtype=x0.dtype,
        )
    )

    xt = (
        sqrt_alpha_bar * x0
        +
        sqrt_one_minus_alpha_bar
        * noise
    )

    return xt, noise


# ==============================================================================
# SINUSOIDAL TIMESTEP EMBEDDING
# ==============================================================================

def sinusoidal_timestep_embedding(
    t: torch.Tensor,
    embedding_dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """
    Convert integer timestep t into a continuous vector.

    t:

        [B]

    output:

        [B, embedding_dim]

    Mathematical form:

        sin(t * frequency)
        cos(t * frequency)

    The embedding gives the network a representation of diffusion
    time rather than only a single scalar.
    """

    if embedding_dim % 2 != 0:

        raise ValueError(
            "embedding_dim must be even"
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

    embedding = torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=1,
    )

    return embedding


# ==============================================================================
# MODEL
# ==============================================================================

class NoisePredictor(nn.Module):
    """
    Small diffusion noise-prediction CNN.

    Input:

        x_t:
            [B, 1, H, W]

        t:
            [B]

    Output:

        predicted epsilon:
            [B, 1, H, W]

    Architecture:

        Conv
        ReLU
        Conv

        + timestep embedding

        ReLU
        Conv
        ReLU
        Conv

    The model keeps spatial resolution constant.
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

        self.base_channels = (
            base_channels
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

        self._initialize_weights()

    def _initialize_weights(self):

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

        # ----------------------------------------------------------------------
        # IMAGE FEATURES
        # ----------------------------------------------------------------------

        h = self.activation(
            self.conv1(x)
        )

        h = self.activation(
            self.conv2(h)
        )

        # ----------------------------------------------------------------------
        # TIMESTEP FEATURES
        # ----------------------------------------------------------------------

        t_embedding = (
            sinusoidal_timestep_embedding(
                t,
                self.time_embed_dim,
            )
        )

        t_embedding = (
            self.time_mlp(
                t_embedding
            )
        )

        # [B,C] -> [B,C,1,1]

        t_embedding = (
            t_embedding
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        # ----------------------------------------------------------------------
        # CONDITION IMAGE FEATURES ON TIMESTEP
        # ----------------------------------------------------------------------

        h = h + t_embedding

        h = self.activation(
            self.conv3(h)
        )

        output = self.conv4(h)

        return output


# ==============================================================================
# MODEL PARAMETER COUNT
# ==============================================================================

def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ==============================================================================
# TEST 1 — TIMESTEP EMBEDDING
# ==============================================================================

def test_timestep_embedding():

    print()
    print("=" * 80)
    print(
        "TEST 1 — TIMESTEP EMBEDDING"
    )
    print("=" * 80)

    t = torch.tensor(
        [
            0,
            1,
            100,
            500,
            999,
        ],
        dtype=torch.long,
    )

    embedding = (
        sinusoidal_timestep_embedding(
            t,
            TIME_EMBED_DIM,
        )
    )

    print(
        f"Input shape  : {tuple(t.shape)}"
    )

    print(
        f"Output shape : {tuple(embedding.shape)}"
    )

    print(
        f"Min          : "
        f"{embedding.min().item():.6f}"
    )

    print(
        f"Max          : "
        f"{embedding.max().item():.6f}"
    )

    assert embedding.shape == (
        5,
        TIME_EMBED_DIM,
    )

    assert torch.isfinite(
        embedding
    ).all()

    assert embedding.abs().max() <= 1.0

    print(
        "[PASS] Timestep embedding."
    )


# ==============================================================================
# TEST 2 — MODEL OUTPUT SHAPE
# ==============================================================================

def test_model_output_shape(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 2 — MODEL OUTPUT SHAPE"
    )
    print("=" * 80)

    x0 = create_image_batch(
        4,
        device=DEVICE,
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
        f"x_t shape         : "
        f"{tuple(xt.shape)}"
    )

    print(
        f"noise shape        : "
        f"{tuple(noise.shape)}"
    )

    print(
        f"prediction shape   : "
        f"{tuple(prediction.shape)}"
    )

    assert prediction.shape == (
        x0.shape
    )

    assert torch.isfinite(
        prediction
    ).all()

    print(
        "[PASS] Model output shape."
    )


# ==============================================================================
# TEST 3 — LOSS FINITENESS
# ==============================================================================

def test_loss_finiteness(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 3 — LOSS FINITENESS"
    )
    print("=" * 80)

    x0 = create_image_batch(
        8,
        device=DEVICE,
    )

    t = torch.randint(
        0,
        T,
        (
            8,
        ),
        device=DEVICE,
    )

    xt, noise = q_sample(
        x0,
        t,
        schedule,
    )

    prediction = model(
        xt,
        t,
    )

    loss = F.mse_loss(
        prediction,
        noise,
    )

    print(
        f"Initial loss : "
        f"{loss.item():.6f}"
    )

    assert torch.isfinite(
        loss
    )

    print(
        "[PASS] Noise-prediction loss is finite."
    )


# ==============================================================================
# TEST 4 — GRADIENT FLOW
# ==============================================================================

def test_gradient_flow(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 4 — GRADIENT FLOW"
    )
    print("=" * 80)

    model.zero_grad(
        set_to_none=True
    )

    x0 = create_image_batch(
        8,
        device=DEVICE,
    )

    t = torch.randint(
        0,
        T,
        (
            8,
        ),
        device=DEVICE,
    )

    xt, noise = q_sample(
        x0,
        t,
        schedule,
    )

    prediction = model(
        xt,
        t,
    )

    loss = F.mse_loss(
        prediction,
        noise,
    )

    loss.backward()

    parameter_count = 0

    parameters_with_grad = 0

    maximum_gradient = 0.0

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        parameter_count += 1

        if parameter.grad is not None:

            parameters_with_grad += 1

            maximum_gradient = max(
                maximum_gradient,
                parameter.grad.abs().max().item(),
            )

    print(
        f"Trainable tensors       : "
        f"{parameter_count}"
    )

    print(
        f"Tensors with gradients  : "
        f"{parameters_with_grad}"
    )

    print(
        f"Maximum gradient        : "
        f"{maximum_gradient:.6e}"
    )

    assert (
        parameters_with_grad
        ==
        parameter_count
    )

    assert maximum_gradient > 0.0

    print(
        "[PASS] Gradient flows through the complete model."
    )


# ==============================================================================
# DEVICE TEST
# ==============================================================================

def test_device_consistency(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 5 — DEVICE CONSISTENCY"
    )
    print("=" * 80)

    model_device = next(
        model.parameters()
    ).device

    schedule_device = (
        schedule["beta"].device
    )

    x0 = create_image_batch(
        2,
        device=DEVICE,
    )

    t = torch.tensor(
        [
            100,
            500,
        ],
        device=DEVICE,
    )

    xt, noise = q_sample(
        x0,
        t,
        schedule,
    )

    prediction = model(
        xt,
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
        f"x_t device        : "
        f"{device_description(xt.device)}"
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
        xt.device,
        DEVICE,
    )

    assert devices_match(
        prediction.device,
        DEVICE,
    )

    print(
        "[PASS] Model and diffusion tensors use compatible devices."
    )


# ==============================================================================
# LOSS FUNCTION
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
        predicted_noise,
        noise,
        xt,
        t,
    )


# ==============================================================================
# TRAINING STEP
# ==============================================================================

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    schedule: dict,
):

    optimizer.zero_grad(
        set_to_none=True
    )

    (
        loss,
        predicted_noise,
        noise,
        xt,
        t,
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

    return (
        loss.item(),
        predicted_noise.detach(),
        noise.detach(),
        xt.detach(),
        t.detach(),
    )


# ==============================================================================
# VALIDATION
# ==============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    schedule: dict,
    steps: int = VALIDATION_STEPS,
):

    model.eval()

    losses = []

    for _ in range(steps):

        x0 = create_image_batch(
            BATCH_SIZE,
            device=DEVICE,
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

        losses.append(
            loss.item()
        )

    model.train()

    return float(
        np.mean(losses)
    )


# ==============================================================================
# TRAINING LOOP
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

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    print()
    print("=" * 80)
    print(
        "TRAINING NOISE PREDICTOR"
    )
    print("=" * 80)

    print(
        f"Training steps  : {TRAIN_STEPS}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"Learning rate   : {LEARNING_RATE}"
    )

    print(
        f"Parameters      : "
        f"{count_parameters(model):,}"
    )

    start_time = time.perf_counter()

    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):

        x0 = create_image_batch(
            BATCH_SIZE,
            device=DEVICE,
        )

        (
            train_loss,
            _,
            _,
            _,
            _,
        ) = train_step(
            model,
            optimizer,
            x0,
            schedule,
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        # Validation is intentionally performed periodically.
        if (
            step == 1
            or step % 300 == 0
            or step == TRAIN_STEPS
        ):

            validation_loss = (
                evaluate_model(
                    model,
                    schedule,
                    steps=30,
                )
            )

            history[
                "validation_loss"
            ].append(
                (
                    step,
                    validation_loss,
                )
            )

            print(
                f"Step "
                f"{step:5d}/{TRAIN_STEPS} | "
                f"train={train_loss:.6f} | "
                f"validation={validation_loss:.6f}"
            )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    print()
    print(
        f"Training time: "
        f"{elapsed:.2f} seconds"
    )

    return history


# ==============================================================================
# TEST 6 — TRAINING IMPROVEMENT
# ==============================================================================

def test_training_improvement(
    history: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 6 — TRAINING IMPROVEMENT"
    )
    print("=" * 80)

    validation_history = (
        history["validation_loss"]
    )

    assert len(
        validation_history
    ) >= 2

    first_validation = (
        validation_history[0][1]
    )

    final_validation = (
        validation_history[-1][1]
    )

    best_validation = min(
        value
        for _, value
        in validation_history
    )

    print(
        f"First validation loss : "
        f"{first_validation:.6f}"
    )

    print(
        f"Best validation loss  : "
        f"{best_validation:.6f}"
    )

    print(
        f"Final validation loss : "
        f"{final_validation:.6f}"
    )

    # We require meaningful learning, but do not demand that the
    # final point is the absolute best point because stochastic
    # training naturally fluctuates.
    improvement = (
        first_validation
        -
        best_validation
    )

    print(
        f"Improvement            : "
        f"{improvement:.6f}"
    )

    assert (
        best_validation
        <
        first_validation
    )

    print(
        "[PASS] The network learned to predict noise."
    )


# ==============================================================================
# EVALUATE AT FIXED TIMESTEPS
# ==============================================================================

@torch.no_grad()
def evaluate_by_timestep(
    model: nn.Module,
    schedule: dict,
    timesteps: list[int],
    batches_per_timestep: int = 20,
):

    model.eval()

    results = {}

    for timestep in timesteps:

        timestep_index_value = (
            timestep - 1
        )

        losses = []

        correlations = []

        for _ in range(
            batches_per_timestep
        ):

            x0 = create_image_batch(
                BATCH_SIZE,
                device=DEVICE,
            )

            t = torch.full(
                (
                    BATCH_SIZE,
                ),
                timestep_index_value,
                device=DEVICE,
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

            losses.append(
                loss.item()
            )

            prediction_flat = (
                predicted_noise.flatten(
                    start_dim=1
                )
            )

            noise_flat = (
                noise.flatten(
                    start_dim=1
                )
            )

            prediction_centered = (
                prediction_flat
                -
                prediction_flat.mean(
                    dim=1,
                    keepdim=True,
                )
            )

            noise_centered = (
                noise_flat
                -
                noise_flat.mean(
                    dim=1,
                    keepdim=True,
                )
            )

            numerator = (
                (
                    prediction_centered
                    *
                    noise_centered
                )
                .sum(dim=1)
            )

            denominator = (
                torch.sqrt(
                    (
                        prediction_centered
                        ** 2
                    ).sum(dim=1)
                    *
                    (
                        noise_centered
                        ** 2
                    ).sum(dim=1)
                )
                +
                EPS
            )

            correlation = (
                numerator
                /
                denominator
            )

            correlations.extend(
                correlation
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

        results[timestep] = {
            "mse":
                float(
                    np.mean(losses)
                ),

            "correlation":
                float(
                    np.mean(
                        correlations
                    )
                ),
        }

    model.train()

    return results


# ==============================================================================
# TEST 7 — X0 RECONSTRUCTION
# ==============================================================================

@torch.no_grad()
def reconstruct_x0_from_prediction(
    xt: torch.Tensor,
    predicted_noise: torch.Tensor,
    t: torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    """
    Recover x_0 estimate using:

        x_0 =
            [
                x_t
                -
                sqrt(1-alpha_bar_t)
                * epsilon_pred
            ]
            /
            sqrt(alpha_bar_t)
    """

    sqrt_alpha_bar = (
        extract_schedule_values(
            schedule[
                "sqrt_alpha_bar"
            ],
            t,
            xt.shape,
        )
        .to(
            device=xt.device,
            dtype=xt.dtype,
        )
    )

    sqrt_one_minus_alpha_bar = (
        extract_schedule_values(
            schedule[
                "sqrt_one_minus_alpha_bar"
            ],
            t,
            xt.shape,
        )
        .to(
            device=xt.device,
            dtype=xt.dtype,
        )
    )

    x0_prediction = (
        xt
        -
        sqrt_one_minus_alpha_bar
        *
        predicted_noise
    ) / (
        sqrt_alpha_bar
        +
        EPS
    )

    return x0_prediction


@torch.no_grad()
def test_x0_reconstruction(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 7 — X0 RECONSTRUCTION FROM PREDICTED NOISE"
    )
    print("=" * 80)

    timesteps = [
        50,
        100,
        250,
        500,
    ]

    for timestep in timesteps:

        x0 = create_image_batch(
            16,
            device=DEVICE,
        )

        t = torch.full(
            (
                16,
            ),
            timestep - 1,
            device=DEVICE,
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

        x0_prediction = (
            reconstruct_x0_from_prediction(
                xt,
                predicted_noise,
                t,
                schedule,
            )
        )

        reconstruction_mse = F.mse_loss(
            x0_prediction,
            x0,
        ).item()

        oracle_x0 = (
            reconstruct_x0_from_prediction(
                xt,
                noise,
                t,
                schedule,
            )
        )

        oracle_error = F.mse_loss(
            oracle_x0,
            x0,
        ).item()

        print(
            f"t={timestep:4d} | "
            f"predicted-x0 MSE="
            f"{reconstruction_mse:.6f} | "
            f"oracle MSE="
            f"{oracle_error:.6e}"
        )

        # Oracle reconstruction should be essentially exact.
        assert oracle_error < 1e-5

        assert torch.isfinite(
            x0_prediction
        ).all()

    print(
        "[PASS] Predicted noise can be converted into an x0 estimate."
    )


# ==============================================================================
# TEST 8 — MULTIPLE TIMESTEPS
# ==============================================================================

def test_multiple_timesteps(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 8 — MULTIPLE TIMESTEPS"
    )
    print("=" * 80)

    timesteps = [
        1,
        50,
        250,
        500,
        750,
        999,
        1000,
    ]

    for timestep in timesteps:

        x0 = create_image_batch(
            4,
            device=DEVICE,
        )

        t = torch.full(
            (
                4,
            ),
            timestep - 1,
            device=DEVICE,
            dtype=torch.long,
        )

        xt, _ = q_sample(
            x0,
            t,
            schedule,
        )

        prediction = model(
            xt,
            t,
        )

        assert prediction.shape == (
            x0.shape
        )

        assert torch.isfinite(
            prediction
        ).all()

        print(
            f"t={timestep:4d} | "
            f"prediction std="
            f"{prediction.std().item():.6f}"
        )

    print(
        "[PASS] Network operates across the complete timestep range."
    )


# ==============================================================================
# VISUALIZATION — TRAINING CURVE
# ==============================================================================

def plot_training_history(
    history: dict,
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["train_loss"],
        label="Training loss",
    )

    validation_steps = [
        step
        for step, _ in
        history["validation_loss"]
    ]

    validation_values = [
        value
        for _, value in
        history["validation_loss"]
    ]

    plt.plot(
        validation_steps,
        validation_values,
        marker="o",
        label="Validation loss",
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "Diffusion Noise-Prediction Training"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — TIMESTEP RESULTS
# ==============================================================================

def plot_timestep_results(
    results: dict,
):

    timesteps = list(
        results.keys()
    )

    mse_values = [
        results[t]["mse"]
        for t in timesteps
    ]

    correlation_values = [
        results[t]["correlation"]
        for t in timesteps
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        timesteps,
        mse_values,
        marker="o",
    )

    plt.xlabel(
        "Diffusion timestep"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "Noise Prediction Error vs Timestep"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        timesteps,
        correlation_values,
        marker="o",
    )

    plt.xlabel(
        "Diffusion timestep"
    )

    plt.ylabel(
        "Prediction / target correlation"
    )

    plt.title(
        "Noise Prediction Correlation vs Timestep"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — NOISE PREDICTION
# ==============================================================================

@torch.no_grad()
def visualize_noise_prediction(
    model: nn.Module,
    schedule: dict,
    timestep: int = 500,
):

    x0 = create_image_batch(
        1,
        device=DEVICE,
    )

    t = torch.tensor(
        [
            timestep - 1,
        ],
        device=DEVICE,
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

    x0_prediction = (
        reconstruct_x0_from_prediction(
            xt,
            predicted_noise,
            t,
            schedule,
        )
    )

    prediction_error = (
        predicted_noise
        -
        noise
    )

    images = [
        x0[0, 0].cpu().numpy(),
        xt[0, 0].cpu().numpy(),
        noise[0, 0].cpu().numpy(),
        predicted_noise[0, 0].cpu().numpy(),
        prediction_error[0, 0].cpu().numpy(),
        x0_prediction[0, 0].cpu().numpy(),
    ]

    titles = [
        "Clean x0",
        f"Noisy xt (t={timestep})",
        "True epsilon",
        "Predicted epsilon",
        "Prediction error",
        "Recovered x0",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
    )

    for axis, image, title in zip(
        axes.flatten(),
        images,
        titles,
    ):

        axis.imshow(
            image,
            cmap="gray",
        )

        axis.set_title(
            title
        )

        axis.axis("off")

    plt.suptitle(
        "Diffusion Noise Prediction"
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — X0 RECONSTRUCTION OVER TIME
# ==============================================================================

@torch.no_grad()
def visualize_reconstruction_over_time(
    model: nn.Module,
    schedule: dict,
    timesteps: list[int],
):

    x0 = create_image_batch(
        1,
        device=DEVICE,
    )

    columns = len(timesteps)

    fig, axes = plt.subplots(
        3,
        columns,
        figsize=(
            3.2 * columns,
            9.0,
        ),
        squeeze=False,
    )

    for column, timestep in enumerate(
        timesteps
    ):

        t = torch.tensor(
            [
                timestep - 1,
            ],
            device=DEVICE,
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

        x0_prediction = (
            reconstruct_x0_from_prediction(
                xt,
                predicted_noise,
                t,
                schedule,
            )
        )

        axes[
            0,
            column
        ].imshow(
            xt[0, 0].cpu().numpy(),
            cmap="gray",
        )

        axes[
            0,
            column
        ].set_title(
            f"x_t, t={timestep}"
        )

        axes[
            1,
            column
        ].imshow(
            predicted_noise[
                0,
                0
            ].cpu().numpy(),
            cmap="gray",
        )

        axes[
            1,
            column
        ].set_title(
            "Predicted noise"
        )

        axes[
            2,
            column
        ].imshow(
            x0_prediction[
                0,
                0
            ].cpu().numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        axes[
            2,
            column
        ].set_title(
            "Predicted x0"
        )

    for axis in axes.flatten():

        axis.axis("off")

    plt.suptitle(
        "Noise Prediction and x0 Reconstruction"
    )

    plt.tight_layout()


# ==============================================================================
# REAL-WORLD EXPERIMENT
# ==============================================================================

def real_world_experiment(
    timestep_results: dict,
):

    print()
    print("=" * 100)
    print(
        "REAL-WORLD EXPERIMENT — ROBOT CAMERA NOISE ESTIMATION"
    )
    print("=" * 100)

    print(
        """
Imagine a robot camera observing an object.

The diffusion system artificially corrupts the observation.

The neural network receives:

    noisy camera-like image
    +
    diffusion timestep

and estimates:

    the corruption/noise pattern.

A low noise-prediction MSE means the model has learned a useful
representation of the corruption process.

This is the first learned component required for reverse diffusion.
"""
    )

    print(
        f"{'t':>8} | "
        f"{'Noise MSE':>14} | "
        f"{'Correlation':>14}"
    )

    print("-" * 45)

    for timestep, values in (
        timestep_results.items()
    ):

        print(
            f"{timestep:8d} | "
            f"{values['mse']:14.6f} | "
            f"{values['correlation']:14.6f}"
        )


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

def print_final_summary(
    model: nn.Module,
    history: dict,
    timestep_results: dict,
):

    initial_loss = (
        history[
            "validation_loss"
        ][0][1]
    )

    best_loss = min(
        value
        for _, value
        in history[
            "validation_loss"
        ]
    )

    print()
    print("=" * 80)
    print(
        "LESSON 18 RESULT"
    )
    print("=" * 80)

    print(
        f"""
The first trainable diffusion noise predictor is now implemented.

Model parameters:

    {count_parameters(model):,}

The training target was:

    epsilon ~ N(0,I)

The model learned:

    epsilon_theta(x_t,t)

The training objective was:

    MSE(
        epsilon_theta(x_t,t),
        epsilon
    )

Validation:

    Initial validation loss:
        {initial_loss:.6f}

    Best validation loss:
        {best_loss:.6f}

The complete learned pipeline is:

    x_0
      |
      | sample t
      v
    forward diffusion
      |
      v
    x_t + t
      |
      v
    neural network
      |
      v
    predicted epsilon
      |
      v
    estimate x_0

This is the central learning mechanism of DDPM.

We have NOT performed reverse diffusion yet.

That is the job of Lesson 19.
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
        "LESSON 18 — LEARNING TO PREDICT NOISE"
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
    # SCHEDULE
    # ==========================================================================

    beta = make_linear_beta_schedule(
        timesteps=T,
        device=DEVICE,
    )

    schedule = prepare_schedule(
        beta
    )

    # ==========================================================================
    # MODEL
    # ==========================================================================

    model = NoisePredictor(
        time_embed_dim=TIME_EMBED_DIM,
        base_channels=BASE_CHANNELS,
    ).to(
        DEVICE
    )

    print()
    print(
        "Model:"
    )

    print(
        model
    )

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ==========================================================================
    # PRE-TRAINING TESTS
    # ==========================================================================

    test_timestep_embedding()

    test_model_output_shape(
        model,
        schedule,
    )

    test_loss_finiteness(
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
    # TIMESTEP EVALUATION
    # ==========================================================================

    evaluation_timesteps = [
        50,
        100,
        250,
        500,
        750,
        900,
    ]

    timestep_results = (
        evaluate_by_timestep(
            model,
            schedule,
            evaluation_timesteps,
            batches_per_timestep=12,
        )
    )

    print()
    print("=" * 80)
    print(
        "TIMESTEP-WISE NOISE PREDICTION"
    )
    print("=" * 80)

    print(
        f"{'t':>8} | "
        f"{'MSE':>14} | "
        f"{'Correlation':>14}"
    )

    print("-" * 45)

    for timestep, values in (
        timestep_results.items()
    ):

        print(
            f"{timestep:8d} | "
            f"{values['mse']:14.6f} | "
            f"{values['correlation']:14.6f}"
        )

    # ==========================================================================
    # X0 RECONSTRUCTION
    # ==========================================================================

    test_x0_reconstruction(
        model,
        schedule,
    )

    # ==========================================================================
    # MULTIPLE TIMESTEP TEST
    # ==========================================================================

    test_multiple_timesteps(
        model,
        schedule,
    )

    # ==========================================================================
    # REAL-WORLD EXPERIMENT
    # ==========================================================================

    real_world_experiment(
        timestep_results
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

    plot_training_history(
        history
    )

    plot_timestep_results(
        timestep_results
    )

    visualize_noise_prediction(
        model,
        schedule,
        timestep=500,
    )

    visualize_reconstruction_over_time(
        model,
        schedule,
        timesteps=[
            50,
            250,
            500,
            750,
        ],
    )

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================

    print_final_summary(
        model,
        history,
        timestep_results,
    )

    print()
    print("=" * 80)
    print(
        "ALL LESSON 18 TESTS PASSED"
    )
    print("=" * 80)

    print()
    print(
        "Next: Lesson 19 — DDPM"
    )

    plt.show()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()