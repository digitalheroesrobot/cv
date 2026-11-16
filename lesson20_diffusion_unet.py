"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 20 — DIFFUSION U-NET
================================================================================

Filename:
    lesson20_diffusion_unet.py

================================================================================
PURPOSE
================================================================================

Lesson 19 implemented the complete DDPM mathematical pipeline.

We had:

    x_T
     |
     v
    noise predictor
     |
     v
    reverse DDPM
     |
     v
    x_0

However, the neural network used in Lesson 19 was intentionally
small.

Real diffusion systems need a much stronger image architecture.

This lesson introduces the Diffusion U-Net.

The U-Net will provide:

    - multi-scale image features
    - downsampling
    - upsampling
    - skip connections
    - residual blocks
    - timestep conditioning
    - spatial detail preservation

The DDPM mathematics does NOT fundamentally change.

We replace the neural network:

    small CNN

with:

    diffusion U-Net


================================================================================
WHAT YOU WILL LEARN
================================================================================

You will learn:

    1. Why a simple CNN is limited for diffusion.
    2. What a U-Net architecture is.
    3. Encoder / bottleneck / decoder structure.
    4. Why downsampling helps.
    5. Why upsampling helps.
    6. Why skip connections are essential.
    7. What residual blocks do.
    8. How timestep embeddings enter the network.
    9. How a U-Net predicts diffusion noise.
   10. How the U-Net connects to DDPM.
   11. How to train the U-Net using epsilon prediction.
   12. How to use the trained U-Net in reverse diffusion.


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
    Lesson 16  Forward diffusion
    Lesson 17  Noise schedules
    Lesson 18  Noise prediction
    Lesson 19  DDPM
    Lesson 20  Diffusion U-Net

Next:

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

A diffusion model must simultaneously understand:

    GLOBAL STRUCTURE

and:

    LOCAL DETAIL

For example, consider a robot camera image.

Global information:

    - object position
    - overall shape
    - scene layout

Local information:

    - edges
    - corners
    - texture
    - small object details

A single-resolution CNN has difficulty maintaining both types
of information efficiently.

A U-Net solves this by processing the image at multiple scales.


================================================================================
U-NET IDEA
================================================================================

The architecture looks conceptually like:

                       timestep
                           |
                           v
                    time embedding
                           |
                           |
    x_t
     |
     v
  Encoder
     |
     |--------------------------+
     |                          |
     v                          |
  Downsample                    |
     |                          |
     v                          |
  Encoder                       |
     |                          |
     |----------------------+    |
                            |    |
                            v    |
                         Bottleneck
                            |
                            v
                         Decoder
                            |
                    +-------+
                    |
                    v
                 Upsample
                    |
                    v
                 Decoder
                    |
                    v
               epsilon_theta


The horizontal connections are:

    SKIP CONNECTIONS


================================================================================
WHY DOWNsample?
================================================================================

Suppose:

    input = 32 x 32

After downsampling:

    16 x 16

Then:

    8 x 8

At 8x8, each feature location can represent a much larger
receptive field in the original image.

Therefore the network can learn more global structure.


================================================================================
WHY UPSAMPLE?
================================================================================

The final prediction must have the same spatial resolution
as the original noisy image:

    x_t:
        [B,1,H,W]

    epsilon_prediction:
        [B,1,H,W]

Therefore the decoder gradually returns to the original
resolution.


================================================================================
WHY SKIP CONNECTIONS?
================================================================================

Downsampling loses spatial precision.

For example:

    exact edge location
    small object boundary
    fine texture

may be degraded by repeated downsampling.

The encoder features still contain this information.

Therefore we copy encoder features directly to the decoder.

Example:

    encoder feature
         |
         +--------------------+
                              |
                              v
                         concatenate
                              |
                              v
                          decoder


This is the central U-Net idea.


================================================================================
RESIDUAL BLOCKS
================================================================================

A residual block computes approximately:

    y = F(x) + x

Instead of forcing every layer to learn a complete transformation,
the network learns a residual correction.

This improves optimization in deeper networks.

Our residual block will contain:

    x
    |
    +------------------------+
    |                        |
    v                        |
    Conv                      |
    |                        |
    normalization             |
    |                        |
    activation                |
    |                        |
    timestep conditioning     |
    |                        |
    Conv                      |
    |                        |
    +------------------------+
             |
             v
            y


================================================================================
TIMESTEP CONDITIONING
================================================================================

The same noisy image format can correspond to very different
diffusion states.

For example:

    t = 900

means:

    "mostly noise"

while:

    t = 100

means:

    "mostly signal"

Therefore the network must receive t.

We encode t using sinusoidal features:

    sin(t f_i)
    cos(t f_i)

Then pass them through a small MLP.

The resulting embedding is injected into residual blocks.


================================================================================
MATHEMATICS — SINUSOIDAL EMBEDDING
================================================================================

For timestep t:

    e_i(t) = sin(t * omega_i)

and:

    e_{i+d/2}(t) = cos(t * omega_i)

where frequencies are distributed across multiple scales.

This is conceptually related to positional encoding in
transformers.


================================================================================
MATHEMATICS — RESIDUAL BLOCK
================================================================================

Let:

    h = Conv(x)

Then:

    h = Norm(h)

    h = Activation(h)

Add timestep information:

    h = h + Linear(t_embedding)

Then:

    h = Conv(h)

Finally:

    output = h + residual(x)


================================================================================
MATHEMATICS — DIFFUSION OBJECTIVE
================================================================================

The U-Net still predicts noise.

Given:

    x_t =
        sqrt(alpha_bar_t) x_0
        +
        sqrt(1-alpha_bar_t) epsilon

the model predicts:

    epsilon_theta(x_t,t)

Training minimizes:

    L =
        E[
            ||epsilon -
            epsilon_theta(x_t,t)||^2
        ]


The U-Net does NOT change the diffusion objective.

It changes the function used to approximate:

    epsilon_theta


================================================================================
ARCHITECTURE USED IN THIS LESSON
================================================================================

Input:

    32 x 32 x 1

Encoder level 1:

    1 -> 32 channels
    32 x 32

Downsample:

    32 -> 16 channels
    16 x 16

Encoder level 2:

    32 -> 64 channels
    16 x 16

Downsample:

    64 -> 8 x 8

Bottleneck:

    64 -> 128 channels
    8 x 8

Decoder:

    128 -> 64
    8 x 8 -> 16 x 16

Skip connection:

    encoder 64 + decoder 64

Then:

    64 -> 32
    16 x 16 -> 32 x 32

Skip connection:

    encoder 32 + decoder 32

Output:

    32 -> 1

Therefore:

    input:
        [B,1,32,32]

    output:
        [B,1,32,32]


================================================================================
IMPLEMENTATION
================================================================================

The implementation uses:

    Conv2d
    GroupNorm
    SiLU
    residual blocks
    timestep embeddings
    MaxPool-like strided convolution
    ConvTranspose2d upsampling
    skip connections

No pretrained model is used.

All weights are randomly initialized.


================================================================================
WHY GROUP NORMALIZATION?
================================================================================

Diffusion models often use small or variable batch sizes.

BatchNorm depends on batch statistics.

GroupNorm normalizes feature groups independently of the batch.

Therefore it is a useful normalization choice for diffusion U-Nets.

For educational simplicity, the number of groups is chosen
to divide the channel count.


================================================================================
TRAINING
================================================================================

For every batch:

    1. Generate clean x_0.
    2. Sample random timestep t.
    3. Sample Gaussian epsilon.
    4. Construct x_t.
    5. U-Net predicts epsilon.
    6. Calculate MSE.
    7. Backpropagate.
    8. Update weights.


================================================================================
DDPM SAMPLING
================================================================================

After training, the same reverse equations from Lesson 19
are used.

The only replacement is:

    Lesson 19:
        epsilon_theta = small CNN

    Lesson 20:
        epsilon_theta = Diffusion U-Net


================================================================================
VERIFICATION / TESTS
================================================================================

This lesson verifies:

    TEST 1
        Schedule validity.

    TEST 2
        Timestep embedding shape and finiteness.

    TEST 3
        Residual block shape.

    TEST 4
        U-Net output shape.

    TEST 5
        U-Net gradient flow.

    TEST 6
        Skip connections preserve expected spatial sizes.

    TEST 7
        Training loss decreases.

    TEST 8
        Predicted noise is finite.

    TEST 9
        DDPM reverse step is finite.

   TEST 10
        Complete sampling produces valid tensors.

   TEST 11
        Device consistency.


================================================================================
VISUALIZATION
================================================================================

The script visualizes:

    1. Training images.
    2. Forward diffusion.
    3. Training loss.
    4. Predicted noise.
    5. True noise.
    6. Noise prediction error.
    7. Reconstructed x_0.
    8. Reverse diffusion trajectory.
    9. Generated samples.


================================================================================
REAL-WORLD APPLICATION
================================================================================

Consider a robot camera observing an object.

The noisy input can be interpreted as:

    corrupted camera observation

The U-Net estimates:

    corruption pattern

and therefore provides:

    predicted clean visual structure

In a future conditional model, additional information could
control what is generated:

    text
    class
    robot task
    object category
    camera condition

That is where conditioning and cross-attention will eventually
enter the course.


================================================================================
REAL-WORLD RESULT
================================================================================

The important result is that the diffusion network is now
multi-scale.

It can process:

    fine spatial information

while simultaneously building:

    coarse/global information.

This is a major architectural step toward practical
high-quality diffusion models.


================================================================================
LESSON RESULT
================================================================================

Lesson 19 established:

    DDPM mathematics.

Lesson 20 establishes:

    DDPM + multi-scale U-Net.

The complete conceptual system is now:

    random noise
        |
        v
    Diffusion U-Net
        |
        v
    predicted noise
        |
        v
    DDPM reverse step
        |
        v
    less noisy image
        |
        v
    ...
        |
        v
    generated image


================================================================================
CONNECTION TO NEXT LESSON
================================================================================

Lesson 21 focuses specifically on diffusion sampling.

We will analyze:

    - reverse trajectories
    - sampling speed
    - timestep selection
    - denoising quality
    - deterministic vs stochastic behavior
    - practical sampling diagnostics

The U-Net will remain the learned denoiser.

The next question becomes:

    "How efficiently can we use the learned denoiser
     to generate an image?"


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

TRAIN_STEPS = 1800

LEARNING_RATE = 2e-3

BASE_CHANNELS = 32

TIME_EMBED_DIM = 128

GROUPS = 8

SAMPLE_COUNT = 8

EPS = 1e-8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TEST_DEVICE = torch.device(
    "cpu"
)

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
# DEVICE HELPERS
# ==============================================================================

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


def device_string(
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


# ==============================================================================
# SYNTHETIC IMAGE GENERATION
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

    # Smooth background.
    image = (
        0.08
        +
        0.12 * xx
        +
        0.10 * yy
    )

    # Smooth radial illumination.
    cx = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.5
        +
        0.25
    )

    cy = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.5
        +
        0.25
    )

    radial = torch.exp(
        -(
            (
                (xx - cx) ** 2
                +
                (yy - cy) ** 2
            )
            /
            0.12
        )
    )

    image = (
        image
        +
        0.12 * radial
    )

    # Rectangle.
    rx = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.55
        +
        0.05
    )

    ry = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.55
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
        0.10
    )

    rh = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.25
        +
        0.10
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
        0.40 * rectangle.to(dtype)
    )

    # Circle.
    circle_x = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.65
        +
        0.175
    )

    circle_y = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.65
        +
        0.175
    )

    radius = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 0.09
        +
        0.055
    )

    circle = (
        (
            xx - circle_x
        ) ** 2
        +
        (
            yy - circle_y
        ) ** 2
        <
        radius ** 2
    )

    image = (
        image
        +
        0.50 * circle.to(dtype)
    )

    # Diagonal line.
    slope = (
        torch.rand(
            (),
            device=device,
            dtype=dtype,
        )
        * 1.0
        -
        0.5
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
        0.22 * diagonal.to(dtype)
    )

    # Fine texture.
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
        torch.cos(
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
        )
        for _ in range(batch_size)
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


def prepare_schedule(
    beta: torch.Tensor,
) -> dict:

    alpha = 1.0 - beta

    alpha_bar = torch.cumprod(
        alpha,
        dim=0,
    ).clamp(
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
    ).clamp(
        min=0.0
    )

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_alpha": sqrt_alpha,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar":
            sqrt_one_minus_alpha_bar,
        "posterior_variance":
            posterior_variance,
    }


def extract(
    values: torch.Tensor,
    t: torch.Tensor,
    x_shape: torch.Size,
) -> torch.Tensor:

    selected = values[t]

    return selected.reshape(
        t.shape[0],
        *(
            [1]
            *
            (
                len(x_shape) - 1
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
# SINUSOIDAL TIMESTEP EMBEDDING
# ==============================================================================

def timestep_embedding(
    t: torch.Tensor,
    embedding_dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:

    if embedding_dim % 2 != 0:

        raise ValueError(
            "embedding_dim must be even."
        )

    half = embedding_dim // 2

    frequencies = torch.exp(
        -math.log(max_period)
        *
        torch.arange(
            half,
            device=t.device,
            dtype=torch.float32,
        )
        /
        max(
            half - 1,
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
# NORMALIZATION HELPER
# ==============================================================================

def make_group_norm(
    channels: int,
    preferred_groups: int = GROUPS,
) -> nn.GroupNorm:

    groups = min(
        preferred_groups,
        channels,
    )

    while (
        channels % groups
        != 0
        and groups > 1
    ):

        groups -= 1

    return nn.GroupNorm(
        groups,
        channels,
    )


# ==============================================================================
# RESIDUAL BLOCK
# ==============================================================================

class ResidualBlock(nn.Module):
    """
    Residual convolutional block with timestep conditioning.

    Input:

        x
        t_embedding

    Output:

        F(x,t) + residual(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
    ):

        super().__init__()

        self.norm1 = make_group_norm(
            in_channels
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

        self.norm2 = make_group_norm(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.activation = nn.SiLU()

        if (
            in_channels
            != out_channels
        ):

            self.residual_projection = (
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                )
            )

        else:

            self.residual_projection = (
                nn.Identity()
            )

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
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:

        residual = (
            self.residual_projection(x)
        )

        h = self.norm1(x)

        h = self.activation(h)

        h = self.conv1(h)

        time_bias = (
            self.time_projection(
                time_embedding
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        h = h + time_bias

        h = self.norm2(h)

        h = self.activation(h)

        h = self.conv2(h)

        return h + residual


# ==============================================================================
# DIFFUSION U-NET
# ==============================================================================

class DiffusionUNet(nn.Module):
    """
    Educational diffusion U-Net.

    Spatial resolution:

        32x32
           |
        16x16
           |
         8x8
           |
        bottleneck
           |
         8x8
           |
        16x16
           |
        32x32

    Skip connections preserve high-resolution information.
    """

    def __init__(
        self,
        base_channels: int = BASE_CHANNELS,
        time_embed_dim: int = TIME_EMBED_DIM,
    ):

        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # ----------------------------------------------------------------------
        # Time embedding
        # ----------------------------------------------------------------------

        self.time_mlp = nn.Sequential(
            nn.Linear(
                time_embed_dim,
                time_embed_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                time_embed_dim,
                time_embed_dim,
            ),
        )

        # ----------------------------------------------------------------------
        # Input projection
        # ----------------------------------------------------------------------

        self.input_conv = nn.Conv2d(
            1,
            c1,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------------------------
        # Encoder level 1
        # ----------------------------------------------------------------------

        self.enc1 = ResidualBlock(
            c1,
            c1,
            time_embed_dim,
        )

        # 32x32 -> 16x16
        self.down1 = nn.Conv2d(
            c1,
            c2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # ----------------------------------------------------------------------
        # Encoder level 2
        # ----------------------------------------------------------------------

        self.enc2 = ResidualBlock(
            c2,
            c2,
            time_embed_dim,
        )

        # 16x16 -> 8x8
        self.down2 = nn.Conv2d(
            c2,
            c3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # ----------------------------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------------------------

        self.mid1 = ResidualBlock(
            c3,
            c3,
            time_embed_dim,
        )

        self.mid2 = ResidualBlock(
            c3,
            c3,
            time_embed_dim,
        )

        # ----------------------------------------------------------------------
        # Decoder level 2
        # ----------------------------------------------------------------------

        # 8x8 -> 16x16
        self.up2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # c2 decoder + c2 encoder
        self.dec2 = ResidualBlock(
            c2 + c2,
            c2,
            time_embed_dim,
        )

        # ----------------------------------------------------------------------
        # Decoder level 1
        # ----------------------------------------------------------------------

        # 16x16 -> 32x32
        self.up1 = nn.ConvTranspose2d(
            c2,
            c1,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # c1 decoder + c1 encoder
        self.dec1 = ResidualBlock(
            c1 + c1,
            c1,
            time_embed_dim,
        )

        # ----------------------------------------------------------------------
        # Output
        # ----------------------------------------------------------------------

        self.output_norm = make_group_norm(
            c1
        )

        self.output_activation = nn.SiLU()

        self.output_conv = nn.Conv2d(
            c1,
            1,
            kernel_size=3,
            padding=1,
        )

        self.initialize_weights()

    def initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                (
                    nn.Conv2d,
                    nn.ConvTranspose2d,
                ),
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
        # Timestep embedding
        # ----------------------------------------------------------------------

        time_features = timestep_embedding(
            t,
            TIME_EMBED_DIM,
        )

        time_features = self.time_mlp(
            time_features
        )

        # ----------------------------------------------------------------------
        # Encoder
        # ----------------------------------------------------------------------

        h0 = self.input_conv(x)

        h1 = self.enc1(
            h0,
            time_features,
        )

        # Save high-resolution skip.
        skip1 = h1

        h2 = self.down1(h1)

        h2 = self.enc2(
            h2,
            time_features,
        )

        # Save medium-resolution skip.
        skip2 = h2

        h3 = self.down2(h2)

        # ----------------------------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------------------------

        h3 = self.mid1(
            h3,
            time_features,
        )

        h3 = self.mid2(
            h3,
            time_features,
        )

        # ----------------------------------------------------------------------
        # Decoder
        # ----------------------------------------------------------------------

        h = self.up2(h3)

        # Skip connection at 16x16.
        h = torch.cat(
            [
                h,
                skip2,
            ],
            dim=1,
        )

        h = self.dec2(
            h,
            time_features,
        )

        h = self.up1(h)

        # Skip connection at 32x32.
        h = torch.cat(
            [
                h,
                skip1,
            ],
            dim=1,
        )

        h = self.dec1(
            h,
            time_features,
        )

        # ----------------------------------------------------------------------
        # Noise prediction
        # ----------------------------------------------------------------------

        h = self.output_norm(h)

        h = self.output_activation(h)

        return self.output_conv(h)


# ==============================================================================
# NOISE-PREDICTION LOSS
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

def reverse_mean(
    xt: torch.Tensor,
    t: torch.Tensor,
    predicted_noise: torch.Tensor,
    schedule: dict,
):

    beta_t = extract(
        schedule["beta"],
        t,
        xt.shape,
    ).to(
        device=xt.device,
        dtype=xt.dtype,
    )

    sqrt_alpha_t = extract(
        torch.sqrt(
            schedule["alpha"]
        ),
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

    return (
        xt
        -
        (
            beta_t
            /
            (
                sqrt_one_minus_alpha_bar_t
                +
                EPS
            )
        )
        *
        predicted_noise
    ) / (
        sqrt_alpha_t
        +
        EPS
    )


# ==============================================================================
# DDPM REVERSE STEP
# ==============================================================================

@torch.no_grad()
def reverse_step(
    model: nn.Module,
    xt: torch.Tensor,
    timestep: int,
    schedule: dict,
):

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

    mean = reverse_mean(
        xt,
        t,
        predicted_noise,
        schedule,
    )

    variance = extract(
        schedule[
            "posterior_variance"
        ],
        t,
        xt.shape,
    ).to(
        device=xt.device,
        dtype=xt.dtype,
    )

    noise_mask = (
        (t > 0)
        .float()
        .reshape(
            t.shape[0],
            1,
            1,
            1,
        )
    )

    noise = torch.randn_like(
        xt
    )

    previous = (
        mean
        +
        noise_mask
        *
        torch.sqrt(
            variance.clamp(
                min=0.0
            )
        )
        *
        noise
    )

    return (
        previous,
        predicted_noise,
    )


# ==============================================================================
# COMPLETE SAMPLER
# ==============================================================================

@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    schedule: dict,
    batch_size: int,
    record_timesteps=None,
):

    model.eval()

    x = torch.randn(
        batch_size,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    if record_timesteps is None:

        record_timesteps = []

    trajectory = {}

    if T in record_timesteps:

        trajectory[T] = (
            x.detach().cpu()
        )

    for timestep in range(
        T,
        0,
        -1,
    ):

        x, _ = reverse_step(
            model,
            x,
            timestep,
            schedule,
        )

        if (
            timestep - 1
            in record_timesteps
        ):

            trajectory[
                timestep - 1
            ] = (
                x.detach().cpu()
            )

    model.train()

    return (
        x,
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

    assert torch.isfinite(
        alpha_bar
    ).all()

    assert torch.all(
        alpha_bar[1:]
        <=
        alpha_bar[:-1]
    )

    print(
        "[PASS] DDPM schedule is valid."
    )


# ==============================================================================
# TEST 2 — TIMESTEP EMBEDDING
# ==============================================================================

def test_timestep_embedding():

    print()
    print("=" * 80)
    print(
        "TEST 2 — TIMESTEP EMBEDDING"
    )
    print("=" * 80)

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

    embedding = timestep_embedding(
        t,
        TIME_EMBED_DIM,
    )

    print(
        f"Input shape     : "
        f"{tuple(t.shape)}"
    )

    print(
        f"Embedding shape : "
        f"{tuple(embedding.shape)}"
    )

    print(
        f"Embedding min   : "
        f"{embedding.min().item():.6f}"
    )

    print(
        f"Embedding max   : "
        f"{embedding.max().item():.6f}"
    )

    assert embedding.shape == (
        4,
        TIME_EMBED_DIM,
    )

    assert torch.isfinite(
        embedding
    ).all()

    assert (
        embedding.abs().max()
        <=
        1.0
        +
        1e-6
    )

    print(
        "[PASS] Timestep embedding is valid."
    )


# ==============================================================================
# TEST 3 — RESIDUAL BLOCK
# ==============================================================================

def test_residual_block():

    print()
    print("=" * 80)
    print(
        "TEST 3 — RESIDUAL BLOCK"
    )
    print("=" * 80)

    block = ResidualBlock(
        32,
        64,
        TIME_EMBED_DIM,
    ).to(
        DEVICE
    )

    x = torch.randn(
        4,
        32,
        16,
        16,
        device=DEVICE,
    )

    t = torch.randint(
        0,
        T,
        (
            4,
        ),
        device=DEVICE,
    )

    embedding = timestep_embedding(
        t,
        TIME_EMBED_DIM,
    )

    output = block(
        x,
        embedding,
    )

    print(
        f"Input shape  : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Output shape : "
        f"{tuple(output.shape)}"
    )

    assert output.shape == (
        4,
        64,
        16,
        16,
    )

    assert torch.isfinite(
        output
    ).all()

    print(
        "[PASS] Residual block works."
    )


# ==============================================================================
# TEST 4 — U-NET OUTPUT
# ==============================================================================

def test_unet_output(
    model: nn.Module,
):

    print()
    print("=" * 80)
    print(
        "TEST 4 — U-NET OUTPUT SHAPE"
    )
    print("=" * 80)

    x = torch.randn(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    t = torch.randint(
        0,
        T,
        (
            4,
        ),
        device=DEVICE,
    )

    with torch.no_grad():

        output = model(
            x,
            t,
        )

    print(
        f"Input shape  : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Output shape : "
        f"{tuple(output.shape)}"
    )

    assert output.shape == x.shape

    assert torch.isfinite(
        output
    ).all()

    print(
        "[PASS] U-Net output matches input spatial shape."
    )


# ==============================================================================
# TEST 5 — GRADIENT FLOW
# ==============================================================================

def test_gradient_flow(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 5 — U-NET GRADIENT FLOW"
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

    trainable = 0

    with_grad = 0

    maximum_gradient = 0.0

    for parameter in model.parameters():

        if not parameter.requires_grad:

            continue

        trainable += 1

        if parameter.grad is not None:

            with_grad += 1

            maximum_gradient = max(
                maximum_gradient,
                parameter.grad.abs().max().item(),
            )

    print(
        f"Trainable tensors : "
        f"{trainable}"
    )

    print(
        f"Gradient tensors  : "
        f"{with_grad}"
    )

    print(
        f"Maximum gradient  : "
        f"{maximum_gradient:.6e}"
    )

    assert trainable == with_grad

    assert maximum_gradient > 0

    print(
        "[PASS] Gradient flows through the complete U-Net."
    )


# ==============================================================================
# TEST 6 — MULTI-SCALE SPATIAL STRUCTURE
# ==============================================================================

def test_multiscale_structure(
    model: DiffusionUNet,
):

    print()
    print("=" * 80)
    print(
        "TEST 6 — MULTI-SCALE U-NET STRUCTURE"
    )
    print("=" * 80)

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
            700,
        ],
        device=DEVICE,
    )

    with torch.no_grad():

        time_features = timestep_embedding(
            t,
            TIME_EMBED_DIM,
        )

        time_features = model.time_mlp(
            time_features
        )

        h0 = model.input_conv(
            x
        )

        h1 = model.enc1(
            h0,
            time_features,
        )

        h2 = model.down1(
            h1
        )

        h2 = model.enc2(
            h2,
            time_features,
        )

        h3 = model.down2(
            h2
        )

    print(
        f"Input resolution      : "
        f"{tuple(x.shape[-2:])}"
    )

    print(
        f"Encoder level 1       : "
        f"{tuple(h1.shape[-2:])}"
    )

    print(
        f"Encoder level 2       : "
        f"{tuple(h2.shape[-2:])}"
    )

    print(
        f"Bottleneck resolution : "
        f"{tuple(h3.shape[-2:])}"
    )

    assert h1.shape[-2:] == (
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert h2.shape[-2:] == (
        IMAGE_SIZE // 2,
        IMAGE_SIZE // 2,
    )

    assert h3.shape[-2:] == (
        IMAGE_SIZE // 4,
        IMAGE_SIZE // 4,
    )

    print(
        "[PASS] U-Net contains multiple spatial scales."
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
        "TRAINING DIFFUSION U-NET"
    )
    print("=" * 80)

    print(
        f"Training steps : "
        f"{TRAIN_STEPS}"
    )

    print(
        f"Batch size     : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Learning rate  : "
        f"{LEARNING_RATE}"
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
                    history[-100:]
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
        f"Best loss             : "
        f"{best_loss:.6f}"
    )

    assert np.isfinite(
        initial_loss
    )

    assert np.isfinite(
        final_loss
    )

    assert (
        best_loss
        <
        initial_loss
    )

    assert (
        final_loss
        <
        initial_loss * 0.97
    )

    print(
        "[PASS] U-Net training reduces noise-prediction loss."
    )


# ==============================================================================
# TEST 8 — NOISE PREDICTION FINITENESS
# ==============================================================================

def test_noise_prediction(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 8 — PREDICTED NOISE"
    )
    print("=" * 80)

    x0 = create_image_batch(
        8,
        DEVICE,
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

    with torch.no_grad():

        predicted = model(
            xt,
            t,
        )

    mse = F.mse_loss(
        predicted,
        noise,
    )

    print(
        f"True noise std      : "
        f"{noise.std().item():.6f}"
    )

    print(
        f"Predicted noise std : "
        f"{predicted.std().item():.6f}"
    )

    print(
        f"Prediction MSE       : "
        f"{mse.item():.6f}"
    )

    assert predicted.shape == noise.shape

    assert torch.isfinite(
        predicted
    ).all()

    assert torch.isfinite(
        mse
    )

    print(
        "[PASS] Predicted noise is finite and correctly shaped."
    )


# ==============================================================================
# TEST 9 — REVERSE STEP
# ==============================================================================

def test_reverse_step(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 9 — DDPM REVERSE STEP"
    )
    print("=" * 80)

    x = torch.randn(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    with torch.no_grad():

        previous, prediction = (
            reverse_step(
                model,
                x,
                500,
                schedule,
            )
        )

    print(
        f"x_t shape        : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Prediction shape  : "
        f"{tuple(prediction.shape)}"
    )

    print(
        f"x_(t-1) shape    : "
        f"{tuple(previous.shape)}"
    )

    assert previous.shape == x.shape

    assert prediction.shape == x.shape

    assert torch.isfinite(
        previous
    ).all()

    assert torch.isfinite(
        prediction
    ).all()

    print(
        "[PASS] DDPM reverse step works with the U-Net."
    )


# ==============================================================================
# TEST 10 — COMPLETE SAMPLER
# ==============================================================================

def test_complete_sampler(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 10 — COMPLETE DDPM SAMPLER"
    )
    print("=" * 80)

    samples, trajectory = (
        sample_ddpm(
            model,
            schedule,
            batch_size=4,
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
        f"Trajectory keys : "
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
        "[PASS] Complete U-Net DDPM sampling works."
    )


# ==============================================================================
# TEST 11 — DEVICE CONSISTENCY
# ==============================================================================

def test_device_consistency(
    model: nn.Module,
    schedule: dict,
):

    print()
    print("=" * 80)
    print(
        "TEST 11 — DEVICE CONSISTENCY"
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

    with torch.no_grad():

        prediction = model(
            x,
            t,
        )

    print(
        f"Configured device : "
        f"{device_string(DEVICE)}"
    )

    print(
        f"Model device      : "
        f"{device_string(model_device)}"
    )

    print(
        f"Schedule device   : "
        f"{device_string(schedule_device)}"
    )

    print(
        f"Input device      : "
        f"{device_string(x.device)}"
    )

    print(
        f"Output device     : "
        f"{device_string(prediction.device)}"
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
# VISUALIZATION — FORWARD DIFFUSION
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
        "Forward Diffusion Before U-Net Denoising"
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
        history
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Noise prediction MSE"
    )

    plt.title(
        "Diffusion U-Net Training Loss"
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
):

    x0 = create_image_batch(
        1,
        DEVICE,
    )

    timesteps = [
        100,
        500,
        900,
    ]

    fig, axes = plt.subplots(
        len(timesteps),
        4,
        figsize=(10, 8),
    )

    for row, timestep in enumerate(
        timesteps
    ):

        t = torch.tensor(
            [
                timestep - 1,
            ],
            device=DEVICE,
        )

        xt, true_noise = q_sample(
            x0,
            t,
            schedule,
        )

        predicted_noise = model(
            xt,
            t,
        )

        x0_estimate = (
            xt
            -
            extract(
                schedule[
                    "sqrt_one_minus_alpha_bar"
                ],
                t,
                xt.shape,
            )
            *
            predicted_noise
        ) / (
            extract(
                schedule[
                    "sqrt_alpha_bar"
                ],
                t,
                xt.shape,
            )
            +
            EPS
        )

        error = (
            predicted_noise
            -
            true_noise
        )

        images = [
            (
                xt[0, 0]
                .cpu()
                .numpy()
            ),
            (
                true_noise[0, 0]
                .cpu()
                .numpy()
            ),
            (
                predicted_noise[0, 0]
                .cpu()
                .numpy()
            ),
            (
                x0_estimate[0, 0]
                .cpu()
                .numpy()
            ),
        ]

        titles = [
            f"x_t, t={timestep}",
            "True noise",
            "Predicted noise",
            "Estimated x_0",
        ]

        for column in range(4):

            axis = axes[
                row,
                column
            ]

            axis.imshow(
                images[column],
                cmap="gray",
            )

            axis.set_title(
                titles[column]
            )

            axis.axis("off")

    plt.suptitle(
        "Diffusion U-Net Noise Prediction"
    )

    plt.tight_layout()


# ==============================================================================
# VISUALIZATION — REVERSE TRAJECTORY
# ==============================================================================

def visualize_reverse_trajectory(
    trajectory: dict,
):

    ordered = [
        T,
        750,
        500,
        250,
        100,
        0,
    ]

    ordered = [
        timestep
        for timestep in ordered
        if timestep in trajectory
    ]

    fig, axes = plt.subplots(
        1,
        len(ordered),
        figsize=(
            3.0 * len(ordered),
            3.5,
        ),
    )

    if len(ordered) == 1:

        axes = [
            axes
        ]

    for axis, timestep in zip(
        axes,
        ordered,
    ):

        image = (
            trajectory[
                timestep
            ][
                0,
                0,
            ]
            .numpy()
        )

        axis.imshow(
            image,
            cmap="gray",
        )

        if timestep == 0:

            axis.set_title(
                "Generated x_0"
            )

        else:

            axis.set_title(
                f"x_{timestep}"
            )

        axis.axis("off")

    plt.suptitle(
        "Reverse Diffusion Using U-Net"
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
            rows * 2.8,
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

        axis.axis("off")

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
        "Diffusion U-Net Generated Samples"
    )

    plt.tight_layout()


# ==============================================================================
# REAL-WORLD EXPERIMENT
# ==============================================================================

def real_world_experiment():

    print()
    print("=" * 100)
    print(
        "REAL-WORLD EXPERIMENT — MULTI-SCALE ROBOT CAMERA INFORMATION"
    )
    print("=" * 100)

    print(
        """
A robot camera image contains information at multiple scales.

HIGH RESOLUTION:

    - edges
    - gripper boundaries
    - object corners
    - small visual details

LOW RESOLUTION:

    - object position
    - scene layout
    - large geometric structure

The diffusion U-Net explicitly processes both.

Encoder:

    camera image
        |
        v
    high-resolution features
        |
        v
    medium-resolution features
        |
        v
    low-resolution global representation

Decoder:

    global representation
        |
        v
    recover medium-scale information
        |
        v
    recover fine spatial information

Skip connections ensure that fine information does not
disappear during downsampling.

This architecture is therefore much better suited to
image generation than a simple single-resolution CNN.
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
        "LESSON 20 RESULT"
    )
    print("=" * 80)

    print(
        f"""
The diffusion model now uses a multi-scale U-Net.

Architecture:

    x_t
      |
      v
    Encoder
      |
      +--------------------------+
      |                          |
      v                          |
    Downsample                   |
      |                          |
      v                          |
    Bottleneck                   |
      |                          |
      v                          |
    Upsample                     |
      |                          |
      +------ Skip connection ---+
      |
      v
    epsilon_theta(x_t,t)


Important components:

    1. Residual blocks
    2. Group normalization
    3. SiLU activation
    4. Timestep embedding
    5. Downsampling
    6. Bottleneck
    7. Upsampling
    8. Skip connections


Training:

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


The key architectural lesson is:

    CNN:
        mostly local / single-scale processing

    U-Net:
        local + global + multi-scale processing


The diffusion mathematics from Lesson 19 remains unchanged.

Only the learned denoiser has become more powerful.


NEXT:

    Lesson 21 — Diffusion Sampling

We will study the reverse process itself in much greater
detail and investigate how sampling steps affect generation.
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
        "LESSON 20 — DIFFUSION U-NET"
    )
    print("=" * 80)

    print(
        f"PyTorch version : "
        f"{torch.__version__}"
    )

    print(
        f"Device          : "
        f"{device_string(DEVICE)}"
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
        device=DEVICE,
    )

    schedule = prepare_schedule(
        beta
    )

    # ==========================================================================
    # CREATE U-NET
    # ==========================================================================

    model = DiffusionUNet(
        base_channels=BASE_CHANNELS,
        time_embed_dim=TIME_EMBED_DIM,
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
    # TESTS
    # ==========================================================================

    test_schedule(
        schedule
    )

    test_timestep_embedding()

    test_residual_block()

    test_unet_output(
        model
    )

    test_gradient_flow(
        model,
        schedule,
    )

    test_multiscale_structure(
        model
    )

    test_device_consistency(
        model,
        schedule,
    )

    # ==========================================================================
    # FORWARD DIFFUSION VISUALIZATION
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
    # POST-TRAINING TESTS
    # ==========================================================================

    test_training_improvement(
        history
    )

    test_noise_prediction(
        model,
        schedule,
    )

    test_reverse_step(
        model,
        schedule,
    )

    # ==========================================================================
    # COMPLETE SAMPLING
    # ==========================================================================

    print()
    print("=" * 80)
    print(
        "GENERATING U-NET DDPM SAMPLES"
    )
    print("=" * 80)

    start = time.perf_counter()

    samples, trajectory = (
        sample_ddpm(
            model,
            schedule,
            batch_size=SAMPLE_COUNT,
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

    elapsed = (
        time.perf_counter()
        -
        start
    )

    print(
        f"Sampling time : "
        f"{elapsed:.2f} seconds"
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
    # VISUALIZATIONS
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

    visualize_noise_prediction(
        model,
        schedule,
    )

    visualize_reverse_trajectory(
        trajectory
    )

    visualize_generated_samples(
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
        "ALL LESSON 20 TESTS PASSED"
    )
    print("=" * 80)

    print()
    print(
        "Next: Lesson 21 — Diffusion Sampling"
    )

    plt.show()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()