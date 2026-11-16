
# =============================================================================
# COMPUTER VISION ADVANCED CRASH COURSE
# LESSON 25 — CROSS-ATTENTION FOR CONDITIONAL DIFFUSION
#
# FILENAME: lesson25_cross_attention.py
#
# PURPOSE
# Build a conditional latent diffusion model using multi-head cross-attention.
#
# WHAT YOU WILL LEARN
# 1. Queries, keys, and values (Q, K, V).
# 2. Self-attention versus cross-attention.
# 3. Multi-head attention.
# 4. Image features querying condition tokens.
# 5. Cross-attention inside a diffusion U-Net.
# 6. Training and sampling a conditional latent diffusion model.
#
# TECHNICAL ROADMAP
# Lessons 16-19: Diffusion mathematics and DDPM.
# Lesson 20: Diffusion U-Net.
# Lesson 21: DDPM sampling.
# Lesson 22: DDIM.
# Lesson 23: Latent diffusion.
# Lesson 24: Additive conditioning.
# Lesson 25: Cross-attention.
# Lesson 26: Text conditioning.
# Lesson 27: Build an own text-to-image model.
#
# REAL-WORLD PROBLEM
# A robot receives an image and an instruction:
# "pick up the bottle and place it in the recycling bin".
#
# Image features contain spatial information.
# Instruction tokens contain semantic information.
# Cross-attention allows image features to selectively retrieve information
# from the instruction tokens.
#
# THEORY
# Q = Wq * image_features
# K = Wk * condition_tokens
# V = Wv * condition_tokens
#
# Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
#
# Image features are queries.
# Condition tokens are keys and values.
#
# MATHEMATICS
# X: image features [B, N, C]
# C: condition tokens [B, M, C_context]
#
# Q = X Wq
# K = C Wk
# V = C Wv
#
# A = softmax(Q K^T / sqrt(d_head))
# Y = A V
#
# Each image position obtains a probability distribution over condition
# tokens. Different spatial positions can retrieve different information.
#
# IMPLEMENTATION
# - Four-class synthetic image dataset.
# - Convolutional autoencoder.
# - Latent diffusion.
# - Learned condition-token embeddings.
# - Multi-head cross-attention.
# - Conditional diffusion U-Net.
# - DDPM reverse sampler.
#
# NO PRETRAINED COMPONENTS
# All neural networks are initialized randomly.
# No pretrained models, checkpoints, or pretrained text encoders are used.
#
# VERIFICATION / TESTS
# 1. GroupNorm divisibility.
# 2. Dataset and condition alignment.
# 3. Autoencoder dimensions.
# 4. Diffusion schedule.
# 5. Time embedding.
# 6. Condition token embedding.
# 7. Cross-attention output shape.
# 8. Attention probability normalization.
# 9. Condition sensitivity.
# 10. Gradient flow.
# 11. Conditional U-Net output.
# 12. U-Net attention-map dimensions.
# 13. Forward diffusion.
# 14. Reverse diffusion.
# 15. Complete sampler.
# 16. Device consistency.
#
# REAL-WORLD APPLICATION
# Robot image features can query task-description tokens to retrieve
# relevant semantic information during manipulation.
#
# REAL-WORLD RESULT
# Demonstrates the core cross-attention mechanism used in language-conditioned
# generative vision architectures.
#
# LESSON RESULT
# A conditional latent diffusion model with multi-head cross-attention.
#
# CONNECTION TO NEXT LESSON
# Lesson 26 introduces natural-language tokenization and text conditioning.
# =============================================================================

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 32
IMAGE_CHANNELS = 1
NUM_CLASSES = 4

LATENT_CHANNELS = 4
LATENT_SIZE = 8

CONDITION_VOCAB_SIZE = 16
CONDITION_SEQ_LEN = 4
CONDITION_EMBED_DIM = 128

BASE_CHANNELS = 32
TIME_EMBED_DIM = 128
NUM_HEADS = 4

DIFFUSION_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.02

BATCH_SIZE = 64
AE_TRAIN_STEPS = 500
DIFFUSION_TRAIN_STEPS = 900

LEARNING_RATE = 2e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "circle",
    "square",
    "diagonal",
    "cross",
]

# U-Net attention resolution:
# Latent input: 8 x 8
# After one downsampling operation: 4 x 4
ATTENTION_SPATIAL_SIZE = LATENT_SIZE // 2
ATTENTION_NUM_POSITIONS = ATTENTION_SPATIAL_SIZE ** 2


# =============================================================================
# 2. GENERAL UTILITIES
# =============================================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def valid_group_count(num_channels, preferred_groups=8):
    num_channels = int(num_channels)
    preferred_groups = min(
        int(preferred_groups),
        num_channels,
    )

    if num_channels <= 0:
        raise ValueError("num_channels must be positive")

    for groups in range(preferred_groups, 0, -1):
        if num_channels % groups == 0:
            return groups

    return 1


def get_model_device(model):
    return next(model.parameters()).device


def move_schedule_to_device(schedule, device):
    return {
        key: value.to(device)
        for key, value in schedule.items()
    }


def assert_finite(tensor, name):
    assert torch.isfinite(tensor).all(), (
        f"{name} contains NaN or Inf"
    )


def print_tensor_info(name, tensor):
    print(
        f"{name:24s} "
        f"shape={str(tuple(tensor.shape)):20s} "
        f"device={str(tensor.device):8s} "
        f"dtype={tensor.dtype}"
    )


# =============================================================================
# 3. SYNTHETIC IMAGE DATASET
# =============================================================================

def make_synthetic_images(batch_size, device, labels=None):
    """
    Return:
        images: [B, 1, 32, 32]
        labels: [B]

    The masks must have shape [4, H, W], not [1, 4, H, W].
    """

    if labels is None:
        labels = torch.randint(
            0,
            NUM_CLASSES,
            (batch_size,),
            device=device,
        )
    else:
        labels = labels.to(
            device=device,
            dtype=torch.long,
        )
        batch_size = labels.shape[0]

    coords = torch.linspace(
        -1.0,
        1.0,
        IMAGE_SIZE,
        device=device,
    )

    yy, xx = torch.meshgrid(
        coords,
        coords,
        indexing="ij",
    )

    circle = (
        xx.square() + yy.square() < 0.48 ** 2
    ).float()

    square = (
        (xx.abs() < 0.43)
        & (yy.abs() < 0.43)
    ).float()

    diagonal = (
        (yy - xx).abs() < 0.12
    ).float()

    cross = (
        (xx.abs() < 0.12)
        | (yy.abs() < 0.12)
    ).float()

    masks = torch.stack(
        [circle, square, diagonal, cross],
        dim=0,
    )

    assert masks.shape == (
        NUM_CLASSES,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    selected = masks[labels]
    images = selected.unsqueeze(1)

    images = (
        images + torch.randn_like(images) * 0.025
    ).clamp(0.0, 1.0)

    assert images.shape == (
        batch_size,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    return images, labels


def make_condition_tokens(labels, device=None):
    if device is None:
        device = labels.device

    labels = labels.to(
        device=device,
        dtype=torch.long,
    )

    token_table = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    return token_table[labels]


# =============================================================================
# 4. AUTOENCODER
# =============================================================================

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                IMAGE_CHANNELS,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                LATENT_CHANNELS,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                LATENT_CHANNELS,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                IMAGE_CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


# =============================================================================
# 5. SINUSOIDAL TIME EMBEDDING
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2

        exponent = (
            -math.log(10000.0)
            / max(half_dim - 1, 1)
        )

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            ) * exponent
        )

        angles = (
            timesteps.float().unsqueeze(1)
            * frequencies.unsqueeze(0)
        )

        embedding = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1,
        )

        if self.embedding_dim % 2:
            embedding = F.pad(embedding, (0, 1))

        return embedding


# =============================================================================
# 6. CONDITION TOKEN EMBEDDING
# =============================================================================

class ConditionTokenEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            CONDITION_VOCAB_SIZE,
            CONDITION_EMBED_DIM,
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                CONDITION_SEQ_LEN,
                CONDITION_EMBED_DIM,
            ) * 0.02
        )

        self.norm = nn.LayerNorm(
            CONDITION_EMBED_DIM
        )

    def forward(self, token_ids):
        tokens = self.embedding(token_ids)
        tokens = tokens + self.position_embedding
        return self.norm(tokens)


# =============================================================================
# 7. MULTI-HEAD CROSS-ATTENTION
# =============================================================================

class CrossAttention(nn.Module):
    """
    image_features: [B, N, feature_dim]
    context_tokens: [B, M, context_dim]

    Q: [B, heads, N, head_dim]
    K: [B, heads, M, head_dim]
    V: [B, heads, M, head_dim]

    Attention weights:
        [B, heads, N, M]

    Output:
        [B, N, feature_dim]
    """

    def __init__(
        self,
        feature_dim,
        context_dim,
        num_heads=NUM_HEADS,
    ):
        super().__init__()

        if feature_dim % num_heads != 0:
            raise ValueError(
                "feature_dim must be divisible by num_heads"
            )

        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.to_q = nn.Linear(
            feature_dim,
            feature_dim,
            bias=False,
        )

        self.to_k = nn.Linear(
            context_dim,
            feature_dim,
            bias=False,
        )

        self.to_v = nn.Linear(
            context_dim,
            feature_dim,
            bias=False,
        )

        self.to_out = nn.Linear(
            feature_dim,
            feature_dim,
        )

    def split_heads(self, x):
        batch, sequence_length, _ = x.shape

        x = x.reshape(
            batch,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def forward(
        self,
        image_features,
        context_tokens,
        return_attention=False,
    ):
        batch = image_features.shape[0]

        q = self.to_q(image_features)
        k = self.to_k(context_tokens)
        v = self.to_v(context_tokens)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(self.head_dim)

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        attended = torch.matmul(
            weights,
            v,
        )

        attended = attended.transpose(
            1,
            2,
        ).contiguous()

        attended = attended.reshape(
            batch,
            image_features.shape[1],
            self.feature_dim,
        )

        output = self.to_out(attended)

        if return_attention:
            return output, weights

        return output


# =============================================================================
# 8. CROSS-ATTENTION BLOCK
# =============================================================================

class CrossAttentionBlock(nn.Module):

    def __init__(
        self,
        feature_dim,
        context_dim,
        num_heads=NUM_HEADS,
    ):
        super().__init__()

        self.norm_image = nn.LayerNorm(feature_dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.attention = CrossAttention(
            feature_dim,
            context_dim,
            num_heads,
        )

        self.norm_ff = nn.LayerNorm(feature_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(
                feature_dim,
                feature_dim * 4,
            ),
            nn.GELU(),
            nn.Linear(
                feature_dim * 4,
                feature_dim,
            ),
        )

    def forward(
        self,
        image_tokens,
        context_tokens,
        return_attention=False,
    ):
        normalized_image = self.norm_image(
            image_tokens
        )

        normalized_context = self.norm_context(
            context_tokens
        )

        if return_attention:
            attended, weights = self.attention(
                normalized_image,
                normalized_context,
                return_attention=True,
            )
        else:
            attended = self.attention(
                normalized_image,
                normalized_context,
            )

        x = image_tokens + attended

        x = x + self.feed_forward(
            self.norm_ff(x)
        )

        if return_attention:
            return x, weights

        return x


# =============================================================================
# 9. TIME-CONDITIONED RESIDUAL BLOCK
# =============================================================================

class TimeResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            valid_group_count(in_channels),
            in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            valid_group_count(out_channels),
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Linear(
            time_dim,
            out_channels,
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        )

    def forward(self, x, time_embedding):
        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_bias = self.time_projection(
            time_embedding
        )

        h = h + time_bias[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


# =============================================================================
# 10. CONDITIONAL LATENT DIFFUSION U-NET
# =============================================================================

class CrossAttentionUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(TIME_EMBED_DIM),

            nn.Linear(
                TIME_EMBED_DIM,
                TIME_EMBED_DIM,
            ),

            nn.SiLU(),

            nn.Linear(
                TIME_EMBED_DIM,
                TIME_EMBED_DIM,
            ),
        )

        self.condition_embedding = (
            ConditionTokenEmbedding()
        )

        self.input_conv = nn.Conv2d(
            LATENT_CHANNELS,
            BASE_CHANNELS,
            kernel_size=3,
            padding=1,
        )

        self.down_block1 = TimeResidualBlock(
            BASE_CHANNELS,
            BASE_CHANNELS,
            TIME_EMBED_DIM,
        )

        self.downsample = nn.Conv2d(
            BASE_CHANNELS,
            BASE_CHANNELS * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.down_block2 = TimeResidualBlock(
            BASE_CHANNELS * 2,
            BASE_CHANNELS * 2,
            TIME_EMBED_DIM,
        )

        self.mid_block = TimeResidualBlock(
            BASE_CHANNELS * 2,
            BASE_CHANNELS * 2,
            TIME_EMBED_DIM,
        )

        self.cross_attention = CrossAttentionBlock(
            feature_dim=BASE_CHANNELS * 2,
            context_dim=CONDITION_EMBED_DIM,
            num_heads=NUM_HEADS,
        )

        self.up_block = TimeResidualBlock(
            BASE_CHANNELS * 2,
            BASE_CHANNELS,
            TIME_EMBED_DIM,
        )

        self.output_norm = nn.GroupNorm(
            valid_group_count(BASE_CHANNELS),
            BASE_CHANNELS,
        )

        self.output_conv = nn.Conv2d(
            BASE_CHANNELS,
            LATENT_CHANNELS,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        noisy_latent,
        timesteps,
        condition_tokens,
        return_attention=False,
    ):
        time_embedding = self.time_embedding(
            timesteps
        )

        context = self.condition_embedding(
            condition_tokens
        )

        # [B, 4, 8, 8] -> [B, 32, 8, 8]
        h1 = self.input_conv(noisy_latent)

        h1 = self.down_block1(
            h1,
            time_embedding,
        )

        # [B, 32, 8, 8] -> [B, 64, 4, 4]
        h2 = self.downsample(h1)

        h2 = self.down_block2(
            h2,
            time_embedding,
        )

        h2 = self.mid_block(
            h2,
            time_embedding,
        )

        batch, channels, height, width = h2.shape

        # [B, 64, 4, 4] -> [B, 16, 64]
        image_tokens = h2.flatten(2).transpose(1, 2)

        if return_attention:
            image_tokens, attention_weights = (
                self.cross_attention(
                    image_tokens,
                    context,
                    return_attention=True,
                )
            )
        else:
            image_tokens = self.cross_attention(
                image_tokens,
                context,
            )

        # [B, 16, 64] -> [B, 64, 4, 4]
        h2 = image_tokens.transpose(
            1,
            2,
        ).reshape(
            batch,
            channels,
            height,
            width,
        )

        # Restore latent resolution: 4 x 4 -> 8 x 8.
        h2 = F.interpolate(
            h2,
            size=h1.shape[-2:],
            mode="nearest",
        )

        h = self.up_block(
            h2,
            time_embedding,
        )

        h = self.output_norm(h)
        h = F.silu(h)

        output = self.output_conv(h)

        if return_attention:
            return output, attention_weights

        return output


# =============================================================================
# 11. DIFFUSION SCHEDULE
# =============================================================================

def make_diffusion_schedule(
    timesteps=DIFFUSION_STEPS,
    device=DEVICE,
):
    betas = torch.linspace(
        BETA_START,
        BETA_END,
        timesteps,
        dtype=torch.float32,
        device=device,
    )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0,
    )

    alpha_bars_previous = torch.cat(
        [
            torch.ones(
                1,
                device=device,
                dtype=torch.float32,
            ),
            alpha_bars[:-1],
        ],
        dim=0,
    )

    posterior_variance = (
        betas
        * (1.0 - alpha_bars_previous)
        / (1.0 - alpha_bars)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_previous": alpha_bars_previous,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(
            1.0 - alpha_bars
        ),
        "sqrt_recip_alphas": torch.sqrt(
            1.0 / alphas
        ),
        "posterior_variance": posterior_variance,
    }


def extract(values, timesteps, target_shape):
    if values.device != timesteps.device:
        raise RuntimeError(
            "Schedule and timestep tensors must share a device"
        )

    result = values.gather(
        0,
        timesteps,
    )

    return result.reshape(
        timesteps.shape[0],
        *((1,) * (len(target_shape) - 1)),
    )


# =============================================================================
# 12. FORWARD DIFFUSION
# =============================================================================

def q_sample(
    clean_latent,
    timesteps,
    noise,
    schedule,
):
    sqrt_alpha_bar = extract(
        schedule["sqrt_alpha_bars"],
        timesteps,
        clean_latent.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule["sqrt_one_minus_alpha_bars"],
        timesteps,
        clean_latent.shape,
    )

    return (
        sqrt_alpha_bar * clean_latent
        + sqrt_one_minus_alpha_bar * noise
    )


# =============================================================================
# 13. DDPM REVERSE STEP
# =============================================================================

@torch.no_grad()
def cross_attention_ddpm_step(
    model,
    noisy_latent,
    timesteps,
    condition_tokens,
    schedule,
):
    model_device = get_model_device(model)

    noisy_latent = noisy_latent.to(model_device)

    timesteps = timesteps.to(
        model_device,
        dtype=torch.long,
    )

    condition_tokens = condition_tokens.to(
        model_device,
        dtype=torch.long,
    )

    schedule = move_schedule_to_device(
        schedule,
        model_device,
    )

    predicted_noise = model(
        noisy_latent,
        timesteps,
        condition_tokens,
    )

    beta_t = extract(
        schedule["betas"],
        timesteps,
        noisy_latent.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule["sqrt_one_minus_alpha_bars"],
        timesteps,
        noisy_latent.shape,
    )

    sqrt_recip_alpha = extract(
        schedule["sqrt_recip_alphas"],
        timesteps,
        noisy_latent.shape,
    )

    mean = sqrt_recip_alpha * (
        noisy_latent
        - beta_t
        * predicted_noise
        / sqrt_one_minus_alpha_bar
    )

    variance = extract(
        schedule["posterior_variance"],
        timesteps,
        noisy_latent.shape,
    )

    noise = torch.randn_like(noisy_latent)

    nonzero_mask = (
        timesteps != 0
    ).float().reshape(
        noisy_latent.shape[0],
        1,
        1,
        1,
    )

    return (
        mean
        + nonzero_mask
        * torch.sqrt(variance.clamp(min=0.0))
        * noise
    )


# =============================================================================
# 14. COMPLETE CONDITIONAL DDPM SAMPLER
# =============================================================================

@torch.no_grad()
def sample_cross_attention_diffusion(
    model,
    condition_tokens,
    schedule,
    num_samples=None,
    initial_noise=None,
    return_trajectory=False,
):
    model_device = get_model_device(model)

    condition_tokens = condition_tokens.to(
        model_device,
        dtype=torch.long,
    )

    schedule = move_schedule_to_device(
        schedule,
        model_device,
    )

    if num_samples is None:
        num_samples = condition_tokens.shape[0]

    if initial_noise is None:
        latent = torch.randn(
            num_samples,
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
            device=model_device,
        )
    else:
        latent = initial_noise.to(
            model_device
        ).clone()

    trajectory = {}

    model.eval()

    for timestep in reversed(
        range(DIFFUSION_STEPS)
    ):
        t = torch.full(
            (num_samples,),
            timestep,
            device=model_device,
            dtype=torch.long,
        )

        latent = cross_attention_ddpm_step(
            model,
            latent,
            t,
            condition_tokens,
            schedule,
        )

        if return_trajectory and timestep in {
            DIFFUSION_STEPS - 1,
            DIFFUSION_STEPS // 2,
            0,
        }:
            trajectory[timestep] = latent.detach().cpu()

    if return_trajectory:
        return latent, trajectory

    return latent


# =============================================================================
# 15. DATASET CREATION
# =============================================================================

def make_training_dataset(
    num_samples=4096,
    device=torch.device("cpu"),
):
    labels = torch.randint(
        0,
        NUM_CLASSES,
        (num_samples,),
        device=device,
    )

    images, labels = make_synthetic_images(
        num_samples,
        device,
        labels,
    )

    return images, labels


# =============================================================================
# 16. AUTOENCODER TRAINING
# =============================================================================

def train_autoencoder(
    autoencoder,
    images,
    steps=AE_TRAIN_STEPS,
):
    model_device = get_model_device(autoencoder)

    autoencoder.train()

    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=LEARNING_RATE,
    )

    losses = []

    for step in range(steps):
        # Images are stored on CPU.
        # Therefore, sample indices are deliberately created on CPU.
        indices = torch.randint(
            0,
            images.shape[0],
            (BATCH_SIZE,),
            device=images.device,
        )

        batch = images[indices].to(model_device)

        reconstruction, _ = autoencoder(batch)

        loss = F.mse_loss(
            reconstruction,
            batch,
        )

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            autoencoder.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(
                f"AE step {step + 1:4d}/{steps} "
                f"loss={loss.item():.6f}"
            )

    return losses


# =============================================================================
# 17. LATENT DIFFUSION TRAINING
# =============================================================================

def train_cross_attention_diffusion(
    model,
    autoencoder,
    images,
    labels,
    schedule,
    steps=DIFFUSION_TRAIN_STEPS,
):
    """
    DEVICE-SAFETY RULE:

    The autoencoder produces latent_dataset on the model device.

    The original images and labels may remain on CPU.

    We move labels to model_device BEFORE indexing them.

    Correct:
        labels_device = labels.to(model_device)
        batch_labels = labels_device[indices]

    Incorrect:
        batch_labels = labels[indices].to(model_device)

    The incorrect version tries to index a CPU tensor using CUDA indices.
    """

    model_device = get_model_device(model)

    autoencoder_device = get_model_device(
        autoencoder
    )

    if autoencoder_device != model_device:
        raise RuntimeError(
            "Autoencoder and diffusion model must be on the same device"
        )

    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            "Images and labels must have the same number of samples"
        )

    # Move labels ONCE before any GPU indexing.
    labels_device = labels.to(
        model_device,
        dtype=torch.long,
    )

    # This is the important fix.
    # The full label tensor now shares the latent dataset's device.
    assert labels_device.device == model_device

    autoencoder.eval()
    model.train()

    schedule = move_schedule_to_device(
        schedule,
        model_device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    losses = []

    # Encode all training images into latent representations.
    with torch.no_grad():
        latent_batches = []

        for start in range(
            0,
            images.shape[0],
            BATCH_SIZE,
        ):
            image_batch = images[
                start:start + BATCH_SIZE
            ].to(model_device)

            latent_batch = autoencoder.encode(
                image_batch
            )

            latent_batches.append(
                latent_batch
            )

        latent_dataset = torch.cat(
            latent_batches,
            dim=0,
        ).detach()

    assert latent_dataset.device == model_device
    assert labels_device.device == model_device

    assert latent_dataset.shape[0] == labels_device.shape[0]

    print_tensor_info(
        "Latent dataset",
        latent_dataset,
    )

    print_tensor_info(
        "Training labels",
        labels_device,
    )

    for step in range(steps):

        # indices are created on the same device as latent_dataset.
        indices = torch.randint(
            0,
            latent_dataset.shape[0],
            (BATCH_SIZE,),
            device=model_device,
        )

        # Both tensors are on model_device.
        clean_latent = latent_dataset[indices]
        batch_labels = labels_device[indices]

        assert clean_latent.device == model_device
        assert batch_labels.device == model_device

        condition_tokens = make_condition_tokens(
            batch_labels,
            model_device,
        )

        timesteps = torch.randint(
            0,
            DIFFUSION_STEPS,
            (BATCH_SIZE,),
            device=model_device,
            dtype=torch.long,
        )

        noise = torch.randn_like(clean_latent)

        noisy_latent = q_sample(
            clean_latent,
            timesteps,
            noise,
            schedule,
        )

        predicted_noise = model(
            noisy_latent,
            timesteps,
            condition_tokens,
        )

        loss = F.mse_loss(
            predicted_noise,
            noise,
        )

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(
                f"Diffusion step {step + 1:4d}/{steps} "
                f"loss={loss.item():.6f}"
            )

    return losses


# =============================================================================
# 18. TEST 1 — GROUP NORMALIZATION
# =============================================================================

def test_groupnorm():
    print("\nTEST 1 — GROUP NORMALIZATION")

    for channels in [1, 2, 4, 8, 16, 32, 64]:
        groups = valid_group_count(channels)

        assert channels % groups == 0

        layer = nn.GroupNorm(
            groups,
            channels,
        )

        x = torch.randn(
            2,
            channels,
            8,
            8,
        )

        assert layer(x).shape == x.shape

    print("PASS")


# =============================================================================
# 19. TEST 2 — DATASET AND CONDITION ALIGNMENT
# =============================================================================

def test_dataset_condition_alignment():
    print("\nTEST 2 — DATASET AND CONDITION ALIGNMENT")

    labels = torch.tensor(
        [0, 1, 2, 3, 2, 1],
        dtype=torch.long,
    )

    images, returned_labels = make_synthetic_images(
        len(labels),
        TEST_DEVICE,
        labels,
    )

    tokens = make_condition_tokens(
        returned_labels,
        TEST_DEVICE,
    )

    assert images.shape == (
        len(labels),
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert tokens.shape == (
        len(labels),
        CONDITION_SEQ_LEN,
    )

    assert torch.equal(
        returned_labels,
        labels,
    )

    assert images.min() >= 0
    assert images.max() <= 1

    print_tensor_info("Synthetic images", images)
    print_tensor_info("Condition tokens", tokens)

    print("PASS")


# =============================================================================
# 20. TEST 3 — AUTOENCODER SHAPES
# =============================================================================

def test_autoencoder_shapes():
    print("\nTEST 3 — AUTOENCODER SHAPES")

    model = Autoencoder().to(TEST_DEVICE)

    images, _ = make_synthetic_images(
        4,
        TEST_DEVICE,
    )

    reconstruction, latent = model(images)

    assert latent.shape == (
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    assert reconstruction.shape == images.shape

    assert_finite(latent, "latent")
    assert_finite(reconstruction, "reconstruction")

    print_tensor_info("Input images", images)
    print_tensor_info("Latent", latent)
    print_tensor_info("Reconstruction", reconstruction)

    print("PASS")


# =============================================================================
# 21. TEST 4 — DIFFUSION SCHEDULE
# =============================================================================

def test_diffusion_schedule():
    print("\nTEST 4 — DIFFUSION SCHEDULE")

    schedule = make_diffusion_schedule(
        DIFFUSION_STEPS,
        TEST_DEVICE,
    )

    assert schedule["betas"].shape == (
        DIFFUSION_STEPS,
    )

    assert torch.all(schedule["betas"] > 0)
    assert torch.all(schedule["betas"] < 1)

    assert torch.all(
        schedule["alpha_bars"][1:]
        <= schedule["alpha_bars"][:-1]
    )

    assert torch.all(
        schedule["alpha_bars"] > 0
    )

    print(
        "Final alpha_bar:",
        schedule["alpha_bars"][-1].item(),
    )

    print("PASS")


# =============================================================================
# 22. TEST 5 — TIME EMBEDDING
# =============================================================================

def test_time_embedding():
    print("\nTEST 5 — TIME EMBEDDING")

    embedding = SinusoidalTimeEmbedding(
        TIME_EMBED_DIM
    ).to(TEST_DEVICE)

    timesteps = torch.tensor(
        [0, 10, 50, 100],
        device=TEST_DEVICE,
    )

    result = embedding(timesteps)

    assert result.shape == (
        4,
        TIME_EMBED_DIM,
    )

    assert_finite(result, "time embedding")

    assert not torch.allclose(
        result[0],
        result[1],
    )

    print("PASS")


# =============================================================================
# 23. TEST 6 — CONDITION TOKEN EMBEDDING
# =============================================================================

def test_condition_token_embedding():
    print("\nTEST 6 — CONDITION TOKEN EMBEDDING")

    embedding = ConditionTokenEmbedding().to(
        TEST_DEVICE
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        device=TEST_DEVICE,
    )

    tokens = make_condition_tokens(
        labels,
        TEST_DEVICE,
    )

    result = embedding(tokens)

    assert result.shape == (
        4,
        CONDITION_SEQ_LEN,
        CONDITION_EMBED_DIM,
    )

    assert_finite(result, "condition embedding")

    print("PASS")


# =============================================================================
# 24. TEST 7 — CROSS-ATTENTION OUTPUT SHAPE
# =============================================================================

def test_cross_attention_shape():
    print("\nTEST 7 — CROSS-ATTENTION OUTPUT SHAPE")

    attention = CrossAttention(
        feature_dim=64,
        context_dim=CONDITION_EMBED_DIM,
        num_heads=NUM_HEADS,
    ).to(TEST_DEVICE)

    image_features = torch.randn(
        2,
        64,
        64,
        device=TEST_DEVICE,
    )

    context = torch.randn(
        2,
        CONDITION_SEQ_LEN,
        CONDITION_EMBED_DIM,
        device=TEST_DEVICE,
    )

    output = attention(
        image_features,
        context,
    )

    assert output.shape == image_features.shape

    print_tensor_info("Image features", image_features)
    print_tensor_info("Context", context)
    print_tensor_info("Attention output", output)

    print("PASS")


# =============================================================================
# 25. TEST 8 — ATTENTION WEIGHT NORMALIZATION
# =============================================================================

def test_attention_normalization():
    print("\nTEST 8 — ATTENTION WEIGHT NORMALIZATION")

    attention = CrossAttention(
        feature_dim=64,
        context_dim=CONDITION_EMBED_DIM,
        num_heads=NUM_HEADS,
    ).to(TEST_DEVICE)

    image_features = torch.randn(
        2,
        16,
        64,
        device=TEST_DEVICE,
    )

    context = torch.randn(
        2,
        CONDITION_SEQ_LEN,
        CONDITION_EMBED_DIM,
        device=TEST_DEVICE,
    )

    output, weights = attention(
        image_features,
        context,
        return_attention=True,
    )

    expected_shape = (
        2,
        NUM_HEADS,
        16,
        CONDITION_SEQ_LEN,
    )

    assert weights.shape == expected_shape

    sums = weights.sum(dim=-1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
    )

    assert torch.all(weights >= 0)

    print("Attention shape:", tuple(weights.shape))
    print("Mean attention sum:", sums.mean().item())

    print("PASS")


# =============================================================================
# 26. TEST 9 — CONDITION SENSITIVITY
# =============================================================================

def test_condition_changes_attention():
    print("\nTEST 9 — CONDITION CHANGES ATTENTION OUTPUT")

    attention = CrossAttention(
        feature_dim=64,
        context_dim=CONDITION_EMBED_DIM,
        num_heads=NUM_HEADS,
    ).to(TEST_DEVICE)

    image_features = torch.randn(
        2,
        16,
        64,
        device=TEST_DEVICE,
    )

    context_a = torch.randn(
        2,
        CONDITION_SEQ_LEN,
        CONDITION_EMBED_DIM,
        device=TEST_DEVICE,
    )

    context_b = torch.randn(
        2,
        CONDITION_SEQ_LEN,
        CONDITION_EMBED_DIM,
        device=TEST_DEVICE,
    )

    output_a = attention(
        image_features,
        context_a,
    )

    output_b = attention(
        image_features,
        context_b,
    )

    difference = (
        output_a - output_b
    ).abs().mean()

    assert difference.item() > 1e-6

    print(
        "Mean output difference:",
        difference.item(),
    )

    print("PASS")


# =============================================================================
# 27. TEST 10 — GRADIENT FLOW
# =============================================================================

def test_cross_attention_gradient_flow():
    print("\nTEST 10 — CROSS-ATTENTION GRADIENT FLOW")

    attention = CrossAttention(
        feature_dim=64,
        context_dim=CONDITION_EMBED_DIM,
        num_heads=NUM_HEADS,
    ).to(TEST_DEVICE)

    image_features = torch.randn(
        2,
        16,
        64,
        device=TEST_DEVICE,
        requires_grad=True,
    )

    context = torch.randn(
        2,
        CONDITION_SEQ_LEN,
        CONDITION_EMBED_DIM,
        device=TEST_DEVICE,
        requires_grad=True,
    )

    output = attention(
        image_features,
        context,
    )

    loss = output.square().mean()
    loss.backward()

    assert image_features.grad is not None
    assert context.grad is not None

    assert attention.to_q.weight.grad is not None
    assert attention.to_k.weight.grad is not None
    assert attention.to_v.weight.grad is not None

    assert image_features.grad.abs().sum() > 0
    assert context.grad.abs().sum() > 0

    print("PASS")


# =============================================================================
# 28. TEST 11 — CONDITIONAL U-NET OUTPUT
# =============================================================================

def test_conditional_unet_output():
    print("\nTEST 11 — CONDITIONAL U-NET OUTPUT")

    model = CrossAttentionUNet().to(TEST_DEVICE)

    latent = torch.randn(
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    timesteps = torch.tensor(
        [10, 50],
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    labels = torch.tensor(
        [0, 2],
        device=TEST_DEVICE,
    )

    tokens = make_condition_tokens(
        labels,
        TEST_DEVICE,
    )

    output = model(
        latent,
        timesteps,
        tokens,
    )

    assert output.shape == latent.shape
    assert_finite(output, "U-Net output")

    print_tensor_info("Latent input", latent)
    print_tensor_info("Predicted noise", output)

    print("PASS")


# =============================================================================
# 29. TEST 12 — CONDITIONAL U-NET ATTENTION OUTPUT
# =============================================================================

def test_unet_attention_output():
    print("\nTEST 12 — CONDITIONAL U-NET ATTENTION OUTPUT")

    model = CrossAttentionUNet().to(TEST_DEVICE)

    latent = torch.randn(
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    timesteps = torch.tensor(
        [10, 20],
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    labels = torch.tensor(
        [0, 1],
        device=TEST_DEVICE,
    )

    tokens = make_condition_tokens(
        labels,
        TEST_DEVICE,
    )

    output, weights = model(
        latent,
        timesteps,
        tokens,
        return_attention=True,
    )

    assert output.shape == latent.shape

    # The latent is 8 x 8.
    # The U-Net downsamples it once to 4 x 4.
    # Cross-attention therefore receives 16 image query positions.
    #
    # Attention layout:
    # [batch, heads, image_positions, condition_tokens]

    expected_shape = (
        2,
        NUM_HEADS,
        ATTENTION_NUM_POSITIONS,
        CONDITION_SEQ_LEN,
    )

    assert weights.shape == expected_shape, (
        f"Expected {expected_shape}, "
        f"received {tuple(weights.shape)}"
    )

    assert ATTENTION_NUM_POSITIONS == 16

    sums = weights.sum(dim=-1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
    )

    print("Latent spatial size:", LATENT_SIZE, "x", LATENT_SIZE)

    print(
        "Attention spatial size:",
        ATTENTION_SPATIAL_SIZE,
        "x",
        ATTENTION_SPATIAL_SIZE,
    )

    print(
        "Actual attention shape:",
        tuple(weights.shape),
    )

    print("PASS")


# =============================================================================
# 30. TEST 13 — FORWARD DIFFUSION
# =============================================================================

def test_forward_diffusion():
    print("\nTEST 13 — FORWARD DIFFUSION")

    schedule = make_diffusion_schedule(
        DIFFUSION_STEPS,
        TEST_DEVICE,
    )

    clean = torch.randn(
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    noise = torch.randn_like(clean)

    timesteps = torch.tensor(
        [
            0,
            10,
            50,
            DIFFUSION_STEPS - 1,
        ],
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    noisy = q_sample(
        clean,
        timesteps,
        noise,
        schedule,
    )

    assert noisy.shape == clean.shape
    assert_finite(noisy, "noisy latent")

    print("PASS")


# =============================================================================
# 31. TEST 14 — ONE-STEP REVERSE DIFFUSION
# =============================================================================

def test_one_step_reverse_diffusion():
    print("\nTEST 14 — ONE-STEP REVERSE DIFFUSION")

    model = CrossAttentionUNet().to(TEST_DEVICE)

    schedule = make_diffusion_schedule(
        DIFFUSION_STEPS,
        TEST_DEVICE,
    )

    latent = torch.randn(
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=TEST_DEVICE,
    )

    timesteps = torch.tensor(
        [50, 100],
        device=TEST_DEVICE,
        dtype=torch.long,
    )

    labels = torch.tensor(
        [0, 3],
        device=TEST_DEVICE,
    )

    tokens = make_condition_tokens(
        labels,
        TEST_DEVICE,
    )

    result = cross_attention_ddpm_step(
        model,
        latent,
        timesteps,
        tokens,
        schedule,
    )

    assert result.shape == latent.shape
    assert_finite(result, "reverse result")

    print("PASS")


# =============================================================================
# 32. TEST 15 — COMPLETE SAMPLER
# =============================================================================

def test_complete_sampler():
    print("\nTEST 15 — COMPLETE CONDITIONAL SAMPLER")

    model = CrossAttentionUNet().to(TEST_DEVICE)

    schedule = make_diffusion_schedule(
        DIFFUSION_STEPS,
        TEST_DEVICE,
    )

    labels = torch.tensor(
        [0, 1],
        device=TEST_DEVICE,
    )

    tokens = make_condition_tokens(
        labels,
        TEST_DEVICE,
    )

    samples, trajectory = sample_cross_attention_diffusion(
        model=model,
        condition_tokens=tokens,
        schedule=schedule,
        num_samples=2,
        return_trajectory=True,
    )

    assert samples.shape == (
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    assert set(trajectory.keys()) == {
        DIFFUSION_STEPS - 1,
        DIFFUSION_STEPS // 2,
        0,
    }

    assert_finite(samples, "generated latent")

    print(
        "Trajectory timesteps:",
        sorted(trajectory.keys()),
    )

    print("PASS")


# =============================================================================
# 33. TEST 16 — DEVICE CONSISTENCY
# =============================================================================

def test_device_consistency():
    print("\nTEST 16 — DEVICE CONSISTENCY")

    model = CrossAttentionUNet().to(TEST_DEVICE)

    schedule = make_diffusion_schedule(
        DIFFUSION_STEPS,
        TEST_DEVICE,
    )

    model_device = get_model_device(model)

    schedule = move_schedule_to_device(
        schedule,
        model_device,
    )

    labels = torch.tensor(
        [0, 1],
        device=model_device,
    )

    tokens = make_condition_tokens(
        labels,
        model_device,
    )

    latent = torch.randn(
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=model_device,
    )

    timesteps = torch.tensor(
        [10, 20],
        device=model_device,
        dtype=torch.long,
    )

    output = model(
        latent,
        timesteps,
        tokens,
    )

    assert output.device == model_device

    for value in schedule.values():
        assert value.device == model_device

    assert tokens.device == model_device

    print("Model device:", model_device)
    print("Schedule device:", schedule["betas"].device)

    print("PASS")


# =============================================================================
# 34. TEST 17 — CPU LABELS WITH GPU LATENT INDEXING
# =============================================================================


def test_cpu_labels_gpu_indexing():
    """
    Verify CPU labels can safely index batches on the actual model device.

    Important:
    torch.device("cuda") and torch.device("cuda:0") can compare unequal.
    Always obtain the actual device from an allocated tensor or model.
    """

    print("\nTEST 17 — CPU LABELS / MODEL-DEVICE INDEXING")

    # Original labels deliberately remain on CPU.
    labels_cpu = torch.tensor(
        [0, 1, 2, 3, 1, 2, 0, 3],
        dtype=torch.long,
        device="cpu",
    )

    # Allocate the latent dataset using the configured device.
    latent_dataset = torch.randn(
        8,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=DEVICE,
    )

    # Use the actual device reported by the allocated tensor.
    actual_device = latent_dataset.device

    # Move labels BEFORE indexing.
    labels_device = labels_cpu.to(
        device=actual_device,
        dtype=torch.long,
    )

    indices = torch.tensor(
        [7, 2, 5, 0],
        device=actual_device,
        dtype=torch.long,
    )

    batch_latents = latent_dataset[indices]
    batch_labels = labels_device[indices]

    # Verify actual device placement.
    assert batch_latents.device == actual_device
    assert batch_labels.device == actual_device
    assert indices.device == actual_device
    assert labels_device.device == actual_device

    # Verify batch dimensions.
    assert batch_latents.shape == (
        4,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    assert batch_labels.shape == (4,)

    # Verify indexing returned the correct labels.
    expected_labels = labels_cpu[indices.cpu()]

    assert torch.equal(
        batch_labels.cpu(),
        expected_labels,
    )

    print("Configured device :", DEVICE)
    print("Actual tensor device:", actual_device)
    print("Original labels    :", labels_cpu.device)
    print("Moved labels       :", labels_device.device)
    print("Indices            :", indices.device)
    print("Selected latents   :", batch_latents.device)
    print("Selected labels    :", batch_labels.device)

    print("PASS")



# =============================================================================
# 35. VISUALIZATION — IMAGE GRID
# =============================================================================

def show_image_grid(
    images,
    title,
    max_images=8,
):
    images = images.detach().cpu().clamp(
        0.0,
        1.0,
    )

    count = min(
        images.shape[0],
        max_images,
    )

    cols = min(4, count)
    rows = math.ceil(count / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.5, rows * 2.5),
    )

    axes = np.array(axes).reshape(-1)

    for index, axis in enumerate(axes):
        axis.axis("off")

        if index < count:
            axis.imshow(
                images[index, 0],
                cmap="gray",
                vmin=0,
                vmax=1,
            )

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 36. VISUALIZATION — TRAINING CURVES
# =============================================================================

def show_training_curves(
    ae_losses,
    diffusion_losses,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
    )

    axes[0].plot(ae_losses)
    axes[0].set_title("Autoencoder Training")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(True)

    axes[1].plot(diffusion_losses)
    axes[1].set_title("Cross-Attention Diffusion Training")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Noise Prediction MSE")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 37. VISUALIZATION — ATTENTION TOKEN IMPORTANCE
# =============================================================================

@torch.no_grad()
def visualize_attention(
    model,
    autoencoder,
    image,
    label,
):
    model_device = get_model_device(model)

    image = image[:1].to(model_device)

    label_tensor = torch.tensor(
        [label],
        device=model_device,
        dtype=torch.long,
    )

    tokens = make_condition_tokens(
        label_tensor,
        model_device,
    )

    latent = autoencoder.encode(image)

    timestep = torch.tensor(
        [DIFFUSION_STEPS // 2],
        device=model_device,
        dtype=torch.long,
    )

    _, attention_weights = model(
        latent,
        timestep,
        tokens,
        return_attention=True,
    )

    # [B, heads, positions, condition_tokens]
    weights = attention_weights[0]

    # Average attention over heads.
    average_attention = weights.mean(dim=0)

    # Average attention over image positions.
    token_importance = average_attention.mean(dim=0)

    plt.figure(figsize=(7, 4))

    plt.bar(
        range(CONDITION_SEQ_LEN),
        token_importance.detach().cpu().numpy(),
    )

    plt.xticks(
        range(CONDITION_SEQ_LEN),
        [f"Token {i}" for i in range(CONDITION_SEQ_LEN)],
    )

    plt.ylabel("Mean Attention Weight")

    plt.title(
        f"Cross-Attention Token Importance — "
        f"{CLASS_NAMES[label]}"
    )

    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 38. EXPERIMENT — SAME NOISE, DIFFERENT CONDITIONS
# =============================================================================

@torch.no_grad()
def experiment_condition_changes_output(
    model,
    autoencoder,
    schedule,
):
    model_device = get_model_device(model)

    initial_noise = torch.randn(
        1,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=model_device,
    )

    generated_images = []

    for label in range(NUM_CLASSES):
        labels = torch.tensor(
            [label],
            device=model_device,
        )

        tokens = make_condition_tokens(
            labels,
            model_device,
        )

        generated_latent = sample_cross_attention_diffusion(
            model=model,
            condition_tokens=tokens,
            schedule=schedule,
            num_samples=1,
            initial_noise=initial_noise,
        )

        generated_image = autoencoder.decode(
            generated_latent
        )

        generated_images.append(
            generated_image.cpu()
        )

    generated_images = torch.cat(
        generated_images,
        dim=0,
    )

    fig, axes = plt.subplots(
        1,
        NUM_CLASSES,
        figsize=(12, 3),
    )

    for index, axis in enumerate(axes):
        axis.imshow(
            generated_images[index, 0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axis.set_title(CLASS_NAMES[index])
        axis.axis("off")

    fig.suptitle(
        "Same Initial Noise — Different Conditions"
    )

    plt.tight_layout()
    plt.show()

    return generated_images


# =============================================================================
# 39. MAIN
# =============================================================================

def main():

    seed_everything()

    print("=" * 80)
    print("LESSON 25 — CROSS-ATTENTION FOR CONDITIONAL DIFFUSION")
    print("=" * 80)

    print("PyTorch version :", torch.__version__)
    print("Training device :", DEVICE)
    print("Test device     :", TEST_DEVICE)
    print("Diffusion steps :", DIFFUSION_STEPS)
    print("Attention heads :", NUM_HEADS)

    print(
        "Attention resolution:",
        ATTENTION_SPATIAL_SIZE,
        "x",
        ATTENTION_SPATIAL_SIZE,
    )

    # -------------------------------------------------------------------------
    # UNIT TESTS
    # -------------------------------------------------------------------------

    test_groupnorm()
    test_dataset_condition_alignment()
    test_autoencoder_shapes()
    test_diffusion_schedule()
    test_time_embedding()
    test_condition_token_embedding()
    test_cross_attention_shape()
    test_attention_normalization()
    test_condition_changes_attention()
    test_cross_attention_gradient_flow()
    test_conditional_unet_output()
    test_unet_attention_output()
    test_forward_diffusion()
    test_one_step_reverse_diffusion()
    test_complete_sampler()
    test_device_consistency()
    test_cpu_labels_gpu_indexing()

    print("\nALL UNIT TESTS PASSED")

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("GENERATING TRAINING DATA")
    print("=" * 80)

    # Keep source images and labels on CPU.
    images, labels = make_training_dataset(
        num_samples=4096,
        device=torch.device("cpu"),
    )

    print_tensor_info("Training images", images)
    print_tensor_info("Training labels", labels)

    show_image_grid(
        images[:8],
        "Synthetic Training Images",
    )

    # -------------------------------------------------------------------------
    # AUTOENCODER
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING AUTOENCODER")
    print("=" * 80)

    autoencoder = Autoencoder().to(DEVICE)

    start_time = time.time()

    ae_losses = train_autoencoder(
        autoencoder,
        images,
    )

    autoencoder.eval()

    print(
        "Autoencoder training time:",
        f"{time.time() - start_time:.2f} seconds",
    )

    with torch.no_grad():
        reconstruction, latent = autoencoder(
            images[:8].to(DEVICE)
        )

    print_tensor_info(
        "Autoencoder latent",
        latent,
    )

    show_image_grid(
        reconstruction,
        "Autoencoder Reconstructions",
    )

    # -------------------------------------------------------------------------
    # DIFFUSION MODEL
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING CROSS-ATTENTION DIFFUSION MODEL")
    print("=" * 80)

    diffusion_model = CrossAttentionUNet().to(DEVICE)

    schedule = make_diffusion_schedule(
        DIFFUSION_STEPS,
        DEVICE,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in diffusion_model.parameters()
    )

    print(
        "Diffusion model parameters:",
        f"{parameter_count:,}",
    )

    start_time = time.time()

    diffusion_losses = train_cross_attention_diffusion(
        model=diffusion_model,
        autoencoder=autoencoder,
        images=images,
        labels=labels,
        schedule=schedule,
    )

    print(
        "Diffusion training time:",
        f"{time.time() - start_time:.2f} seconds",
    )

    # -------------------------------------------------------------------------
    # FINAL DEVICE CHECK
    # -------------------------------------------------------------------------

    model_device = get_model_device(
        diffusion_model
    )

    assert get_model_device(autoencoder) == model_device

    schedule = move_schedule_to_device(
        schedule,
        model_device,
    )

    for value in schedule.values():
        assert value.device == model_device

    print("\nFinal model device:", model_device)
    print("Final schedule device:", schedule["betas"].device)

    # -------------------------------------------------------------------------
    # CONDITIONAL SAMPLE GENERATION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("GENERATING CONDITIONAL SAMPLES")
    print("=" * 80)

    labels_for_sampling = torch.arange(
        NUM_CLASSES,
        device=model_device,
    )

    tokens = make_condition_tokens(
        labels_for_sampling,
        model_device,
    )

    generated_latents = sample_cross_attention_diffusion(
        model=diffusion_model,
        condition_tokens=tokens,
        schedule=schedule,
        num_samples=NUM_CLASSES,
    )

    with torch.no_grad():
        generated_images = autoencoder.decode(
            generated_latents
        )

    show_image_grid(
        generated_images,
        "Generated Conditional Samples",
        max_images=NUM_CLASSES,
    )

    # -------------------------------------------------------------------------
    # ATTENTION VISUALIZATION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("VISUALIZING CROSS-ATTENTION")
    print("=" * 80)

    visualize_attention(
        diffusion_model,
        autoencoder,
        images[:1],
        label=0,
    )

    # -------------------------------------------------------------------------
    # SAME NOISE / DIFFERENT CONDITION EXPERIMENT
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAME NOISE — DIFFERENT CONDITIONS")
    print("=" * 80)

    condition_experiment_images = (
        experiment_condition_changes_output(
            diffusion_model,
            autoencoder,
            schedule,
        )
    )

    condition_difference = (
        condition_experiment_images[0]
        - condition_experiment_images[1]
    ).abs().mean()

    print(
        "Mean difference between first two conditions:",
        f"{condition_difference.item():.6f}",
    )

    # -------------------------------------------------------------------------
    # TRAINING CURVES
    # -------------------------------------------------------------------------

    show_training_curves(
        ae_losses,
        diffusion_losses,
    )

    # -------------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LESSON 25 COMPLETE")
    print("=" * 80)

    print("Implemented:")
    print("  [OK] Synthetic condition-token sequences")
    print("  [OK] Convolutional autoencoder")
    print("  [OK] Latent diffusion")
    print("  [OK] Multi-head cross-attention")
    print("  [OK] Conditional diffusion U-Net")
    print("  [OK] DDPM reverse sampling")
    print("  [OK] Attention visualization")
    print("  [OK] Same-noise / different-condition experiment")
    print("  [OK] CPU/GPU indexing verification")
    print("  [OK] Device consistency checks")

    print("\nREAL-WORLD CONNECTION")

    print(
        "Image features can query task-description tokens, "
        "allowing spatially selective conditioning."
    )

    print("\nNEXT LESSON")
    print("LESSON 26 — TEXT CONDITIONING")

    print("=" * 80)


if __name__ == "__main__":
    main()
