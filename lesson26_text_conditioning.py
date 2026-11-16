
"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 26 — TEXT CONDITIONING (CORRECTED GENERATION)
================================================================================

PURPOSE
-------
Build a text-conditioned latent diffusion model from scratch and generate
recognizable shapes that correspond to text prompts.

WHAT YOU WILL LEARN
-------------------
1. Tokenization and vocabulary construction.
2. Token IDs, padding, and attention masks.
3. Learned token and positional embeddings.
4. Transformer text encoding.
5. Multi-head cross-attention.
6. Latent normalization for diffusion.
7. A noise schedule suitable for the selected number of steps.
8. Text-conditioned latent diffusion training and DDPM sampling.
9. Quantitative evaluation of generated shapes.

TECHNICAL ROADMAP
-----------------
Lesson 23: Latent diffusion.
Lesson 24: Class conditioning.
Lesson 25: Cross-attention.
Lesson 26: Text conditioning.
Lesson 27: Build our own text-to-image model.

REAL-WORLD PROBLEM
------------------
A robot receives a visual instruction such as "make a circle".
A language-conditioned visual model must associate the instruction
with the corresponding visual concept.

THEORY
------
Latent normalization:

    z_normalized = (z - mean) / std

    z = z_normalized * std + mean

The diffusion model is trained on normalized latents. This is important
because the reverse sampler begins with standard Gaussian noise.

Forward diffusion:

    z_t = sqrt(alpha_bar_t) * z_0
          + sqrt(1-alpha_bar_t) * epsilon

Noise prediction objective:

    L = E[||epsilon - epsilon_theta(z_t,t,H)||^2]

Cross-attention:

    Q = image features
    K = text features
    V = text features

    Attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V

IMPLEMENTATION
--------------
- Synthetic shape dataset.
- Convolutional autoencoder.
- Per-channel latent normalization.
- Custom tokenizer.
- Random-initialized Transformer text encoder.
- Text cross-attention.
- Conditional latent diffusion U-Net.
- DDPM sampler.
- Shape-template evaluation.
- No pretrained models or checkpoints.

IMPORTANT CORRECTIONS
--------------------
1. Latents are normalized before diffusion training.
2. Generated latents are denormalized before decoding.
3. Diffusion uses 200 steps and a stronger beta schedule.
4. Sampling uses the same schedule used during training.
5. Dataset alignment is checked before indexing.
6. Generated image intensity and shape similarity are evaluated.

REAL-WORLD APPLICATION
----------------------
The architecture is a simplified language-conditioning mechanism
for text-to-image generation and vision-language robotics.

LIMITATION
----------
This is a four-prompt demonstration, not a general-purpose language model.
All model weights are initialized from scratch.

CONNECTION TO NEXT LESSON
-------------------------
Lesson 27: Build our own text-to-image model.
================================================================================
"""

import copy
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

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

VOCABULARY = [
    PAD_TOKEN,
    UNK_TOKEN,
    "make",
    "a",
    "circle",
    "square",
    "diagonal",
    "cross",
]

VOCAB = {word: i for i, word in enumerate(VOCABULARY)}
VOCAB_SIZE = len(VOCABULARY)

MAX_TEXT_LENGTH = 4

TEXT_EMBED_DIM = 128
TEXT_NUM_HEADS = 4
TEXT_NUM_LAYERS = 2
TEXT_FF_DIM = 256

BASE_CHANNELS = 32
TIME_EMBED_DIM = 128
CROSS_ATTENTION_HEADS = 4

# A longer schedule makes the terminal distribution much closer to N(0,I).
DIFFUSION_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.03

BATCH_SIZE = 128
AE_TRAIN_STEPS = 700
DIFFUSION_TRAIN_STEPS = 3000

AE_LEARNING_RATE = 2e-3
DIFFUSION_LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROMPTS = [
    "make a circle",
    "make a square",
    "make a diagonal",
    "make a cross",
]

CLASS_NAMES = [
    "circle",
    "square",
    "diagonal",
    "cross",
]


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 3. TOKENIZER
# =============================================================================

class SimpleTokenizer:

    def __init__(self, vocabulary, max_length):
        self.vocabulary = list(vocabulary)
        self.word_to_id = {
            word: i for i, word in enumerate(self.vocabulary)
        }

        self.max_length = max_length
        self.pad_id = self.word_to_id[PAD_TOKEN]
        self.unk_id = self.word_to_id[UNK_TOKEN]

    def tokenize(self, text):
        return text.lower().strip().split()

    def encode(self, text):

        words = self.tokenize(text)

        ids = [
            self.word_to_id.get(word, self.unk_id)
            for word in words
        ]

        ids = ids[:self.max_length]
        valid_length = len(ids)

        ids += [
            self.pad_id
        ] * (self.max_length - valid_length)

        mask = (
            [True] * valid_length
            + [False] * (self.max_length - valid_length)
        )

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )

    def batch_encode(self, texts):

        encoded = [self.encode(text) for text in texts]

        token_ids = torch.stack([
            item[0] for item in encoded
        ])

        masks = torch.stack([
            item[1] for item in encoded
        ])

        return token_ids, masks


TOKENIZER = SimpleTokenizer(
    VOCABULARY,
    MAX_TEXT_LENGTH,
)


# =============================================================================
# 4. SYNTHETIC DATASET
# =============================================================================

def create_synthetic_dataset(samples_per_class=256):

    images = []
    labels = []
    prompts = []

    axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE)

    yy, xx = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    for class_id in range(NUM_CLASSES):

        for _ in range(samples_per_class):

            image = torch.zeros(
                IMAGE_SIZE,
                IMAGE_SIZE,
            )

            noise = 0.015 * torch.randn_like(image)

            # Small size variation prevents the model from memorizing
            # only one exact pixel pattern for each class.
            if class_id == 0:
                radius = random.uniform(0.38, 0.52)
                mask = (
                    xx.square() + yy.square()
                    <= radius ** 2
                )

            elif class_id == 1:
                half_width = random.uniform(0.37, 0.52)
                mask = (
                    (xx.abs() <= half_width)
                    & (yy.abs() <= half_width)
                )

            elif class_id == 2:
                thickness = random.uniform(0.045, 0.085)
                mask = (
                    (xx - yy).abs() < thickness
                )

            else:
                thickness = random.uniform(0.045, 0.085)
                mask = (
                    (xx.abs() < thickness)
                    | (yy.abs() < thickness)
                )

            image[mask] = 1.0

            image = torch.clamp(
                image + noise,
                0.0,
                1.0,
            )

            images.append(image.unsqueeze(0))
            labels.append(class_id)
            prompts.append(PROMPTS[class_id])

    return (
        torch.stack(images),
        torch.tensor(labels, dtype=torch.long),
        prompts,
    )


def make_reference_templates():

    axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE)

    yy, xx = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    templates = []

    circle = (
        xx.square() + yy.square()
        <= 0.45 ** 2
    )

    square = (
        (xx.abs() <= 0.45)
        & (yy.abs() <= 0.45)
    )

    diagonal = (
        (xx - yy).abs() < 0.065
    )

    cross = (
        (xx.abs() < 0.065)
        | (yy.abs() < 0.065)
    )

    for mask in [circle, square, diagonal, cross]:
        templates.append(mask.float().unsqueeze(0))

    return torch.stack(templates)


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def show_image_grid(images, titles, title, columns=4):

    images = images.detach().float().cpu()

    count = min(len(images), len(titles))
    rows = math.ceil(count / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.5 * columns, 3.5 * rows),
    )

    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):

        if i < count:
            ax.imshow(
                images[i, 0],
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_title(titles[i])

        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_losses(losses, title):

    plt.figure(figsize=(9, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. AUTOENCODER
# =============================================================================

class ConvAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                IMAGE_CHANNELS, 32,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                32, 64,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                64, LATENT_CHANNELS,
                kernel_size=3, padding=1,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                LATENT_CHANNELS, 64,
                kernel_size=3, padding=1,
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                32, IMAGE_CHANNELS,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


def train_autoencoder(model, images):

    model_device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=AE_LEARNING_RATE,
    )

    images_cpu = images.detach().cpu()
    losses = []

    model.train()

    for step in range(AE_TRAIN_STEPS):

        indices = torch.randint(
            0,
            len(images_cpu),
            (BATCH_SIZE,),
            device="cpu",
        )

        batch = images_cpu[indices].to(model_device)

        reconstruction = model(batch)

        loss = F.mse_loss(
            reconstruction,
            batch,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(
                f"AE step {step + 1}/{AE_TRAIN_STEPS} "
                f"loss={loss.item():.6f}"
            )

    return losses


# =============================================================================
# 7. LATENT NORMALIZATION
# =============================================================================

def calculate_latent_statistics(latents):

    # Per-channel statistics, shared across spatial positions.
    mean = latents.mean(
        dim=(0, 2, 3),
        keepdim=True,
    )

    std = latents.std(
        dim=(0, 2, 3),
        keepdim=True,
        unbiased=False,
    )

    std = std.clamp_min(1e-5)

    return mean.detach(), std.detach()


def normalize_latents(latents, mean, std):
    return (latents - mean) / std


def denormalize_latents(normalized_latents, mean, std):
    return normalized_latents * std + mean


# =============================================================================
# 8. TEXT EMBEDDING
# =============================================================================

class TextEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        max_length,
        pad_id,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_id,
        )

        self.position_embedding = nn.Parameter(
            torch.randn(1, max_length, embed_dim) * 0.02
        )

        self.max_length = max_length

    def forward(self, token_ids):

        sequence_length = token_ids.shape[1]

        if sequence_length > self.max_length:
            raise ValueError("Sequence exceeds maximum length")

        token_vectors = self.token_embedding(token_ids)

        position_vectors = self.position_embedding[
            :, :sequence_length, :
        ]

        return token_vectors + position_vectors


# =============================================================================
# 9. TRANSFORMER TEXT ENCODER
# =============================================================================

class TextEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = TextEmbedding(
            VOCAB_SIZE,
            TEXT_EMBED_DIM,
            MAX_TEXT_LENGTH,
            TOKENIZER.pad_id,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=TEXT_EMBED_DIM,
            nhead=TEXT_NUM_HEADS,
            dim_feedforward=TEXT_FF_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=TEXT_NUM_LAYERS,
            enable_nested_tensor=False,
        )

        self.final_norm = nn.LayerNorm(
            TEXT_EMBED_DIM
        )

    def forward(self, token_ids, attention_mask):

        x = self.embedding(token_ids)

        padding_mask = ~attention_mask.bool()

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.final_norm(x)

        x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        return x


# =============================================================================
# 10. TIMESTEP EMBEDDING
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, timesteps):

        half_dim = self.embedding_dim // 2

        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / max(half_dim - 1, 1)
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

        if embedding.shape[1] < self.embedding_dim:
            embedding = F.pad(
                embedding,
                (0, self.embedding_dim - embedding.shape[1]),
            )

        return self.projection(embedding)


# =============================================================================
# 11. GROUP NORMALIZATION
# =============================================================================

def valid_group_count(channels, preferred_groups=8):

    preferred_groups = min(
        preferred_groups,
        channels,
    )

    for groups in range(preferred_groups, 0, -1):
        if channels % groups == 0:
            return groups

    return 1


# =============================================================================
# 12. CONDITIONAL RESIDUAL BLOCK
# =============================================================================

class ConditionalResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        condition_dim,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            valid_group_count(in_channels),
            in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            valid_group_count(out_channels),
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
        )

        self.condition_projection = nn.Linear(
            condition_dim,
            out_channels,
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, condition):

        residual = self.skip(x)

        h = self.conv1(
            F.silu(self.norm1(x))
        )

        condition_bias = self.condition_projection(
            condition
        ).unsqueeze(-1).unsqueeze(-1)

        h = h + condition_bias

        h = self.conv2(
            F.silu(self.norm2(h))
        )

        return h + residual


# =============================================================================
# 13. TEXT CROSS-ATTENTION
# =============================================================================

class TextCrossAttention(nn.Module):

    def __init__(
        self,
        image_channels,
        text_dim,
        num_heads,
    ):
        super().__init__()

        if image_channels % num_heads != 0:
            raise ValueError(
                "Image channels must be divisible by attention heads"
            )

        self.image_norm = nn.LayerNorm(image_channels)
        self.text_norm = nn.LayerNorm(text_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=image_channels,
            num_heads=num_heads,
            kdim=text_dim,
            vdim=text_dim,
            batch_first=True,
        )

        self.output_projection = nn.Linear(
            image_channels,
            image_channels,
        )

    def forward(
        self,
        image_features,
        text_features,
        attention_mask,
        return_attention=False,
    ):

        batch, channels, height, width = image_features.shape

        queries = image_features.flatten(
            2
        ).transpose(1, 2)

        queries = self.image_norm(queries)

        text = self.text_norm(text_features)

        key_padding_mask = ~attention_mask.bool()

        attended, weights = self.attention(
            query=queries,
            key=text,
            value=text,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        attended = self.output_projection(attended)

        attended = attended.transpose(
            1, 2
        ).reshape(
            batch,
            channels,
            height,
            width,
        )

        output = image_features + attended

        if return_attention:
            return output, weights

        return output


# =============================================================================
# 14. TEXT-CONDITIONED U-NET
# =============================================================================

class TextConditionedUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(
            TIME_EMBED_DIM
        )

        self.input_conv = nn.Conv2d(
            LATENT_CHANNELS,
            BASE_CHANNELS,
            3,
            padding=1,
        )

        self.encoder_block = ConditionalResidualBlock(
            BASE_CHANNELS,
            BASE_CHANNELS,
            TIME_EMBED_DIM,
        )

        self.downsample = nn.Conv2d(
            BASE_CHANNELS,
            BASE_CHANNELS * 2,
            3,
            stride=2,
            padding=1,
        )

        self.middle_block = ConditionalResidualBlock(
            BASE_CHANNELS * 2,
            BASE_CHANNELS * 2,
            TIME_EMBED_DIM,
        )

        self.cross_attention = TextCrossAttention(
            BASE_CHANNELS * 2,
            TEXT_EMBED_DIM,
            CROSS_ATTENTION_HEADS,
        )

        self.decoder_block = ConditionalResidualBlock(
            BASE_CHANNELS * 2,
            BASE_CHANNELS,
            TIME_EMBED_DIM,
        )

        self.upsample = nn.ConvTranspose2d(
            BASE_CHANNELS,
            BASE_CHANNELS,
            4,
            stride=2,
            padding=1,
        )

        self.output_norm = nn.GroupNorm(
            valid_group_count(BASE_CHANNELS),
            BASE_CHANNELS,
        )

        self.output_conv = nn.Conv2d(
            BASE_CHANNELS,
            LATENT_CHANNELS,
            3,
            padding=1,
        )

    def forward(
        self,
        noisy_latents,
        timesteps,
        text_features,
        attention_mask,
        return_attention=False,
    ):

        time_features = self.time_embedding(timesteps)

        h = self.input_conv(noisy_latents)

        h = self.encoder_block(
            h,
            time_features,
        )

        skip = h

        h = self.downsample(h)

        h = self.middle_block(
            h,
            time_features,
        )

        if return_attention:
            h, attention_weights = self.cross_attention(
                h,
                text_features,
                attention_mask,
                return_attention=True,
            )
        else:
            h = self.cross_attention(
                h,
                text_features,
                attention_mask,
            )

        h = self.decoder_block(
            h,
            time_features,
        )

        h = self.upsample(h)

        if h.shape != skip.shape:
            raise RuntimeError(
                f"Skip mismatch: {h.shape} vs {skip.shape}"
            )

        h = h + skip

        h = F.silu(self.output_norm(h))

        output = self.output_conv(h)

        if return_attention:
            return output, attention_weights

        return output


# =============================================================================
# 15. DIFFUSION SCHEDULE
# =============================================================================

def make_diffusion_schedule(device):

    betas = torch.linspace(
        BETA_START,
        BETA_END,
        DIFFUSION_STEPS,
        device=device,
        dtype=torch.float32,
    )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0,
    )

    alpha_bars_previous = torch.cat(
        [
            torch.ones(1, device=device),
            alpha_bars[:-1],
        ]
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
            "Schedule and timestep devices do not match"
        )

    result = values.gather(
        0,
        timesteps,
    )

    return result.reshape(
        timesteps.shape[0],
        *([1] * (len(target_shape) - 1)),
    )


# =============================================================================
# 16. FORWARD DIFFUSION
# =============================================================================

def q_sample(
    clean_latents,
    timesteps,
    noise,
    schedule,
):

    sqrt_alpha_bar = extract(
        schedule["sqrt_alpha_bars"],
        timesteps,
        clean_latents.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule["sqrt_one_minus_alpha_bars"],
        timesteps,
        clean_latents.shape,
    )

    return (
        sqrt_alpha_bar * clean_latents
        + sqrt_one_minus_alpha_bar * noise
    )


# =============================================================================
# 17. DATASET ALIGNMENT
# =============================================================================

def validate_aligned_dataset(
    latent_dataset,
    token_ids,
    attention_masks,
):

    if latent_dataset.shape[0] != token_ids.shape[0]:
        raise ValueError(
            f"{latent_dataset.shape[0]} latents but "
            f"{token_ids.shape[0]} token sequences"
        )

    if latent_dataset.shape[0] != attention_masks.shape[0]:
        raise ValueError(
            f"{latent_dataset.shape[0]} latents but "
            f"{attention_masks.shape[0]} attention masks"
        )

    if token_ids.shape != attention_masks.shape:
        raise ValueError(
            "Token IDs and attention masks must have equal shapes"
        )

    return True


# =============================================================================
# 18. TRAIN TEXT-CONDITIONED DIFFUSION
# =============================================================================

def train_text_conditioned_diffusion(
    diffusion_model,
    text_encoder,
    normalized_latents,
    token_ids_cpu,
    attention_masks_cpu,
):

    model_device = next(
        diffusion_model.parameters()
    ).device

    text_device = next(
        text_encoder.parameters()
    ).device

    if model_device != text_device:
        raise RuntimeError(
            "Text encoder and diffusion model must share a device"
        )

    validate_aligned_dataset(
        normalized_latents,
        token_ids_cpu,
        attention_masks_cpu,
    )

    normalized_latents = normalized_latents.to(model_device)

    token_ids_device = token_ids_cpu.to(
        model_device,
        dtype=torch.long,
    )

    masks_device = attention_masks_cpu.to(
        model_device,
        dtype=torch.bool,
    )

    validate_aligned_dataset(
        normalized_latents,
        token_ids_device,
        masks_device,
    )

    schedule = make_diffusion_schedule(model_device)

    optimizer = torch.optim.AdamW(
        list(diffusion_model.parameters())
        + list(text_encoder.parameters()),
        lr=DIFFUSION_LEARNING_RATE,
        weight_decay=1e-5,
    )

    diffusion_model.train()
    text_encoder.train()

    losses = []

    for step in range(DIFFUSION_TRAIN_STEPS):

        indices = torch.randint(
            0,
            normalized_latents.shape[0],
            (BATCH_SIZE,),
            device=model_device,
        )

        clean_latents = normalized_latents[indices]
        batch_tokens = token_ids_device[indices]
        batch_masks = masks_device[indices]

        timesteps = torch.randint(
            0,
            DIFFUSION_STEPS,
            (BATCH_SIZE,),
            device=model_device,
            dtype=torch.long,
        )

        noise = torch.randn_like(clean_latents)

        noisy_latents = q_sample(
            clean_latents,
            timesteps,
            noise,
            schedule,
        )

        text_features = text_encoder(
            batch_tokens,
            batch_masks,
        )

        predicted_noise = diffusion_model(
            noisy_latents,
            timesteps,
            text_features,
            batch_masks,
        )

        loss = F.mse_loss(
            predicted_noise,
            noise,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(diffusion_model.parameters())
            + list(text_encoder.parameters()),
            1.0,
        )

        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(
                f"Diffusion step {step + 1}/"
                f"{DIFFUSION_TRAIN_STEPS} "
                f"loss={loss.item():.6f}"
            )

    return losses


# =============================================================================
# 19. DDPM SAMPLING
# =============================================================================

@torch.no_grad()
def sample_text_conditioned(
    diffusion_model,
    text_encoder,
    token_ids,
    attention_mask,
    schedule,
    num_samples,
):

    model_device = next(
        diffusion_model.parameters()
    ).device

    text_device = next(
        text_encoder.parameters()
    ).device

    if model_device != text_device:
        raise RuntimeError("Model device mismatch")

    token_ids = token_ids.to(
        model_device,
        dtype=torch.long,
    )

    attention_mask = attention_mask.to(
        model_device,
        dtype=torch.bool,
    )

    schedule = {
        key: value.to(model_device)
        for key, value in schedule.items()
    }

    if token_ids.shape[0] == 1 and num_samples > 1:
        token_ids = token_ids.expand(num_samples, -1)
        attention_mask = attention_mask.expand(num_samples, -1)

    if token_ids.shape[0] != num_samples:
        raise ValueError(
            "Prompt batch size must equal num_samples"
        )

    diffusion_model.eval()
    text_encoder.eval()

    text_features = text_encoder(
        token_ids,
        attention_mask,
    )

    # The training data was normalized to approximately N(0,I).
    # Start the reverse process from the matching Gaussian prior.
    latents = torch.randn(
        num_samples,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=model_device,
    )

    for timestep in reversed(range(DIFFUSION_STEPS)):

        t = torch.full(
            (num_samples,),
            timestep,
            device=model_device,
            dtype=torch.long,
        )

        predicted_noise = diffusion_model(
            latents,
            t,
            text_features,
            attention_mask,
        )

        beta_t = extract(
            schedule["betas"],
            t,
            latents.shape,
        )

        sqrt_one_minus_alpha_bar = extract(
            schedule["sqrt_one_minus_alpha_bars"],
            t,
            latents.shape,
        )

        sqrt_recip_alpha = extract(
            schedule["sqrt_recip_alphas"],
            t,
            latents.shape,
        )

        mean = sqrt_recip_alpha * (
            latents
            - beta_t * predicted_noise
            / sqrt_one_minus_alpha_bar
        )

        if timestep > 0:

            variance = extract(
                schedule["posterior_variance"],
                t,
                latents.shape,
            )

            latents = (
                mean
                + torch.sqrt(
                    torch.clamp(variance, min=0.0)
                ) * torch.randn_like(latents)
            )

        else:
            latents = mean

    return latents


# =============================================================================
# 20. SHAPE QUALITY EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_shape_templates(generated_images):

    templates = make_reference_templates().to(
        generated_images.device
    )

    generated_binary = generated_images > 0.35
    template_binary = templates > 0.5

    results = []

    for image_index in range(generated_images.shape[0]):

        image = generated_binary[image_index]

        scores = []

        for class_index in range(NUM_CLASSES):

            template = template_binary[class_index]

            intersection = (
                image & template
            ).sum().float()

            union = (
                image | template
            ).sum().float()

            iou = (
                intersection / union.clamp_min(1.0)
            ).item()

            scores.append(iou)

        predicted_class = int(np.argmax(scores))

        results.append({
            "predicted_class": predicted_class,
            "scores": scores,
            "foreground_fraction": image.float().mean().item(),
            "mean_intensity": generated_images[
                image_index
            ].mean().item(),
            "max_intensity": generated_images[
                image_index
            ].max().item(),
        })

    return results


def print_shape_evaluation(generated_images):

    results = evaluate_shape_templates(generated_images)

    print("\nGENERATED SHAPE QUALITY")

    correct = 0

    for index, result in enumerate(results):

        predicted_name = CLASS_NAMES[
            result["predicted_class"]
        ]

        expected_name = CLASS_NAMES[index]

        is_correct = (
            result["predicted_class"] == index
        )

        correct += int(is_correct)

        print(
            f"Prompt: {PROMPTS[index]:20s} "
            f"Predicted: {predicted_name:10s} "
            f"Foreground: {result['foreground_fraction']:.3f} "
            f"Mean: {result['mean_intensity']:.3f} "
            f"Max: {result['max_intensity']:.3f} "
            f"{'PASS' if is_correct else 'CHECK'}"
        )

    print(
        f"Template classification: "
        f"{correct}/{len(results)}"
    )

    return results


# =============================================================================
# 21. UNIT TESTS
# =============================================================================

def test_tokenizer():

    print("\nTEST 1 — TOKENIZER")

    ids, mask = TOKENIZER.encode("make a circle")

    assert ids.tolist() == [
        VOCAB["make"],
        VOCAB["a"],
        VOCAB["circle"],
        VOCAB[PAD_TOKEN],
    ]

    assert mask.tolist() == [True, True, True, False]

    print("PASS")


def test_batch_tokenization():

    print("\nTEST 2 — BATCH TOKENIZATION")

    ids, masks = TOKENIZER.batch_encode(PROMPTS)

    assert ids.shape == (4, MAX_TEXT_LENGTH)
    assert masks.shape == ids.shape

    print("PASS")


def test_text_encoder():

    print("\nTEST 3 — TEXT ENCODER")

    model = TextEncoder().to(DEVICE)

    ids, masks = TOKENIZER.batch_encode(PROMPTS)

    output = model(
        ids.to(DEVICE),
        masks.to(DEVICE),
    )

    assert output.shape == (
        4,
        MAX_TEXT_LENGTH,
        TEXT_EMBED_DIM,
    )

    assert torch.isfinite(output).all()

    print("Output:", tuple(output.shape))
    print("PASS")


def test_padding_mask():

    print("\nTEST 4 — PADDING MASK")

    model = TextEncoder().to(DEVICE)
    model.eval()

    ids, masks = TOKENIZER.encode("make a circle")

    ids = ids.unsqueeze(0).to(DEVICE)
    masks = masks.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(ids, masks)

    assert torch.allclose(
        output[~masks],
        torch.zeros_like(output[~masks]),
        atol=1e-6,
    )

    print("PASS")


def test_cross_attention():

    print("\nTEST 5 — CROSS-ATTENTION")

    attention = TextCrossAttention(
        64,
        TEXT_EMBED_DIM,
        4,
    ).to(DEVICE)

    image = torch.randn(
        2, 64, 4, 4,
        device=DEVICE,
    )

    text = torch.randn(
        2, MAX_TEXT_LENGTH, TEXT_EMBED_DIM,
        device=DEVICE,
    )

    masks = torch.ones(
        2, MAX_TEXT_LENGTH,
        dtype=torch.bool,
        device=DEVICE,
    )

    output, weights = attention(
        image,
        text,
        masks,
        return_attention=True,
    )

    assert output.shape == image.shape
    assert weights.shape == (
        2, 4, 16, MAX_TEXT_LENGTH
    )

    print("Output:", tuple(output.shape))
    print("Attention:", tuple(weights.shape))
    print("PASS")


def test_diffusion_schedule():

    print("\nTEST 6 — DIFFUSION SCHEDULE")

    schedule = make_diffusion_schedule(DEVICE)

    assert schedule["betas"].shape == (DIFFUSION_STEPS,)
    assert torch.all(schedule["betas"] > 0)
    assert torch.all(schedule["betas"] < 1)

    final_alpha_bar = schedule["alpha_bars"][-1].item()

    print("Final alpha_bar:", f"{final_alpha_bar:.6f}")

    assert final_alpha_bar < 0.1

    print("PASS")


def test_forward_diffusion():

    print("\nTEST 7 — FORWARD DIFFUSION")

    schedule = make_diffusion_schedule(DEVICE)

    clean = torch.randn(
        4, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
        device=DEVICE,
    )

    noise = torch.randn_like(clean)

    timesteps = torch.tensor(
        [0, 10, 50, DIFFUSION_STEPS - 1],
        device=DEVICE,
        dtype=torch.long,
    )

    noisy = q_sample(
        clean,
        timesteps,
        noise,
        schedule,
    )

    assert noisy.shape == clean.shape
    assert torch.isfinite(noisy).all()

    print("PASS")


def test_latent_normalization():

    print("\nTEST 8 — LATENT NORMALIZATION")

    latents = (
        torch.randn(
            128, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE
        ) * 2.5 + 3.0
    )

    mean, std = calculate_latent_statistics(latents)

    normalized = normalize_latents(
        latents,
        mean,
        std,
    )

    recovered = denormalize_latents(
        normalized,
        mean,
        std,
    )

    assert torch.allclose(
        recovered,
        latents,
        atol=1e-5,
    )

    assert abs(normalized.mean().item()) < 0.03
    assert abs(normalized.std().item() - 1.0) < 0.03

    print("Normalized mean:", normalized.mean().item())
    print("Normalized std:", normalized.std().item())
    print("PASS")


def test_unet():

    print("\nTEST 9 — U-NET OUTPUT")

    text_encoder = TextEncoder().to(DEVICE)
    model = TextConditionedUNet().to(DEVICE)

    ids, masks = TOKENIZER.batch_encode(PROMPTS[:2])

    ids = ids.to(DEVICE)
    masks = masks.to(DEVICE)

    text_features = text_encoder(ids, masks)

    latents = torch.randn(
        2, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
        device=DEVICE,
    )

    timesteps = torch.tensor(
        [10, 50],
        device=DEVICE,
        dtype=torch.long,
    )

    output = model(
        latents,
        timesteps,
        text_features,
        masks,
    )

    assert output.shape == latents.shape
    assert torch.isfinite(output).all()

    print("Output:", tuple(output.shape))
    print("PASS")


def test_gradient_propagation():

    print("\nTEST 10 — GRADIENT PROPAGATION")

    text_encoder = TextEncoder().to(DEVICE)
    diffusion_model = TextConditionedUNet().to(DEVICE)

    ids, masks = TOKENIZER.batch_encode(PROMPTS[:2])

    ids = ids.to(DEVICE)
    masks = masks.to(DEVICE)

    latents = torch.randn(
        2, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
        device=DEVICE,
    )

    timesteps = torch.tensor(
        [5, 30],
        device=DEVICE,
        dtype=torch.long,
    )

    text_features = text_encoder(ids, masks)

    output = diffusion_model(
        latents,
        timesteps,
        text_features,
        masks,
    )

    output.square().mean().backward()

    text_gradients = [
        p.grad for p in text_encoder.parameters()
        if p.grad is not None
    ]

    diffusion_gradients = [
        p.grad for p in diffusion_model.parameters()
        if p.grad is not None
    ]

    assert len(text_gradients) > 0
    assert len(diffusion_gradients) > 0

    assert any(
        g.abs().sum().item() > 0
        for g in text_gradients
    )

    assert any(
        g.abs().sum().item() > 0
        for g in diffusion_gradients
    )

    print("Text gradient tensors:", len(text_gradients))
    print("Diffusion gradient tensors:", len(diffusion_gradients))
    print("PASS")


def test_dataset_alignment():

    print("\nTEST 11 — DATASET ALIGNMENT")

    latents = torch.randn(
        8, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE
    )

    tokens = torch.zeros(
        8, MAX_TEXT_LENGTH,
        dtype=torch.long,
    )

    masks = torch.ones(
        8, MAX_TEXT_LENGTH,
        dtype=torch.bool,
    )

    assert validate_aligned_dataset(
        latents,
        tokens,
        masks,
    )

    caught = False

    try:
        validate_aligned_dataset(
            latents,
            tokens[:4],
            masks,
        )
    except ValueError:
        caught = True

    assert caught

    print("PASS")


def test_cpu_token_device_indexing():

    print("\nTEST 12 — CPU TOKENS / MODEL-DEVICE INDEXING")

    test_prompts = PROMPTS * 2

    tokens_cpu, masks_cpu = TOKENIZER.batch_encode(
        test_prompts
    )

    latents = torch.randn(
        len(test_prompts),
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=DEVICE,
    )

    validate_aligned_dataset(
        latents,
        tokens_cpu,
        masks_cpu,
    )

    tokens = tokens_cpu.to(DEVICE)
    masks = masks_cpu.to(DEVICE)

    indices = torch.tensor(
        [7, 2, 5, 0],
        device=DEVICE,
        dtype=torch.long,
    )

    batch_latents = latents[indices]
    batch_tokens = tokens[indices]
    batch_masks = masks[indices]

    expected_tokens = tokens_cpu[indices.cpu()]
    expected_masks = masks_cpu[indices.cpu()]

    assert torch.equal(batch_tokens.cpu(), expected_tokens)
    assert torch.equal(batch_masks.cpu(), expected_masks)

    assert batch_latents.shape[0] == batch_tokens.shape[0]

    print("PASS")


def test_complete_sampler():

    print("\nTEST 13 — COMPLETE SAMPLER")

    text_encoder = TextEncoder().to(DEVICE)
    diffusion_model = TextConditionedUNet().to(DEVICE)

    schedule = make_diffusion_schedule(DEVICE)

    ids, masks = TOKENIZER.batch_encode(PROMPTS[:2])

    generated = sample_text_conditioned(
        diffusion_model,
        text_encoder,
        ids,
        masks,
        schedule,
        num_samples=2,
    )

    assert generated.shape == (
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    assert torch.isfinite(generated).all()

    print("Generated:", tuple(generated.shape))
    print("PASS")


def run_all_tests():

    tests = [
        test_tokenizer,
        test_batch_tokenization,
        test_text_encoder,
        test_padding_mask,
        test_cross_attention,
        test_diffusion_schedule,
        test_forward_diffusion,
        test_latent_normalization,
        test_unet,
        test_gradient_propagation,
        test_dataset_alignment,
        test_cpu_token_device_indexing,
        test_complete_sampler,
    ]

    print("\nRUNNING UNIT TESTS")

    for test in tests:
        test()

    print("\nALL UNIT TESTS PASSED")


# =============================================================================
# 22. GENERATE ALL FOUR PROMPTS
# =============================================================================

@torch.no_grad()
def generate_all_prompts(
    diffusion_model,
    text_encoder,
    autoencoder,
    latent_mean,
    latent_std,
):

    model_device = next(
        diffusion_model.parameters()
    ).device

    token_ids, masks = TOKENIZER.batch_encode(PROMPTS)

    schedule = make_diffusion_schedule(model_device)

    normalized_latents = sample_text_conditioned(
        diffusion_model,
        text_encoder,
        token_ids,
        masks,
        schedule,
        num_samples=NUM_CLASSES,
    )

    # Critical correction:
    # The decoder was trained on original, unnormalized AE latents.
    # Convert sampled normalized latents back to that distribution.
    decoded_latents = denormalize_latents(
        normalized_latents,
        latent_mean,
        latent_std,
    )

    images = autoencoder.decode(decoded_latents)

    images = images.clamp(0.0, 1.0)

    return images


# =============================================================================
# 23. SAME NOISE / DIFFERENT TEXT EXPERIMENT
# =============================================================================

@torch.no_grad()
def experiment_same_noise_different_text(
    diffusion_model,
    text_encoder,
    autoencoder,
    latent_mean,
    latent_std,
):

    print("\nEXPERIMENT — SAME INITIAL NOISE / DIFFERENT TEXT")

    model_device = next(
        diffusion_model.parameters()
    ).device

    schedule = make_diffusion_schedule(model_device)

    ids, masks = TOKENIZER.batch_encode(PROMPTS)

    ids = ids.to(model_device)
    masks = masks.to(model_device)

    text_encoder.eval()
    diffusion_model.eval()
    autoencoder.eval()

    text_features = text_encoder(ids, masks)

    base_noise = torch.randn(
        1,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=model_device,
    )

    latents = base_noise.repeat(
        NUM_CLASSES,
        1,
        1,
        1,
    )

    for timestep in reversed(range(DIFFUSION_STEPS)):

        t = torch.full(
            (NUM_CLASSES,),
            timestep,
            device=model_device,
            dtype=torch.long,
        )

        predicted_noise = diffusion_model(
            latents,
            t,
            text_features,
            masks,
        )

        beta_t = extract(
            schedule["betas"],
            t,
            latents.shape,
        )

        sqrt_one_minus_alpha_bar = extract(
            schedule["sqrt_one_minus_alpha_bars"],
            t,
            latents.shape,
        )

        sqrt_recip_alpha = extract(
            schedule["sqrt_recip_alphas"],
            t,
            latents.shape,
        )

        mean = sqrt_recip_alpha * (
            latents
            - beta_t * predicted_noise
            / sqrt_one_minus_alpha_bar
        )

        if timestep > 0:

            variance = extract(
                schedule["posterior_variance"],
                t,
                latents.shape,
            )

            # Shared random noise isolates the effect of text conditioning.
            shared_noise = torch.randn_like(
                latents[:1]
            ).expand_as(latents)

            latents = (
                mean
                + torch.sqrt(
                    torch.clamp(variance, min=0.0)
                ) * shared_noise
            )

        else:
            latents = mean

    decoded_latents = denormalize_latents(
        latents,
        latent_mean,
        latent_std,
    )

    images = autoencoder.decode(
        decoded_latents
    ).clamp(0.0, 1.0)

    differences = []

    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            differences.append(
                torch.mean(
                    torch.abs(images[i] - images[j])
                ).item()
            )

    print("Prompts:", PROMPTS)
    print("Pairwise image differences:", differences)

    show_image_grid(
        images,
        PROMPTS,
        "Same initial noise — different text prompts",
    )

    return images


# =============================================================================
# 24. MAIN
# =============================================================================

def main():

    set_seed()

    print("=" * 80)
    print("LESSON 26 — TEXT CONDITIONING")
    print("=" * 80)

    print("PyTorch:", torch.__version__)
    print("Configured device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Vocabulary size:", VOCAB_SIZE)
    print("Text embedding dimension:", TEXT_EMBED_DIM)
    print("Diffusion steps:", DIFFUSION_STEPS)
    print("Diffusion training steps:", DIFFUSION_TRAIN_STEPS)

    # -------------------------------------------------------------------------
    # UNIT TESTS
    # -------------------------------------------------------------------------

    run_all_tests()

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------

    print("\nCREATING SYNTHETIC DATASET")

    images, labels, prompts = create_synthetic_dataset(
        samples_per_class=256
    )

    print("Images:", tuple(images.shape))
    print("Labels:", tuple(labels.shape))
    print("Prompt rows:", len(prompts))

    show_image_grid(
        images[:16],
        [
            PROMPTS[label]
            for label in labels[:16].tolist()
        ],
        "Synthetic training dataset",
    )

    # -------------------------------------------------------------------------
    # TOKENIZE DATASET PROMPTS
    # -------------------------------------------------------------------------

    print("\nTOKENIZING TRAINING PROMPTS")

    token_ids, attention_masks = TOKENIZER.batch_encode(
        prompts
    )

    print("Token IDs:", tuple(token_ids.shape))
    print("Attention masks:", tuple(attention_masks.shape))

    # -------------------------------------------------------------------------
    # AUTOENCODER
    # -------------------------------------------------------------------------

    print("\nTRAINING AUTOENCODER")

    autoencoder = ConvAutoencoder().to(DEVICE)

    start_time = time.time()

    ae_losses = train_autoencoder(
        autoencoder,
        images,
    )

    print(
        "AE training time:",
        f"{time.time() - start_time:.2f} seconds",
    )

    plot_losses(
        ae_losses,
        "Autoencoder training loss",
    )

    # -------------------------------------------------------------------------
    # AUTOENCODER RECONSTRUCTION
    # -------------------------------------------------------------------------

    print("\nCHECKING AUTOENCODER RECONSTRUCTION")

    autoencoder.eval()

    images_device = images.to(DEVICE)

    with torch.no_grad():
        reconstruction = autoencoder(
            images_device[:8]
        )

    reconstruction_mse = F.mse_loss(
        reconstruction,
        images_device[:8],
    ).item()

    print("Reconstruction MSE:", f"{reconstruction_mse:.6f}")
    print(
        "Reconstruction intensity:",
        f"min={reconstruction.min().item():.4f}",
        f"max={reconstruction.max().item():.4f}",
        f"mean={reconstruction.mean().item():.4f}",
    )

    show_image_grid(
        reconstruction,
        ["Reconstruction"] * len(reconstruction),
        "Autoencoder reconstruction",
    )

    # -------------------------------------------------------------------------
    # ENCODE DATASET
    # -------------------------------------------------------------------------

    print("\nENCODING DATASET INTO LATENT SPACE")

    with torch.no_grad():
        latent_dataset = autoencoder.encode(
            images_device
        )

    print("Raw latent dataset:", tuple(latent_dataset.shape))
    print("Raw latent mean:", latent_dataset.mean().item())
    print("Raw latent std:", latent_dataset.std().item())

    # -------------------------------------------------------------------------
    # LATENT NORMALIZATION
    # -------------------------------------------------------------------------

    print("\nNORMALIZING LATENT DATASET")

    latent_mean, latent_std = calculate_latent_statistics(
        latent_dataset
    )

    normalized_latents = normalize_latents(
        latent_dataset,
        latent_mean,
        latent_std,
    )

    print("Latent mean per channel:", latent_mean.flatten().tolist())
    print("Latent std per channel:", latent_std.flatten().tolist())
    print("Normalized mean:", normalized_latents.mean().item())
    print("Normalized std:", normalized_latents.std().item())

    assert torch.isfinite(normalized_latents).all()
    assert abs(normalized_latents.mean().item()) < 0.05

    # -------------------------------------------------------------------------
    # DATASET ALIGNMENT
    # -------------------------------------------------------------------------

    print("\nVALIDATING LATENT / TEXT ALIGNMENT")

    validate_aligned_dataset(
        normalized_latents,
        token_ids,
        attention_masks,
    )

    print("Latent rows:", normalized_latents.shape[0])
    print("Text rows:", token_ids.shape[0])
    print("PASS")

    # -------------------------------------------------------------------------
    # TEXT ENCODER
    # -------------------------------------------------------------------------

    print("\nBUILDING TEXT ENCODER")

    text_encoder = TextEncoder().to(DEVICE)

    text_parameters = sum(
        p.numel()
        for p in text_encoder.parameters()
    )

    print("Text encoder parameters:", text_parameters)

    # -------------------------------------------------------------------------
    # DIFFUSION MODEL
    # -------------------------------------------------------------------------

    print("\nBUILDING TEXT-CONDITIONED DIFFUSION U-NET")

    diffusion_model = TextConditionedUNet().to(DEVICE)

    diffusion_parameters = sum(
        p.numel()
        for p in diffusion_model.parameters()
    )

    print("Diffusion parameters:", diffusion_parameters)

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    print("\nTRAINING TEXT-CONDITIONED DIFFUSION")

    start_time = time.time()

    diffusion_losses = train_text_conditioned_diffusion(
        diffusion_model,
        text_encoder,
        normalized_latents,
        token_ids,
        attention_masks,
    )

    print(
        "Diffusion training time:",
        f"{time.time() - start_time:.2f} seconds",
    )

    plot_losses(
        diffusion_losses,
        "Text-conditioned diffusion training loss",
    )

    # -------------------------------------------------------------------------
    # GENERATION
    # -------------------------------------------------------------------------

    print("\nGENERATING IMAGES FROM TEXT")

    diffusion_model.eval()
    text_encoder.eval()
    autoencoder.eval()

    generated_images = generate_all_prompts(
        diffusion_model,
        text_encoder,
        autoencoder,
        latent_mean,
        latent_std,
    )

    print(
        "Generated image range:",
        f"min={generated_images.min().item():.4f}",
        f"max={generated_images.max().item():.4f}",
        f"mean={generated_images.mean().item():.4f}",
    )

    show_image_grid(
        generated_images,
        PROMPTS,
        "Text-conditioned generated images",
    )

    # -------------------------------------------------------------------------
    # QUANTITATIVE SHAPE EVALUATION
    # -------------------------------------------------------------------------

    print_shape_evaluation(
        generated_images
    )

    # -------------------------------------------------------------------------
    # SAME NOISE / DIFFERENT TEXT
    # -------------------------------------------------------------------------

    experiment_images = experiment_same_noise_different_text(
        diffusion_model,
        text_encoder,
        autoencoder,
        latent_mean,
        latent_std,
    )

    print("\nSAME-NOISE EXPERIMENT EVALUATION")

    print_shape_evaluation(
        experiment_images
    )

    # -------------------------------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LESSON 26 COMPLETE")
    print("=" * 80)

    print("Tokenizer: custom vocabulary")
    print("Text encoder: random-initialized Transformer")
    print("Conditioning: multi-head cross-attention")
    print("Generator: latent diffusion U-Net")
    print("Latent normalization: enabled")
    print("Latent denormalization before decoding: enabled")
    print("Pretrained components: NONE")

    print("\nWHAT YOU BUILT:")
    print("1. Tokenizer")
    print("2. Token and positional embeddings")
    print("3. Transformer text encoder")
    print("4. Text cross-attention")
    print("5. Normalized latent diffusion")
    print("6. DDPM sampler")
    print("7. Shape-template evaluation")

    print("\nNEXT LESSON:")
    print("LESSON 27 — BUILD OUR OWN TEXT-TO-IMAGE MODEL")


if __name__ == "__main__":
    main()
