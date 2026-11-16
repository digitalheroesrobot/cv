
"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 26 — TEXT CONDITIONING (IMPROVED VERSION)
================================================================================

PURPOSE
-------
Build a more reliable text-conditioned latent diffusion model from scratch.
This version corrects latent scaling, strengthens text conditioning, and
adds independent semantic evaluation.

WHAT YOU WILL LEARN
-------------------
1. Tokenization, vocabulary, padding, and attention masks.
2. Learned token and positional embeddings.
3. Transformer text encoding.
4. Multi-resolution text cross-attention.
5. Autoencoder latent normalization.
6. Classifier-free condition dropout and guidance.
7. Text-conditioned latent DDPM.
8. Validation loss and independent semantic evaluation.

TECHNICAL ROADMAP
-----------------
Lesson 23: Latent diffusion.
Lesson 24: Class conditioning.
Lesson 25: Cross-attention.
Lesson 26: Text conditioning.
Lesson 27: Build our own text-to-image model.

REAL-WORLD PROBLEM
------------------
A robot receives a visual instruction such as:
    "make a circle"
    "make a square"

The visual generator must produce an image that matches the instruction.

THEORY
------
Text:
    H = Transformer(TokenEmbedding + PositionEmbedding)

Cross-attention:
    Q = ImageFeatures
    K = TextFeatures
    V = TextFeatures

    Attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V

Latent normalization:
    z_normalized = (z - mean) / std
    z = z_normalized * std + mean

Forward diffusion:
    z_t = sqrt(alpha_bar_t) z_0
          + sqrt(1-alpha_bar_t) epsilon

Training objective:
    L = E[||epsilon - epsilon_theta(z_t,t,H)||^2]

Classifier-free guidance:
    epsilon_guided =
        epsilon_unconditional
        + guidance_scale *
          (epsilon_conditional - epsilon_unconditional)

IMPLEMENTATION
--------------
- Controlled synthetic shape dataset.
- Convolutional autoencoder.
- Per-channel latent normalization.
- Custom tokenizer.
- Random-initialized Transformer text encoder.
- Two-resolution text cross-attention.
- Classifier-free condition dropout.
- Guided DDPM sampling.
- Independent shape classifier.
- Validation loss and semantic evaluation.

VERIFICATION
------------
Tests cover tokenizer, padding, text encoding, attention, latent
normalization, diffusion, gradients, device consistency, dataset
alignment, classifier-free guidance, and complete sampling.

REAL-WORLD APPLICATION
----------------------
This is a simplified language-conditioned image generation pipeline.
The same general principle is useful when translating robot instructions
into visual goals or conditioning a perception/generation model.

REAL-WORLD RESULT
-----------------
The model must generate images that correspond to their requested
shape categories, not merely different images for different prompts.

LESSON RESULT
-------------
A more stable, evaluated, text-conditioned latent diffusion pipeline.

CONNECTION TO NEXT LESSON
-------------------------
Lesson 27: Build our own text-to-image model.

IMPORTANT
---------
This is a four-class educational model, not general-purpose language
understanding. Every model is randomly initialized. No pretrained
weights, pretrained encoders, or pretrained checkpoints are used.
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

DIFFUSION_STEPS = 100
BETA_START = 1e-4
BETA_END = 0.02

SAMPLES_PER_CLASS = 512
BATCH_SIZE = 64

AE_TRAIN_STEPS = 600
DIFFUSION_TRAIN_STEPS = 4000
CLASSIFIER_TRAIN_STEPS = 700

LEARNING_RATE = 1e-3
CLASSIFIER_FREE_DROPOUT = 0.15
GUIDANCE_SCALE = 3.0

VALIDATION_INTERVAL = 200

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROMPTS = [
    "make a circle",
    "make a square",
    "make a diagonal",
    "make a cross",
]


# =============================================================================
# 2. REPRODUCIBILITY AND DEVICE HELPERS
# =============================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model):
    return next(model.parameters()).device


def valid_group_count(channels, preferred_groups=8):
    preferred_groups = min(preferred_groups, channels)

    for groups in range(preferred_groups, 0, -1):
        if channels % groups == 0:
            return groups

    return 1


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

        ids += [self.pad_id] * (
            self.max_length - valid_length
        )

        mask = (
            [True] * valid_length
            + [False] * (self.max_length - valid_length)
        )

        # An empty prompt must still have one valid token.
        if valid_length == 0:
            ids[0] = self.unk_id
            mask[0] = True

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )

    def batch_encode(self, texts):
        encoded = [self.encode(text) for text in texts]

        return (
            torch.stack([item[0] for item in encoded]),
            torch.stack([item[1] for item in encoded]),
        )


TOKENIZER = SimpleTokenizer(
    VOCABULARY,
    MAX_TEXT_LENGTH,
)

NULL_TOKEN_IDS, NULL_ATTENTION_MASK = TOKENIZER.encode(
    "unknown"
)


# =============================================================================
# 4. CONTROLLED SYNTHETIC DATASET
# =============================================================================

def create_synthetic_dataset(samples_per_class=SAMPLES_PER_CLASS):
    """
    Shape coordinates are translated and rotated.

    Class 0: circle
    Class 1: square
    Class 2: diagonal line
    Class 3: cross

    This variation helps prevent the model from memorizing one fixed
    position or one fixed size for each class.
    """

    images = []
    labels = []
    prompts = []

    axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")

    for class_id in range(NUM_CLASSES):
        for _ in range(samples_per_class):

            image = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)

            center_x = random.uniform(-0.12, 0.12)
            center_y = random.uniform(-0.12, 0.12)

            x = xx - center_x
            y = yy - center_y

            angle = random.uniform(-math.pi / 12, math.pi / 12)

            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            xr = x * cos_a + y * sin_a
            yr = -x * sin_a + y * cos_a

            if class_id == 0:
                radius = random.uniform(0.35, 0.50)
                distance = torch.sqrt(x.square() + y.square())
                mask = distance <= radius

            elif class_id == 1:
                half_width = random.uniform(0.34, 0.48)
                mask = (
                    (xr.abs() <= half_width)
                    & (yr.abs() <= half_width)
                )

            elif class_id == 2:
                thickness = random.uniform(0.045, 0.085)
                mask = yr.abs() <= thickness

            else:
                thickness = random.uniform(0.045, 0.085)
                mask = (
                    (xr.abs() <= thickness)
                    | (yr.abs() <= thickness)
                )

            image[mask] = 1.0

            # Small image noise prevents an unrealistically perfect
            # training distribution without destroying shape structure.
            image = torch.clamp(
                image + 0.008 * torch.randn_like(image),
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
        figsize=(3 * columns, 3 * rows),
    )

    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < count:
            ax.imshow(
                images[i, 0],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            ax.set_title(titles[i])

        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, validation_losses=None, title="Training loss"):
    plt.figure(figsize=(9, 4))
    plt.plot(train_losses, label="Training")

    if validation_losses:
        x_values = [
            item[0] for item in validation_losses
        ]
        y_values = [
            item[1] for item in validation_losses
        ]
        plt.plot(
            x_values,
            y_values,
            marker="o",
            label="Validation",
        )

    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
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
    device = model_device(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    images_cpu = images.detach().cpu()
    losses = []

    model.train()

    for step in range(AE_TRAIN_STEPS):
        indices = torch.randint(
            0,
            len(images_cpu),
            (BATCH_SIZE,),
        )

        batch = images_cpu[indices].to(device)

        reconstruction = model(batch)

        loss = (
            F.mse_loss(reconstruction, batch)
            + 0.25 * F.l1_loss(reconstruction, batch)
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
    """
    Compute per-channel statistics over batch and spatial dimensions.

    The diffusion process starts from standard Gaussian noise.
    Therefore, the data presented to diffusion should have a
    reasonably comparable scale.

    This normalization is important: raw autoencoder latents are
    not guaranteed to have mean zero and standard deviation one.
    """

    mean = latents.mean(
        dim=(0, 2, 3),
        keepdim=True,
    )

    std = latents.std(
        dim=(0, 2, 3),
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-5)

    return mean, std


def normalize_latents(latents, mean, std):
    return (latents - mean) / std


def denormalize_latents(latents, mean, std):
    return latents * std + mean


# =============================================================================
# 8. TEXT EMBEDDINGS
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

        return (
            self.token_embedding(token_ids)
            + self.position_embedding[:, :sequence_length, :]
        )


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
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=TEXT_NUM_LAYERS,
            enable_nested_tensor=False,
        )

        self.final_norm = nn.LayerNorm(TEXT_EMBED_DIM)

    def forward(self, token_ids, attention_mask):
        x = self.embedding(token_ids)

        padding_mask = ~attention_mask.bool()

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.final_norm(x)

        return x * attention_mask.unsqueeze(-1).to(x.dtype)


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
            [torch.sin(angles), torch.cos(angles)],
            dim=1,
        )

        if embedding.shape[1] < self.embedding_dim:
            embedding = F.pad(
                embedding,
                (0, self.embedding_dim - embedding.shape[1]),
            )

        return self.projection(embedding)


# =============================================================================
# 11. CONDITIONAL RESIDUAL BLOCK
# =============================================================================

class ConditionalResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, condition_dim):
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

        bias = self.condition_projection(
            condition
        ).unsqueeze(-1).unsqueeze(-1)

        h = h + bias

        h = self.conv2(
            F.silu(self.norm2(h))
        )

        return h + residual


# =============================================================================
# 12. TEXT CROSS-ATTENTION
# =============================================================================

class TextCrossAttention(nn.Module):

    def __init__(self, image_channels, text_dim, num_heads):
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
# 13. TWO-RESOLUTION TEXT-CONDITIONED U-NET
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

        # Attention at the 4x4 bottleneck.
        self.middle_attention = TextCrossAttention(
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

        # A second attention layer at 8x8.
        self.output_attention = TextCrossAttention(
            BASE_CHANNELS,
            TEXT_EMBED_DIM,
            CROSS_ATTENTION_HEADS,
        )

        self.output_block = ConditionalResidualBlock(
            BASE_CHANNELS,
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
            h, middle_weights = self.middle_attention(
                h,
                text_features,
                attention_mask,
                return_attention=True,
            )
        else:
            h = self.middle_attention(
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

        if return_attention:
            h, output_weights = self.output_attention(
                h,
                text_features,
                attention_mask,
                return_attention=True,
            )
        else:
            h = self.output_attention(
                h,
                text_features,
                attention_mask,
            )

        h = self.output_block(
            h,
            time_features,
        )

        h = F.silu(self.output_norm(h))
        output = self.output_conv(h)

        if return_attention:
            return output, {
                "middle": middle_weights,
                "output": output_weights,
            }

        return output


# =============================================================================
# 14. DIFFUSION SCHEDULE
# =============================================================================

def make_diffusion_schedule(device):
    betas = torch.linspace(
        BETA_START,
        BETA_END,
        DIFFUSION_STEPS,
        device=device,
    )

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

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
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "posterior_variance": posterior_variance,
    }


def extract(values, timesteps, target_shape):
    if values.device != timesteps.device:
        raise RuntimeError(
            "Schedule and timestep devices do not match"
        )

    result = values.gather(0, timesteps)

    return result.reshape(
        timesteps.shape[0],
        *([1] * (len(target_shape) - 1)),
    )


# =============================================================================
# 15. FORWARD DIFFUSION
# =============================================================================

def q_sample(clean_latents, timesteps, noise, schedule):
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
# 16. DATASET ALIGNMENT
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
# 17. CLASSIFIER-FREE CONDITIONING
# =============================================================================

def apply_classifier_free_dropout(
    token_ids,
    attention_masks,
    probability=CLASSIFIER_FREE_DROPOUT,
):
    """
    Replace selected prompts with a valid UNK-only null prompt.

    The null prompt retains one valid attention position, avoiding
    an all-masked Transformer attention sequence.
    """

    token_ids = token_ids.clone()
    attention_masks = attention_masks.clone()

    batch_size = token_ids.shape[0]

    drop = (
        torch.rand(batch_size, device=token_ids.device)
        < probability
    )

    null_ids = NULL_TOKEN_IDS.to(token_ids.device)
    null_mask = NULL_ATTENTION_MASK.to(attention_masks.device)

    token_ids[drop] = null_ids
    attention_masks[drop] = null_mask

    return token_ids, attention_masks


# =============================================================================
# 18. VALIDATION LOSS
# =============================================================================

@torch.no_grad()
def calculate_validation_loss(
    diffusion_model,
    text_encoder,
    latent_dataset,
    token_ids,
    attention_masks,
    schedule,
):
    device = model_device(diffusion_model)

    diffusion_model.eval()
    text_encoder.eval()

    latent_dataset = latent_dataset.to(device)
    token_ids = token_ids.to(device)
    attention_masks = attention_masks.to(device)

    count = min(BATCH_SIZE, latent_dataset.shape[0])

    indices = torch.randperm(
        latent_dataset.shape[0],
        device=device,
    )[:count]

    clean = latent_dataset[indices]
    tokens = token_ids[indices]
    masks = attention_masks[indices]

    timesteps = torch.randint(
        0,
        DIFFUSION_STEPS,
        (count,),
        device=device,
        dtype=torch.long,
    )

    noise = torch.randn_like(clean)

    noisy = q_sample(
        clean,
        timesteps,
        noise,
        schedule,
    )

    text_features = text_encoder(tokens, masks)

    prediction = diffusion_model(
        noisy,
        timesteps,
        text_features,
        masks,
    )

    return F.mse_loss(prediction, noise).item()


# =============================================================================
# 19. TRAIN TEXT-CONDITIONED DIFFUSION
# =============================================================================

def train_text_conditioned_diffusion(
    diffusion_model,
    text_encoder,
    train_latents,
    train_tokens_cpu,
    train_masks_cpu,
    validation_latents,
    validation_tokens_cpu,
    validation_masks_cpu,
):
    device = model_device(diffusion_model)

    if model_device(text_encoder) != device:
        raise RuntimeError(
            "Text encoder and diffusion model must share a device"
        )

    validate_aligned_dataset(
        train_latents,
        train_tokens_cpu,
        train_masks_cpu,
    )

    validate_aligned_dataset(
        validation_latents,
        validation_tokens_cpu,
        validation_masks_cpu,
    )

    train_latents = train_latents.to(device)
    train_tokens = train_tokens_cpu.to(device)
    train_masks = train_masks_cpu.to(device)

    validation_latents = validation_latents.to(device)
    validation_tokens = validation_tokens_cpu.to(device)
    validation_masks = validation_masks_cpu.to(device)

    schedule = make_diffusion_schedule(device)

    parameters = (
        list(diffusion_model.parameters())
        + list(text_encoder.parameters())
    )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    diffusion_model.train()
    text_encoder.train()

    losses = []
    validation_losses = []

    for step in range(DIFFUSION_TRAIN_STEPS):

        indices = torch.randint(
            0,
            train_latents.shape[0],
            (BATCH_SIZE,),
            device=device,
        )

        clean = train_latents[indices]
        tokens = train_tokens[indices]
        masks = train_masks[indices]

        # Classifier-free training: sometimes remove the text condition.
        tokens, masks = apply_classifier_free_dropout(
            tokens,
            masks,
        )

        timesteps = torch.randint(
            0,
            DIFFUSION_STEPS,
            (BATCH_SIZE,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(clean)

        noisy = q_sample(
            clean,
            timesteps,
            noise,
            schedule,
        )

        text_features = text_encoder(tokens, masks)

        prediction = diffusion_model(
            noisy,
            timesteps,
            text_features,
            masks,
        )

        loss = F.mse_loss(prediction, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            parameters,
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

        if (step + 1) % VALIDATION_INTERVAL == 0:
            validation_loss = calculate_validation_loss(
                diffusion_model,
                text_encoder,
                validation_latents,
                validation_tokens,
                validation_masks,
                schedule,
            )

            validation_losses.append(
                (step + 1, validation_loss)
            )

            print(
                f"  Validation loss={validation_loss:.6f}"
            )

            diffusion_model.train()
            text_encoder.train()

    return losses, validation_losses


# =============================================================================
# 20. DDPM REVERSE STEP
# =============================================================================

@torch.no_grad()
def ddpm_reverse_step(
    diffusion_model,
    latents,
    timesteps,
    text_features,
    attention_mask,
    schedule,
):
    prediction = diffusion_model(
        latents,
        timesteps,
        text_features,
        attention_mask,
    )

    beta = extract(
        schedule["betas"],
        timesteps,
        latents.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule["sqrt_one_minus_alpha_bars"],
        timesteps,
        latents.shape,
    )

    sqrt_recip_alpha = extract(
        schedule["sqrt_recip_alphas"],
        timesteps,
        latents.shape,
    )

    mean = sqrt_recip_alpha * (
        latents
        - beta * prediction / sqrt_one_minus_alpha_bar
    )

    variance = extract(
        schedule["posterior_variance"],
        timesteps,
        latents.shape,
    )

    noise = torch.randn_like(latents)

    nonzero_mask = (
        timesteps > 0
    ).float().reshape(
        -1,
        1,
        1,
        1,
    )

    return (
        mean
        + nonzero_mask
        * torch.sqrt(torch.clamp(variance, min=0.0))
        * noise
    )


# =============================================================================
# 21. CLASSIFIER-FREE GUIDANCE
# =============================================================================

@torch.no_grad()
def guided_noise_prediction(
    diffusion_model,
    latents,
    timesteps,
    conditional_features,
    conditional_mask,
    unconditional_features,
    unconditional_mask,
    guidance_scale,
):
    conditional_prediction = diffusion_model(
        latents,
        timesteps,
        conditional_features,
        conditional_mask,
    )

    unconditional_prediction = diffusion_model(
        latents,
        timesteps,
        unconditional_features,
        unconditional_mask,
    )

    return (
        unconditional_prediction
        + guidance_scale
        * (
            conditional_prediction
            - unconditional_prediction
        )
    )


# =============================================================================
# 22. COMPLETE TEXT-CONDITIONED SAMPLER
# =============================================================================

@torch.no_grad()
def sample_text_conditioned(
    diffusion_model,
    text_encoder,
    token_ids,
    attention_mask,
    schedule,
    num_samples,
    initial_noise=None,
    guidance_scale=GUIDANCE_SCALE,
):
    device = model_device(diffusion_model)

    if model_device(text_encoder) != device:
        raise RuntimeError("Text encoder and diffusion model device mismatch")

    token_ids = token_ids.to(device, dtype=torch.long)
    attention_mask = attention_mask.to(device, dtype=torch.bool)

    if token_ids.shape[0] == 1 and num_samples > 1:
        token_ids = token_ids.expand(num_samples, -1)
        attention_mask = attention_mask.expand(num_samples, -1)

    if token_ids.shape[0] != num_samples:
        raise ValueError(
            "Prompt batch size must equal num_samples"
        )

    schedule = {
        key: value.to(device)
        for key, value in schedule.items()
    }

    diffusion_model.eval()
    text_encoder.eval()

    conditional_features = text_encoder(
        token_ids,
        attention_mask,
    )

    null_ids = NULL_TOKEN_IDS.to(device).unsqueeze(0)
    null_mask = NULL_ATTENTION_MASK.to(device).unsqueeze(0)

    null_ids = null_ids.expand(num_samples, -1)
    null_mask = null_mask.expand(num_samples, -1)

    unconditional_features = text_encoder(
        null_ids,
        null_mask,
    )

    if initial_noise is None:
        latents = torch.randn(
            num_samples,
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
            device=device,
        )
    else:
        latents = initial_noise.to(device).clone()

        if latents.shape != (
            num_samples,
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
        ):
            raise ValueError("Initial noise has incorrect shape")

    for timestep in reversed(range(DIFFUSION_STEPS)):

        t = torch.full(
            (num_samples,),
            timestep,
            device=device,
            dtype=torch.long,
        )

        guided_prediction = guided_noise_prediction(
            diffusion_model,
            latents,
            t,
            conditional_features,
            attention_mask,
            unconditional_features,
            null_mask,
            guidance_scale,
        )

        beta = extract(
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
            - beta * guided_prediction
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
# 23. INDEPENDENT SHAPE CLASSIFIER
# =============================================================================

class ShapeClassifier(nn.Module):
    """
    This model is trained from scratch and is NOT used to generate images.

    Its purpose is independent evaluation:
    Does a generated image resemble the requested shape category?
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        features = self.features(x).flatten(1)
        return self.classifier(features)


def train_shape_classifier(model, images, labels):
    device = model_device(model)

    images = images.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    model.train()
    losses = []

    for step in range(CLASSIFIER_TRAIN_STEPS):
        indices = torch.randint(
            0,
            images.shape[0],
            (BATCH_SIZE,),
            device=device,
        )

        batch_images = images[indices]
        batch_labels = labels[indices]

        logits = model(batch_images)
        loss = F.cross_entropy(logits, batch_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(
                f"Classifier step {step + 1}/"
                f"{CLASSIFIER_TRAIN_STEPS} "
                f"loss={loss.item():.6f}"
            )

    return losses


@torch.no_grad()
def evaluate_classifier(model, images, labels):
    device = model_device(model)

    model.eval()

    logits = model(images.to(device))
    predictions = logits.argmax(dim=1)

    labels = labels.to(device)

    accuracy = (
        predictions == labels
    ).float().mean().item()

    return accuracy, predictions, logits.softmax(dim=1)


# =============================================================================
# 24. GENERATE AND EVALUATE PROMPTS
# =============================================================================

@torch.no_grad()
def generate_and_evaluate(
    autoencoder,
    diffusion_model,
    text_encoder,
    classifier,
    latent_mean,
    latent_std,
):
    print("\nGENERATING IMAGES FROM TEXT")

    device = model_device(diffusion_model)

    token_ids, masks = TOKENIZER.batch_encode(PROMPTS)

    token_ids = token_ids.to(device)
    masks = masks.to(device)

    schedule = make_diffusion_schedule(device)

    generated_normalized_latents = sample_text_conditioned(
        diffusion_model,
        text_encoder,
        token_ids,
        masks,
        schedule,
        num_samples=NUM_CLASSES,
        guidance_scale=GUIDANCE_SCALE,
    )

    generated_raw_latents = denormalize_latents(
        generated_normalized_latents,
        latent_mean,
        latent_std,
    )

    generated_images = autoencoder.decode(
        generated_raw_latents
    ).clamp(0.0, 1.0)

    expected_labels = torch.arange(
        NUM_CLASSES,
        device=device,
    )

    accuracy, predictions, probabilities = evaluate_classifier(
        classifier,
        generated_images,
        expected_labels,
    )

    print("\nSEMANTIC EVALUATION")

    for i, prompt in enumerate(PROMPTS):
        predicted_class = predictions[i].item()
        confidence = probabilities[i, predicted_class].item()

        print(
            f"Prompt: {prompt:20s} "
            f"Predicted: {PROMPTS[predicted_class]:20s} "
            f"Confidence: {confidence:.3f}"
        )

    print(
        f"Generated shape-class accuracy: {accuracy * 100:.1f}%"
    )

    titles = []

    for i, prompt in enumerate(PROMPTS):
        predicted = PROMPTS[predictions[i].item()]
        titles.append(
            f"Prompt: {prompt}\nPredicted: {predicted}"
        )

    show_image_grid(
        generated_images,
        titles,
        "Generated images and independent classifier predictions",
    )

    return generated_images, accuracy


# =============================================================================
# 25. SAME NOISE / DIFFERENT TEXT EXPERIMENT
# =============================================================================

@torch.no_grad()
def experiment_same_noise_different_text(
    autoencoder,
    diffusion_model,
    text_encoder,
    classifier,
    latent_mean,
    latent_std,
):
    print("\nEXPERIMENT — SAME NOISE / DIFFERENT TEXT")

    device = model_device(diffusion_model)

    ids, masks = TOKENIZER.batch_encode(PROMPTS)

    ids = ids.to(device)
    masks = masks.to(device)

    initial_noise = torch.randn(
        NUM_CLASSES,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=device,
    )

    schedule = make_diffusion_schedule(device)

    normalized_latents = sample_text_conditioned(
        diffusion_model,
        text_encoder,
        ids,
        masks,
        schedule,
        num_samples=NUM_CLASSES,
        initial_noise=initial_noise,
        guidance_scale=GUIDANCE_SCALE,
    )

    raw_latents = denormalize_latents(
        normalized_latents,
        latent_mean,
        latent_std,
    )

    images = autoencoder.decode(
        raw_latents
    ).clamp(0.0, 1.0)

    accuracy, predictions, probabilities = evaluate_classifier(
        classifier,
        images,
        torch.arange(NUM_CLASSES, device=device),
    )

    print("Prompts:", PROMPTS)

    pairwise_differences = []

    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            pairwise_differences.append(
                torch.mean(
                    torch.abs(images[i] - images[j])
                ).item()
            )

    print("Pairwise image differences:", pairwise_differences)

    for i in range(NUM_CLASSES):
        print(
            f"{PROMPTS[i]} -> "
            f"{PROMPTS[predictions[i].item()]}, "
            f"confidence={probabilities[i, predictions[i]].item():.3f}"
        )

    print(
        f"Same-noise semantic accuracy: {accuracy * 100:.1f}%"
    )

    show_image_grid(
        images,
        PROMPTS,
        "Same initial noise — different text prompts",
    )

    return images


# =============================================================================
# 26. UNIT TEST 1 — VOCABULARY
# =============================================================================

def test_vocabulary():
    print("\nTEST 1 — VOCABULARY")

    assert VOCAB[PAD_TOKEN] == 0
    assert VOCAB[UNK_TOKEN] == 1
    assert len(VOCAB) == VOCAB_SIZE
    assert len(set(VOCAB.values())) == VOCAB_SIZE

    print("Vocabulary size:", VOCAB_SIZE)
    print("PASS")


# =============================================================================
# 27. UNIT TEST 2 — TOKENIZER
# =============================================================================

def test_tokenizer():
    print("\nTEST 2 — TOKENIZER")

    ids, mask = TOKENIZER.encode("make a circle")

    assert ids.tolist() == [
        VOCAB["make"],
        VOCAB["a"],
        VOCAB["circle"],
        VOCAB[PAD_TOKEN],
    ]

    assert mask.tolist() == [
        True, True, True, False,
    ]

    print("Token IDs:", ids.tolist())
    print("Mask:", mask.tolist())
    print("PASS")


# =============================================================================
# 28. UNIT TEST 3 — UNKNOWN TOKENS AND TRUNCATION
# =============================================================================

def test_unknown_and_truncation():
    print("\nTEST 3 — UNKNOWN TOKENS AND TRUNCATION")

    ids, mask = TOKENIZER.encode("unknown object")

    assert ids[0].item() == TOKENIZER.unk_id
    assert ids[1].item() == TOKENIZER.unk_id

    ids, mask = TOKENIZER.encode(
        "make a circle square diagonal"
    )

    assert ids.shape[0] == MAX_TEXT_LENGTH
    assert mask.all()

    empty_ids, empty_mask = TOKENIZER.encode("")

    assert empty_mask[0]
    assert empty_ids[0].item() == TOKENIZER.unk_id

    print("PASS")


# =============================================================================
# 29. UNIT TEST 4 — BATCH TOKENIZATION
# =============================================================================

def test_batch_tokenization():
    print("\nTEST 4 — BATCH TOKENIZATION")

    ids, masks = TOKENIZER.batch_encode(PROMPTS)

    assert ids.shape == (4, MAX_TEXT_LENGTH)
    assert masks.shape == ids.shape
    assert ids.dtype == torch.long
    assert masks.dtype == torch.bool

    print("PASS")


# =============================================================================
# 30. UNIT TEST 5 — TEXT EMBEDDING
# =============================================================================

def test_text_embedding():
    print("\nTEST 5 — TEXT EMBEDDING")

    embedding = TextEmbedding(
        VOCAB_SIZE,
        TEXT_EMBED_DIM,
        MAX_TEXT_LENGTH,
        TOKENIZER.pad_id,
    )

    ids, _ = TOKENIZER.batch_encode(PROMPTS)
    output = embedding(ids)

    assert output.shape == (
        4, MAX_TEXT_LENGTH, TEXT_EMBED_DIM
    )

    assert torch.isfinite(output).all()

    print("Output:", tuple(output.shape))
    print("PASS")


# =============================================================================
# 31. UNIT TEST 6 — TEXT ENCODER
# =============================================================================

def test_text_encoder():
    print("\nTEST 6 — TEXT ENCODER")

    model = TextEncoder().to(DEVICE)

    ids, masks = TOKENIZER.batch_encode(PROMPTS)

    output = model(
        ids.to(DEVICE),
        masks.to(DEVICE),
    )

    assert output.shape == (
        4, MAX_TEXT_LENGTH, TEXT_EMBED_DIM
    )

    assert output.device == model_device(model)
    assert torch.isfinite(output).all()

    print("Output:", tuple(output.shape))
    print("PASS")


# =============================================================================
# 32. UNIT TEST 7 — PADDING OUTPUT
# =============================================================================

def test_padding_output():
    print("\nTEST 7 — PADDING OUTPUT")

    model = TextEncoder().to(DEVICE).eval()

    ids, masks = TOKENIZER.encode("make a circle")

    ids = ids.unsqueeze(0).to(DEVICE)
    masks = masks.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(ids, masks)

    padded = output[~masks]

    assert torch.allclose(
        padded,
        torch.zeros_like(padded),
        atol=1e-6,
    )

    print("PASS")


# =============================================================================
# 33. UNIT TEST 8 — PADDING MASK INVARIANCE
# =============================================================================

def test_padding_mask_invariance():
    print("\nTEST 8 — PADDING MASK INVARIANCE")

    model = TextEncoder().to(DEVICE).eval()

    ids_a, masks_a = TOKENIZER.encode("make a circle")
    ids_b, masks_b = TOKENIZER.encode("make a circle")

    ids_b[-1] = VOCAB["square"]

    ids_a = ids_a.unsqueeze(0).to(DEVICE)
    ids_b = ids_b.unsqueeze(0).to(DEVICE)

    masks_a = masks_a.unsqueeze(0).to(DEVICE)
    masks_b = masks_b.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_a = model(ids_a, masks_a)
        out_b = model(ids_b, masks_b)

    valid_a = out_a[:, masks_a[0], :]
    valid_b = out_b[:, masks_b[0], :]

    assert torch.allclose(
        valid_a,
        valid_b,
        atol=1e-5,
        rtol=1e-5,
    )

    print("PASS")


# =============================================================================
# 34. UNIT TEST 9 — CROSS-ATTENTION
# =============================================================================

def test_cross_attention():
    print("\nTEST 9 — CROSS-ATTENTION OUTPUT")

    attention = TextCrossAttention(
        64,
        TEXT_EMBED_DIM,
        4,
    ).to(DEVICE)

    image = torch.randn(2, 64, 4, 4, device=DEVICE)

    text = torch.randn(
        2, MAX_TEXT_LENGTH, TEXT_EMBED_DIM,
        device=DEVICE,
    )

    masks = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
        ],
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

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()

    print("Output:", tuple(output.shape))
    print("Weights:", tuple(weights.shape))
    print("PASS")


# =============================================================================
# 35. UNIT TEST 10 — CROSS-ATTENTION PADDING MASK
# =============================================================================

def test_cross_attention_padding_mask():
    print("\nTEST 10 — CROSS-ATTENTION PADDING MASK")

    attention = TextCrossAttention(
        64,
        TEXT_EMBED_DIM,
        4,
    ).to(DEVICE).eval()

    image = torch.randn(1, 64, 4, 4, device=DEVICE)

    text_a = torch.randn(
        1, MAX_TEXT_LENGTH, TEXT_EMBED_DIM,
        device=DEVICE,
    )

    text_b = text_a.clone()
    text_b[:, -1, :] = 1000.0

    mask = torch.tensor(
        [[True, True, True, False]],
        device=DEVICE,
    )

    with torch.no_grad():
        output_a = attention(image, text_a, mask)
        output_b = attention(image, text_b, mask)

    assert torch.allclose(
        output_a,
        output_b,
        atol=1e-5,
        rtol=1e-5,
    )

    print("PASS")


# =============================================================================
# 36. UNIT TEST 11 — DIFFUSION SCHEDULE
# =============================================================================

def test_diffusion_schedule():
    print("\nTEST 11 — DIFFUSION SCHEDULE")

    schedule = make_diffusion_schedule(DEVICE)

    assert schedule["betas"].shape == (DIFFUSION_STEPS,)
    assert torch.all(schedule["betas"] > 0)
    assert torch.all(schedule["betas"] < 1)

    assert torch.all(
        schedule["alpha_bars"][1:]
        <= schedule["alpha_bars"][:-1]
    )

    print("PASS")


# =============================================================================
# 37. UNIT TEST 12 — FORWARD DIFFUSION
# =============================================================================

def test_forward_diffusion():
    print("\nTEST 12 — FORWARD DIFFUSION")

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
    assert noisy.device == clean.device
    assert torch.isfinite(noisy).all()

    print("PASS")


# =============================================================================
# 38. UNIT TEST 13 — LATENT NORMALIZATION
# =============================================================================

def test_latent_normalization():
    print("\nTEST 13 — LATENT NORMALIZATION")

    raw = (
        torch.randn(
            128,
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
        ) * 2.5 + 4.0
    )

    mean, std = calculate_latent_statistics(raw)

    normalized = normalize_latents(raw, mean, std)
    reconstructed = denormalize_latents(
        normalized,
        mean,
        std,
    )

    assert torch.allclose(
        reconstructed,
        raw,
        atol=1e-5,
    )

    channel_means = normalized.mean(dim=(0, 2, 3))
    channel_stds = normalized.std(
        dim=(0, 2, 3),
        unbiased=False,
    )

    assert torch.allclose(
        channel_means,
        torch.zeros_like(channel_means),
        atol=1e-5,
    )

    assert torch.allclose(
        channel_stds,
        torch.ones_like(channel_stds),
        atol=1e-4,
    )

    print("Normalized means:", channel_means.tolist())
    print("Normalized stds:", channel_stds.tolist())
    print("PASS")


# =============================================================================
# 39. UNIT TEST 14 — U-NET OUTPUT AND ATTENTION RESOLUTIONS
# =============================================================================

def test_unet_output():
    print("\nTEST 14 — U-NET OUTPUT AND ATTENTION RESOLUTIONS")

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

    output, attention = model(
        latents,
        timesteps,
        text_features,
        masks,
        return_attention=True,
    )

    assert output.shape == latents.shape
    assert attention["middle"].shape == (
        2, CROSS_ATTENTION_HEADS, 16, MAX_TEXT_LENGTH
    )
    assert attention["output"].shape == (
        2, CROSS_ATTENTION_HEADS, 64, MAX_TEXT_LENGTH
    )

    print("Output:", tuple(output.shape))
    print("Middle attention:", tuple(attention["middle"].shape))
    print("Output attention:", tuple(attention["output"].shape))
    print("PASS")


# =============================================================================
# 40. UNIT TEST 15 — GRADIENT PROPAGATION
# =============================================================================

def test_gradient_propagation():
    print("\nTEST 15 — GRADIENT PROPAGATION")

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

    features = text_encoder(ids, masks)

    output = diffusion_model(
        latents,
        timesteps,
        features,
        masks,
    )

    output.square().mean().backward()

    text_grads = [
        p.grad for p in text_encoder.parameters()
        if p.grad is not None
    ]

    diffusion_grads = [
        p.grad for p in diffusion_model.parameters()
        if p.grad is not None
    ]

    assert len(text_grads) > 0
    assert len(diffusion_grads) > 0

    assert any(
        g.abs().sum().item() > 0
        for g in text_grads
    )

    assert any(
        g.abs().sum().item() > 0
        for g in diffusion_grads
    )

    print("Text encoder gradient tensors:", len(text_grads))
    print("Diffusion gradient tensors:", len(diffusion_grads))
    print("PASS")


# =============================================================================
# 41. UNIT TEST 16 — CLASSIFIER-FREE DROPOUT
# =============================================================================

def test_classifier_free_dropout():
    print("\nTEST 16 — CLASSIFIER-FREE DROPOUT")

    ids, masks = TOKENIZER.batch_encode(PROMPTS)
    ids = ids.to(DEVICE)
    masks = masks.to(DEVICE)

    dropped_ids, dropped_masks = apply_classifier_free_dropout(
        ids,
        masks,
        probability=1.0,
    )

    expected_ids = NULL_TOKEN_IDS.to(DEVICE).expand_as(dropped_ids)
    expected_masks = NULL_ATTENTION_MASK.to(DEVICE).expand_as(
        dropped_masks
    )

    assert torch.equal(dropped_ids, expected_ids)
    assert torch.equal(dropped_masks, expected_masks)

    assert dropped_masks[:, 0].all()
    assert not dropped_masks[:, 1:].any()

    print("PASS")


# =============================================================================
# 42. UNIT TEST 17 — GUIDANCE MATHEMATICS
# =============================================================================

def test_guidance_math():
    print("\nTEST 17 — GUIDANCE MATHEMATICS")

    unconditional = torch.ones(2, 4, 8, 8, device=DEVICE)
    conditional = torch.full_like(unconditional, 3.0)

    scale = 2.0

    guided = (
        unconditional
        + scale * (conditional - unconditional)
    )

    assert torch.allclose(
        guided,
        torch.full_like(guided, 5.0),
    )

    # Guidance scale 0 means unconditional prediction.
    guided_zero = (
        unconditional
        + 0.0 * (conditional - unconditional)
    )

    assert torch.equal(guided_zero, unconditional)

    print("PASS")


# =============================================================================
# 43. UNIT TEST 18 — DEVICE CONSISTENCY
# =============================================================================

def test_device_consistency():
    print("\nTEST 18 — DEVICE CONSISTENCY")

    text_encoder = TextEncoder().to(DEVICE)
    diffusion_model = TextConditionedUNet().to(DEVICE)

    device = model_device(diffusion_model)

    ids, masks = TOKENIZER.batch_encode(PROMPTS[:2])

    ids = ids.to(device)
    masks = masks.to(device)

    features = text_encoder(ids, masks)

    latents = torch.randn(
        2, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
        device=device,
    )

    timesteps = torch.zeros(
        2,
        device=device,
        dtype=torch.long,
    )

    output = diffusion_model(
        latents,
        timesteps,
        features,
        masks,
    )

    assert model_device(text_encoder) == device
    assert output.device == device

    print("Configured device:", DEVICE)
    print("Actual model device:", device)
    print("PASS")


# =============================================================================
# 44. UNIT TEST 19 — CPU TOKENS / MODEL-DEVICE INDEXING
# =============================================================================

def test_cpu_tokens_model_device_indexing():
    print("\nTEST 19 — CPU TOKENS / MODEL-DEVICE INDEXING")

    prompts = PROMPTS * 2

    tokens_cpu, masks_cpu = TOKENIZER.batch_encode(prompts)

    latent_dataset = torch.randn(
        len(prompts),
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=DEVICE,
    )

    validate_aligned_dataset(
        latent_dataset,
        tokens_cpu,
        masks_cpu,
    )

    device = latent_dataset.device

    tokens_device = tokens_cpu.to(device)
    masks_device = masks_cpu.to(device)

    indices = torch.tensor(
        [7, 2, 5, 0],
        device=device,
        dtype=torch.long,
    )

    batch_latents = latent_dataset[indices]
    batch_tokens = tokens_device[indices]
    batch_masks = masks_device[indices]

    assert batch_latents.device == device
    assert batch_tokens.device == device
    assert batch_masks.device == device

    assert torch.equal(
        batch_tokens.cpu(),
        tokens_cpu[indices.cpu()],
    )

    assert torch.equal(
        batch_masks.cpu(),
        masks_cpu[indices.cpu()],
    )

    print("Number of latent rows:", latent_dataset.shape[0])
    print("Number of text rows:", tokens_cpu.shape[0])
    print("Selected rows:", len(indices))
    print("PASS")


# =============================================================================
# 45. UNIT TEST 20 — DATASET ALIGNMENT
# =============================================================================

def test_dataset_alignment():
    print("\nTEST 20 — DATASET ALIGNMENT")

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

    print("Matching dataset accepted.")
    print("Mismatched dataset rejected.")
    print("PASS")


# =============================================================================
# 46. UNIT TEST 21 — COMPLETE SAMPLER
# =============================================================================

def test_complete_sampler():
    print("\nTEST 21 — COMPLETE SAMPLER")

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
        guidance_scale=GUIDANCE_SCALE,
    )

    assert generated.shape == (
        2,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    )

    assert generated.device == model_device(diffusion_model)
    assert torch.isfinite(generated).all()

    print("Generated shape:", tuple(generated.shape))
    print("PASS")


# =============================================================================
# 47. MAIN
# =============================================================================

def main():
    set_seed()

    print("=" * 80)
    print("LESSON 26 — TEXT CONDITIONING (IMPROVED)")
    print("=" * 80)

    print("PyTorch:", torch.__version__)
    print("Configured device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Vocabulary size:", VOCAB_SIZE)
    print("Text embedding dimension:", TEXT_EMBED_DIM)
    print("Diffusion steps:", DIFFUSION_STEPS)
    print("Diffusion training steps:", DIFFUSION_TRAIN_STEPS)
    print("Classifier-free dropout:", CLASSIFIER_FREE_DROPOUT)
    print("Guidance scale:", GUIDANCE_SCALE)

    # -------------------------------------------------------------------------
    # UNIT TESTS
    # -------------------------------------------------------------------------

    tests = [
        test_vocabulary,
        test_tokenizer,
        test_unknown_and_truncation,
        test_batch_tokenization,
        test_text_embedding,
        test_text_encoder,
        test_padding_output,
        test_padding_mask_invariance,
        test_cross_attention,
        test_cross_attention_padding_mask,
        test_diffusion_schedule,
        test_forward_diffusion,
        test_latent_normalization,
        test_unet_output,
        test_gradient_propagation,
        test_classifier_free_dropout,
        test_guidance_math,
        test_device_consistency,
        test_cpu_tokens_model_device_indexing,
        test_dataset_alignment,
        test_complete_sampler,
    ]

    print("\nRUNNING UNIT TESTS")

    for test in tests:
        test()

    print("\nALL UNIT TESTS PASSED")

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------

    print("\nCREATING SYNTHETIC DATASET")

    images, labels, prompts = create_synthetic_dataset(
        samples_per_class=SAMPLES_PER_CLASS
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
        "Synthetic shape training dataset",
    )

    # -------------------------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # -------------------------------------------------------------------------

    print("\nCREATING TRAIN / VALIDATION SPLIT")

    permutation = torch.randperm(len(images))

    validation_count = int(0.10 * len(images))

    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]

    train_images = images[training_indices]
    train_labels = labels[training_indices]

    validation_images = images[validation_indices]
    validation_labels = labels[validation_indices]

    train_prompts = [
        prompts[i] for i in training_indices.tolist()
    ]

    validation_prompts = [
        prompts[i] for i in validation_indices.tolist()
    ]

    print("Training images:", len(train_images))
    print("Validation images:", len(validation_images))

    # -------------------------------------------------------------------------
    # TOKENIZATION
    # -------------------------------------------------------------------------

    print("\nTOKENIZING TRAINING AND VALIDATION PROMPTS")

    train_tokens, train_masks = TOKENIZER.batch_encode(
        train_prompts
    )

    validation_tokens, validation_masks = TOKENIZER.batch_encode(
        validation_prompts
    )

    print("Training token IDs:", tuple(train_tokens.shape))
    print("Validation token IDs:", tuple(validation_tokens.shape))

    # -------------------------------------------------------------------------
    # AUTOENCODER
    # -------------------------------------------------------------------------

    print("\nTRAINING AUTOENCODER")

    autoencoder = ConvAutoencoder().to(DEVICE)

    start_time = time.time()

    ae_losses = train_autoencoder(
        autoencoder,
        train_images,
    )

    print(
        "AE training time:",
        f"{time.time() - start_time:.2f} seconds",
    )

    plot_losses(
        ae_losses,
        title="Autoencoder training loss",
    )

    # -------------------------------------------------------------------------
    # LATENT ENCODING
    # -------------------------------------------------------------------------

    print("\nENCODING DATASET INTO LATENT SPACE")

    autoencoder.eval()

    with torch.no_grad():
        train_latents = autoencoder.encode(
            train_images.to(DEVICE)
        )

        validation_latents = autoencoder.encode(
            validation_images.to(DEVICE)
        )

    print("Training latents:", tuple(train_latents.shape))
    print("Validation latents:", tuple(validation_latents.shape))

    # -------------------------------------------------------------------------
    # LATENT NORMALIZATION
    # -------------------------------------------------------------------------

    print("\nCALCULATING LATENT NORMALIZATION")

    latent_mean, latent_std = calculate_latent_statistics(
        train_latents
    )

    train_latents = normalize_latents(
        train_latents,
        latent_mean,
        latent_std,
    )

    validation_latents = normalize_latents(
        validation_latents,
        latent_mean,
        latent_std,
    )

    print("Latent channel means:", latent_mean.flatten().tolist())
    print("Latent channel stds:", latent_std.flatten().tolist())

    normalized_mean = train_latents.mean(
        dim=(0, 2, 3)
    )

    normalized_std = train_latents.std(
        dim=(0, 2, 3),
        unbiased=False,
    )

    print("Normalized latent means:", normalized_mean.tolist())
    print("Normalized latent stds:", normalized_std.tolist())

    # -------------------------------------------------------------------------
    # AUTOENCODER RECONSTRUCTION
    # -------------------------------------------------------------------------

    print("\nAUTOENCODER RECONSTRUCTION")

    with torch.no_grad():
        reconstruction = autoencoder(
            validation_images[:8].to(DEVICE)
        )

    reconstruction_error = F.mse_loss(
        reconstruction,
        validation_images[:8].to(DEVICE),
    ).item()

    print("Reconstruction MSE:", reconstruction_error)

    show_image_grid(
        validation_images[:8],
        ["Original"] * 8,
        "Original validation images",
    )

    show_image_grid(
        reconstruction,
        ["Reconstruction"] * 8,
        "Autoencoder reconstruction",
    )

    # -------------------------------------------------------------------------
    # TEXT ENCODER
    # -------------------------------------------------------------------------

    print("\nBUILDING TEXT ENCODER")

    text_encoder = TextEncoder().to(DEVICE)

    print(
        "Text encoder parameters:",
        sum(p.numel() for p in text_encoder.parameters()),
    )

    # -------------------------------------------------------------------------
    # DIFFUSION U-NET
    # -------------------------------------------------------------------------

    print("\nBUILDING TEXT-CONDITIONED DIFFUSION U-NET")

    diffusion_model = TextConditionedUNet().to(DEVICE)

    print(
        "Diffusion parameters:",
        sum(p.numel() for p in diffusion_model.parameters()),
    )

    # -------------------------------------------------------------------------
    # TRAIN DIFFUSION
    # -------------------------------------------------------------------------

    print("\nTRAINING TEXT-CONDITIONED DIFFUSION")

    start_time = time.time()

    diffusion_losses, validation_losses = (
        train_text_conditioned_diffusion(
            diffusion_model,
            text_encoder,
            train_latents,
            train_tokens,
            train_masks,
            validation_latents,
            validation_tokens,
            validation_masks,
        )
    )

    print(
        "Diffusion training time:",
        f"{time.time() - start_time:.2f} seconds",
    )

    plot_losses(
        diffusion_losses,
        validation_losses,
        "Text-conditioned diffusion training and validation loss",
    )

    # -------------------------------------------------------------------------
    # INDEPENDENT CLASSIFIER
    # -------------------------------------------------------------------------

    print("\nTRAINING INDEPENDENT SHAPE CLASSIFIER")

    classifier = ShapeClassifier().to(DEVICE)

    start_time = time.time()

    classifier_losses = train_shape_classifier(
        classifier,
        train_images,
        train_labels,
    )

    print(
        "Classifier training time:",
        f"{time.time() - start_time:.2f} seconds",
    )

    plot_losses(
        classifier_losses,
        title="Independent shape classifier training loss",
    )

    classifier_accuracy, _, _ = evaluate_classifier(
        classifier,
        validation_images,
        validation_labels,
    )

    print(
        "Classifier validation accuracy:",
        f"{classifier_accuracy * 100:.2f}%",
    )

    # A weak evaluator cannot provide meaningful semantic assessment.
    if classifier_accuracy < 0.90:
        print(
            "WARNING: classifier validation accuracy is below 90%. "
            "Generated-image semantic evaluation may be unreliable."
        )

    # -------------------------------------------------------------------------
    # GENERATION
    # -------------------------------------------------------------------------

    generated_images, generated_accuracy = generate_and_evaluate(
        autoencoder,
        diffusion_model,
        text_encoder,
        classifier,
        latent_mean,
        latent_std,
    )

    # -------------------------------------------------------------------------
    # SAME NOISE / DIFFERENT PROMPTS
    # -------------------------------------------------------------------------

    experiment_same_noise_different_text(
        autoencoder,
        diffusion_model,
        text_encoder,
        classifier,
        latent_mean,
        latent_std,
    )

    # -------------------------------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LESSON 26 COMPLETE")
    print("=" * 80)

    print("Tokenizer: custom vocabulary")
    print("Text encoder: random-initialized Transformer")
    print("Conditioning: two-resolution multi-head cross-attention")
    print("Latent diffusion: normalized autoencoder latents")
    print("Conditioning dropout: classifier-free")
    print("Sampling: DDPM with classifier-free guidance")
    print("Evaluation: independent shape classifier")
    print("Pretrained components: NONE")

    print("\nFINAL METRICS")
    print(
        "Classifier validation accuracy:",
        f"{classifier_accuracy * 100:.2f}%",
    )
    print(
        "Generated shape-class accuracy:",
        f"{generated_accuracy * 100:.2f}%",
    )

    print("\nWHAT YOU BUILT:")
    print("1. Tokenizer")
    print("2. Token and positional embeddings")
    print("3. Transformer text encoder")
    print("4. Multi-resolution text cross-attention")
    print("5. Normalized latent diffusion")
    print("6. Classifier-free condition dropout")
    print("7. Classifier-free guidance")
    print("8. DDPM sampler")
    print("9. Independent semantic evaluator")

    print("\nIMPORTANT:")
    print(
        "Classifier accuracy is a useful proxy, not proof of "
        "general natural-language understanding."
    )

    print("\nNEXT LESSON:")
    print("LESSON 27 — BUILD OUR OWN TEXT-TO-IMAGE MODEL")


if __name__ == "__main__":
    main()
