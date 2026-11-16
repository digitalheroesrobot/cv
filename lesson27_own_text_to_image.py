
"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 27 — BUILD OUR OWN TEXT-TO-IMAGE MODEL (IMPROVED)
================================================================================

PURPOSE
-------
Train a small text-conditioned image diffusion model from scratch.

This improved version addresses the weak and inconsistent outputs from the
previous version by:
1. Starting with centered, consistently sized shapes.
2. Reducing unnecessary variation in the training dataset.
3. Injecting text conditioning into every U-Net residual block.
4. Applying cross-attention at multiple image resolutions.
5. Increasing training steps and batch size.
6. Using an independent classifier only as a diagnostic.

WHAT YOU WILL LEARN
-------------------
1. Text tokenization and Transformer encoding.
2. Text-conditioned feature modulation (FiLM).
3. Multi-resolution cross-attention.
4. Image-space DDPM training.
5. Complete DDPM reverse sampling.
6. Independent evaluation of generated images.
7. Diagnosing conditioning and image-quality failures.

TECHNICAL ROADMAP
-----------------
Lesson 23: Latent diffusion.
Lesson 24: Class conditioning.
Lesson 25: Cross-attention.
Lesson 26: Text conditioning.
Lesson 27: Complete text-to-image diffusion.

REAL-WORLD PROBLEM
------------------
A vision system receives a language instruction and must create a
corresponding visual representation.

Initial vocabulary:
    "make a circle"
    "make a square"
    "make a diagonal"
    "make a cross"

THEORY
------
Text:
    token IDs -> embeddings -> Transformer -> contextual token features

Conditioning:
    pooled text features -> residual-block feature modulation
    contextual text features -> cross-attention keys and values

Cross-attention:
    Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V

Forward diffusion:
    x_t = sqrt(alpha_bar_t) x_0
          + sqrt(1-alpha_bar_t) epsilon

Training objective:
    L = E[||epsilon - epsilon_theta(x_t,t,text)||^2]

Reverse diffusion:
    Predict noise and iteratively remove it from Gaussian noise.

IMPLEMENTATION
--------------
- Synthetic 32x32 shape dataset.
- Custom tokenizer.
- Random-initialized Transformer text encoder.
- Text-conditioned residual blocks.
- Multi-resolution cross-attention.
- Image-space diffusion U-Net.
- DDPM noise-prediction training.
- Complete DDPM reverse sampler.
- Independently trained diagnostic classifier.

VERIFICATION
------------
Tests cover tokenizer, padding, text encoder, schedule, forward diffusion,
cross-attention, U-Net shape, gradient flow, device consistency, dataset
alignment, and complete sampling.

REAL-WORLD APPLICATION
----------------------
The architecture demonstrates how language features can control visual
generation. The same general conditioning idea can be used in robotics
for language-conditioned visual representations.

REAL-WORLD RESULT
-----------------
The model should generate recognizable examples of all four trained shapes.

LESSON RESULT
-------------
A complete small text-to-image diffusion model trained from scratch.

CONNECTION TO NEXT LESSON
-------------------------
Next: classifier-free guidance, EMA, improved sampling, and more expressive
text conditioning.

IMPORTANT
---------
This is an educational four-prompt model, not a general-purpose image model.
No pretrained model, pretrained text encoder, or pretrained checkpoint
is used.

The independent classifier is not part of the generator and does not guide
the generation process.
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

PROMPTS = [
    "make a circle",
    "make a square",
    "make a diagonal",
    "make a cross",
]

NUM_CLASSES = len(PROMPTS)

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

TIME_EMBED_DIM = 128
CONDITION_DIM = 256

BASE_CHANNELS = 32
ATTENTION_HEADS = 4

DIFFUSION_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.02

# A smaller, more controlled dataset makes the first generation task easier.
SAMPLES_PER_CLASS = 768

BATCH_SIZE = 128
TRAIN_STEPS = 6000
LEARNING_RATE = 1e-3

CLASSIFIER_EPOCHS = 10
CLASSIFIER_BATCH_SIZE = 128

NUM_GENERATED = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# 3. DEVICE HELPERS
# =============================================================================

def get_model_device(model):

    return next(model.parameters()).device


def check_model_devices(text_encoder, diffusion_model):

    text_device = get_model_device(text_encoder)
    diffusion_device = get_model_device(diffusion_model)

    if text_device != diffusion_device:
        raise RuntimeError(
            f"Text encoder is on {text_device}, "
            f"but diffusion model is on {diffusion_device}"
        )

    return diffusion_device


def move_schedule_to_device(schedule, device):

    return {
        key: value.to(device)
        for key, value in schedule.items()
    }


# =============================================================================
# 4. TOKENIZER
# =============================================================================

class SimpleTokenizer:

    def __init__(self, vocabulary, max_length):

        self.word_to_id = {
            word: index
            for index, word in enumerate(vocabulary)
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

        ids += (
            [self.pad_id]
            * (self.max_length - valid_length)
        )

        mask = (
            [True] * valid_length
            + [False] * (self.max_length - valid_length)
        )

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )

    def batch_encode(self, texts):

        encoded = [
            self.encode(text)
            for text in texts
        ]

        return (
            torch.stack([item[0] for item in encoded]),
            torch.stack([item[1] for item in encoded]),
        )


TOKENIZER = SimpleTokenizer(
    VOCABULARY,
    MAX_TEXT_LENGTH,
)


# =============================================================================
# 5. SYNTHETIC DATASET
# =============================================================================

def create_synthetic_dataset(samples_per_class=SAMPLES_PER_CLASS):

    images = []
    labels = []

    axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE)

    yy, xx = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    for class_id in range(NUM_CLASSES):

        for _ in range(samples_per_class):

            # Keep the task controlled:
            # only small position, size, and thickness variations.
            center_x = random.uniform(-0.025, 0.025)
            center_y = random.uniform(-0.025, 0.025)

            x = xx - center_x
            y = yy - center_y

            if class_id == 0:

                radius = random.uniform(0.42, 0.46)

                mask = (
                    x.square() + y.square()
                    <= radius ** 2
                )

            elif class_id == 1:

                half_width = random.uniform(0.39, 0.43)

                mask = (
                    (x.abs() <= half_width)
                    & (y.abs() <= half_width)
                )

            elif class_id == 2:

                thickness = random.uniform(0.055, 0.075)

                mask = (
                    (x - y).abs() <= thickness
                )

            else:

                thickness = random.uniform(0.055, 0.075)

                mask = (
                    (x.abs() <= thickness)
                    | (y.abs() <= thickness)
                )

            image = torch.zeros(
                IMAGE_SIZE,
                IMAGE_SIZE,
            )

            image[mask] = 1.0

            # Very small image noise.
            image += 0.003 * torch.randn_like(image)

            image = torch.clamp(image, 0.0, 1.0)

            # Diffusion data range.
            image = image * 2.0 - 1.0

            images.append(image.unsqueeze(0))
            labels.append(class_id)

    images = torch.stack(images)

    labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return images, labels


# =============================================================================
# 6. VISUALIZATION
# =============================================================================

def show_images(images, titles, title, columns=4):

    images = images.detach().float().cpu()
    images = ((images + 1.0) / 2.0).clamp(0.0, 1.0)

    count = min(len(images), len(titles))
    rows = math.ceil(count / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.5 * columns, 3.5 * rows),
    )

    axes = np.array(axes).reshape(-1)

    for index, ax in enumerate(axes):

        if index < count:

            ax.imshow(
                images[index, 0],
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )

            ax.set_title(titles[index])

        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_losses(losses):

    plt.figure(figsize=(9, 4))
    plt.plot(losses)

    plt.xlabel("Training step")
    plt.ylabel("Noise prediction MSE")
    plt.title("Improved text-to-image diffusion training")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 7. TEXT EMBEDDING
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
            raise ValueError("Input sequence exceeds maximum length")

        token_features = self.token_embedding(token_ids)

        position_features = self.position_embedding[
            :, :sequence_length, :
        ]

        return token_features + position_features


# =============================================================================
# 8. TRANSFORMER TEXT ENCODER
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

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=TEXT_EMBED_DIM,
            nhead=TEXT_NUM_HEADS,
            dim_feedforward=TEXT_FF_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=TEXT_NUM_LAYERS,
        )

        self.norm = nn.LayerNorm(TEXT_EMBED_DIM)

    def forward(self, token_ids, attention_mask):

        x = self.embedding(token_ids)

        padding_mask = ~attention_mask.bool()

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.norm(x)

        # Padded positions do not contribute to cross-attention or pooling.
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        return x


def masked_mean_pool(text_features, attention_mask):

    mask = attention_mask.unsqueeze(-1).to(
        text_features.dtype
    )

    numerator = (text_features * mask).sum(dim=1)

    denominator = mask.sum(dim=1).clamp(min=1.0)

    return numerator / denominator


# =============================================================================
# 9. TIME EMBEDDING
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
# 10. GROUP NORMALIZATION
# =============================================================================

def valid_group_count(channels, preferred_groups=8):

    preferred_groups = min(channels, preferred_groups)

    for groups in range(preferred_groups, 0, -1):

        if channels % groups == 0:
            return groups

    return 1


# =============================================================================
# 11. TEXT-CONDITIONED RESIDUAL BLOCK
# =============================================================================

class ConditionalResidualBlock(nn.Module):

    """
    Text and timestep features modulate every residual block.

    FiLM-style modulation:
        h = norm(h) * (1 + scale) + shift

    This provides a direct conditioning path even when cross-attention
    is not applied at the current resolution.
    """

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

        self.condition_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                condition_dim,
                out_channels * 2,
            ),
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, condition):

        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)

        h = self.conv1(h)

        scale_shift = self.condition_projection(condition)

        scale, shift = scale_shift.chunk(2, dim=1)

        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)

        h = h * (1.0 + scale) + shift

        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


# =============================================================================
# 12. TEXT CROSS-ATTENTION
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
    ):

        batch, channels, height, width = image_features.shape

        queries = image_features.flatten(2).transpose(1, 2)
        queries = self.image_norm(queries)

        text = self.text_norm(text_features)

        key_padding_mask = ~attention_mask.bool()

        attended, _ = self.attention(
            query=queries,
            key=text,
            value=text,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        attended = self.output_projection(attended)

        attended = attended.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )

        return image_features + attended


# =============================================================================
# 13. IMPROVED MULTI-RESOLUTION TEXT-CONDITIONED U-NET
# =============================================================================

class TextConditionedUNet(nn.Module):

    def __init__(self):
        super().__init__()

        c = BASE_CHANNELS

        self.time_embedding = SinusoidalTimeEmbedding(
            TIME_EMBED_DIM
        )

        self.condition_projection = nn.Sequential(
            nn.Linear(
                TIME_EMBED_DIM + TEXT_EMBED_DIM,
                CONDITION_DIM,
            ),
            nn.SiLU(),
            nn.Linear(CONDITION_DIM, CONDITION_DIM),
        )

        self.input_conv = nn.Conv2d(
            IMAGE_CHANNELS,
            c,
            3,
            padding=1,
        )

        # 32 x 32
        self.enc1 = ConditionalResidualBlock(
            c,
            c,
            CONDITION_DIM,
        )

        # 16 x 16
        self.down1 = nn.Conv2d(
            c,
            c * 2,
            3,
            stride=2,
            padding=1,
        )

        self.enc2 = ConditionalResidualBlock(
            c * 2,
            c * 2,
            CONDITION_DIM,
        )

        self.attn16_encoder = TextCrossAttention(
            c * 2,
            TEXT_EMBED_DIM,
            ATTENTION_HEADS,
        )

        # 8 x 8
        self.down2 = nn.Conv2d(
            c * 2,
            c * 4,
            3,
            stride=2,
            padding=1,
        )

        self.middle1 = ConditionalResidualBlock(
            c * 4,
            c * 4,
            CONDITION_DIM,
        )

        self.attn8 = TextCrossAttention(
            c * 4,
            TEXT_EMBED_DIM,
            ATTENTION_HEADS,
        )

        self.middle2 = ConditionalResidualBlock(
            c * 4,
            c * 4,
            CONDITION_DIM,
        )

        # Back to 16 x 16.
        self.up1 = nn.ConvTranspose2d(
            c * 4,
            c * 2,
            4,
            stride=2,
            padding=1,
        )

        self.dec1 = ConditionalResidualBlock(
            c * 4,
            c * 2,
            CONDITION_DIM,
        )

        self.attn16_decoder = TextCrossAttention(
            c * 2,
            TEXT_EMBED_DIM,
            ATTENTION_HEADS,
        )

        # Back to 32 x 32.
        self.up2 = nn.ConvTranspose2d(
            c * 2,
            c,
            4,
            stride=2,
            padding=1,
        )

        self.dec2 = ConditionalResidualBlock(
            c * 2,
            c,
            CONDITION_DIM,
        )

        self.output_norm = nn.GroupNorm(
            valid_group_count(c),
            c,
        )

        self.output_conv = nn.Conv2d(
            c,
            IMAGE_CHANNELS,
            3,
            padding=1,
        )

    def forward(
        self,
        noisy_images,
        timesteps,
        text_features,
        attention_mask,
    ):

        time_features = self.time_embedding(timesteps)

        pooled_text = masked_mean_pool(
            text_features,
            attention_mask,
        )

        combined_condition = torch.cat(
            [time_features, pooled_text],
            dim=1,
        )

        condition = self.condition_projection(
            combined_condition
        )

        # Encoder: 32 x 32.
        h = self.input_conv(noisy_images)

        skip32 = self.enc1(h, condition)

        # Encoder: 16 x 16.
        h = self.down1(skip32)

        skip16 = self.enc2(h, condition)

        skip16 = self.attn16_encoder(
            skip16,
            text_features,
            attention_mask,
        )

        # Bottleneck: 8 x 8.
        h = self.down2(skip16)

        h = self.middle1(h, condition)

        h = self.attn8(
            h,
            text_features,
            attention_mask,
        )

        h = self.middle2(h, condition)

        # Decoder: 16 x 16.
        h = self.up1(h)

        if h.shape[-2:] != skip16.shape[-2:]:
            raise RuntimeError("16x16 skip connection mismatch")

        h = torch.cat([h, skip16], dim=1)

        h = self.dec1(h, condition)

        h = self.attn16_decoder(
            h,
            text_features,
            attention_mask,
        )

        # Decoder: 32 x 32.
        h = self.up2(h)

        if h.shape[-2:] != skip32.shape[-2:]:
            raise RuntimeError("32x32 skip connection mismatch")

        h = torch.cat([h, skip32], dim=1)

        h = self.dec2(h, condition)

        h = F.silu(self.output_norm(h))

        return self.output_conv(h)


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
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "posterior_variance": posterior_variance,
    }


def extract(values, timesteps, target_shape):

    if values.device != timesteps.device:
        raise RuntimeError(
            f"Schedule device {values.device} does not match "
            f"timestep device {timesteps.device}"
        )

    result = values.gather(0, timesteps)

    return result.reshape(
        timesteps.shape[0],
        *([1] * (len(target_shape) - 1)),
    )


# =============================================================================
# 15. FORWARD DIFFUSION
# =============================================================================

def q_sample(clean_images, timesteps, noise, schedule):

    sqrt_alpha_bar = extract(
        schedule["sqrt_alpha_bars"],
        timesteps,
        clean_images.shape,
    )

    sqrt_one_minus_alpha_bar = extract(
        schedule["sqrt_one_minus_alpha_bars"],
        timesteps,
        clean_images.shape,
    )

    return (
        sqrt_alpha_bar * clean_images
        + sqrt_one_minus_alpha_bar * noise
    )


# =============================================================================
# 16. DATASET ALIGNMENT
# =============================================================================

def validate_dataset_alignment(
    images,
    token_ids,
    attention_masks,
):

    if images.shape[0] != token_ids.shape[0]:
        raise ValueError("Image and token row counts do not match")

    if images.shape[0] != attention_masks.shape[0]:
        raise ValueError("Image and mask row counts do not match")

    if token_ids.shape != attention_masks.shape:
        raise ValueError("Token and mask shapes do not match")

    return True


# =============================================================================
# 17. TRAINING
# =============================================================================

def train_text_to_image(
    diffusion_model,
    text_encoder,
    images_cpu,
    token_ids_cpu,
    masks_cpu,
):

    device = check_model_devices(
        text_encoder,
        diffusion_model,
    )

    validate_dataset_alignment(
        images_cpu,
        token_ids_cpu,
        masks_cpu,
    )

    # Keep complete training data on CPU.
    images_cpu = images_cpu.detach().cpu()
    token_ids_cpu = token_ids_cpu.detach().cpu()
    masks_cpu = masks_cpu.detach().cpu()

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

    for step in range(TRAIN_STEPS):

        # CPU indices index CPU datasets.
        indices = torch.randint(
            0,
            len(images_cpu),
            (BATCH_SIZE,),
            device="cpu",
        )

        batch_images = images_cpu[indices].to(device)
        batch_tokens = token_ids_cpu[indices].to(device)
        batch_masks = masks_cpu[indices].to(device)

        timesteps = torch.randint(
            0,
            DIFFUSION_STEPS,
            (BATCH_SIZE,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(batch_images)

        noisy_images = q_sample(
            batch_images,
            timesteps,
            noise,
            schedule,
        )

        text_features = text_encoder(
            batch_tokens,
            batch_masks,
        )

        predicted_noise = diffusion_model(
            noisy_images,
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
            parameters,
            max_norm=1.0,
        )

        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 200 == 0:

            recent_loss = float(np.mean(losses[-200:]))

            print(
                f"Step {step + 1:5d}/{TRAIN_STEPS} "
                f"recent MSE={recent_loss:.6f}"
            )

    return losses


# =============================================================================
# 18. DDPM SAMPLER
# =============================================================================

@torch.no_grad()
def sample_text_to_image(
    diffusion_model,
    text_encoder,
    token_ids,
    attention_masks,
    schedule,
    num_samples,
    initial_noise=None,
    step_noise_seed=None,
):

    device = check_model_devices(
        text_encoder,
        diffusion_model,
    )

    token_ids = token_ids.to(
        device,
        dtype=torch.long,
    )

    attention_masks = attention_masks.to(
        device,
        dtype=torch.bool,
    )

    if token_ids.shape[0] == 1 and num_samples > 1:

        token_ids = token_ids.expand(num_samples, -1)
        attention_masks = attention_masks.expand(num_samples, -1)

    if token_ids.shape[0] != num_samples:
        raise ValueError("Prompt batch size must equal num_samples")

    schedule = move_schedule_to_device(schedule, device)

    diffusion_model.eval()
    text_encoder.eval()

    text_features = text_encoder(
        token_ids,
        attention_masks,
    )

    if initial_noise is None:

        images = torch.randn(
            num_samples,
            IMAGE_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=device,
        )

    else:

        images = initial_noise.to(device).clone()

        expected_shape = (
            num_samples,
            IMAGE_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        if tuple(images.shape) != expected_shape:
            raise ValueError(
                f"Initial noise shape must be {expected_shape}"
            )

    noise_generator = None

    if step_noise_seed is not None:

        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(step_noise_seed)

    for timestep in reversed(range(DIFFUSION_STEPS)):

        t = torch.full(
            (num_samples,),
            timestep,
            device=device,
            dtype=torch.long,
        )

        predicted_noise = diffusion_model(
            images,
            t,
            text_features,
            attention_masks,
        )

        beta_t = extract(
            schedule["betas"],
            t,
            images.shape,
        )

        sqrt_one_minus_alpha_bar = extract(
            schedule["sqrt_one_minus_alpha_bars"],
            t,
            images.shape,
        )

        sqrt_recip_alpha = extract(
            schedule["sqrt_recip_alphas"],
            t,
            images.shape,
        )

        mean = sqrt_recip_alpha * (
            images
            - beta_t
            * predicted_noise
            / sqrt_one_minus_alpha_bar
        )

        if timestep > 0:

            variance = extract(
                schedule["posterior_variance"],
                t,
                images.shape,
            )

            if noise_generator is None:
                step_noise = torch.randn_like(images)
            else:
                step_noise = torch.randn(
                    images.shape,
                    device=device,
                    dtype=images.dtype,
                    generator=noise_generator,
                )

            images = (
                mean
                + torch.sqrt(torch.clamp(variance, min=0.0))
                * step_noise
            )

        else:

            images = mean

    return images.clamp(-1.0, 1.0)


# =============================================================================
# 19. INDEPENDENT CLASSIFIER
# =============================================================================

class ShapeClassifier(nn.Module):

    """
    Independent diagnostic model.

    It does not participate in diffusion training or image generation.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, 3, padding=1),
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

        x = self.features(x)
        x = x.flatten(1)

        return self.classifier(x)


def train_independent_classifier(
    classifier,
    images_cpu,
    labels_cpu,
):

    device = get_model_device(classifier)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=1e-3,
    )

    classifier.train()

    for epoch in range(CLASSIFIER_EPOCHS):

        permutation = torch.randperm(len(images_cpu))

        correct = 0
        total = 0
        total_loss = 0.0

        for start in range(0, len(images_cpu), CLASSIFIER_BATCH_SIZE):

            indices = permutation[
                start:start + CLASSIFIER_BATCH_SIZE
            ]

            x = images_cpu[indices].to(device)
            y = labels_cpu[indices].to(device)

            logits = classifier(x)

            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += len(y)

        print(
            f"Classifier epoch {epoch + 1:2d}/{CLASSIFIER_EPOCHS} "
            f"accuracy={correct / max(total, 1):.4f} "
            f"loss={total_loss:.4f}"
        )

    return classifier


@torch.no_grad()
def evaluate_generated_images(classifier, generated_images):

    device = get_model_device(classifier)

    classifier.eval()

    logits = classifier(generated_images.to(device))

    probabilities = torch.softmax(logits, dim=1)

    predictions = probabilities.argmax(dim=1)

    return predictions, probabilities


# =============================================================================
# 20. IMAGE DIAGNOSTICS
# =============================================================================

def print_image_diagnostics(images):

    images = images.detach().float()

    print("\nIMAGE DIAGNOSTICS")

    print(f"Minimum: {images.min().item():.4f}")
    print(f"Maximum: {images.max().item():.4f}")
    print(f"Mean: {images.mean().item():.4f}")
    print(f"Standard deviation: {images.std().item():.4f}")

    for index in range(len(images)):

        image_std = images[index].std().item()
        image_mean = images[index].mean().item()

        print(
            f"Image {index}: mean={image_mean:.4f}, "
            f"std={image_std:.4f}"
        )

        if image_std < 0.05:
            print("  WARNING: image is nearly constant")


# =============================================================================
# 21. TESTS
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


def test_unknown_tokens():

    print("\nTEST 2 — UNKNOWN TOKENS")

    ids, mask = TOKENIZER.encode("unknown object")

    assert ids[0].item() == TOKENIZER.unk_id
    assert ids[1].item() == TOKENIZER.unk_id
    assert mask[:2].all()

    print("PASS")


def test_text_encoder():

    print("\nTEST 3 — TEXT ENCODER")

    model = TextEncoder().to(DEVICE)

    ids, masks = TOKENIZER.batch_encode(PROMPTS)

    ids = ids.to(get_model_device(model))
    masks = masks.to(get_model_device(model))

    output = model(ids, masks)

    assert output.shape == (
        NUM_CLASSES,
        MAX_TEXT_LENGTH,
        TEXT_EMBED_DIM,
    )

    assert output.device == get_model_device(model)
    assert torch.isfinite(output).all()

    print("Output:", tuple(output.shape))
    print("Device:", output.device)
    print("PASS")


def test_padding_mask():

    print("\nTEST 4 — PADDING MASK")

    model = TextEncoder().to(DEVICE)
    model.eval()

    ids, masks = TOKENIZER.encode("make a circle")

    device = get_model_device(model)

    ids = ids.unsqueeze(0).to(device)
    masks = masks.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(ids, masks)

    padded_output = output[~masks]

    assert torch.allclose(
        padded_output,
        torch.zeros_like(padded_output),
        atol=1e-6,
    )

    print("PASS")


def test_diffusion_schedule():

    print("\nTEST 5 — DIFFUSION SCHEDULE")

    schedule = make_diffusion_schedule(DEVICE)

    assert schedule["betas"].shape == (DIFFUSION_STEPS,)
    assert torch.all(schedule["betas"] > 0)
    assert torch.all(schedule["betas"] < 1)

    assert torch.all(
        schedule["alpha_bars"][1:]
        <= schedule["alpha_bars"][:-1]
    )

    print("Final alpha_bar:", schedule["alpha_bars"][-1].item())
    print("PASS")


def test_forward_diffusion():

    print("\nTEST 6 — FORWARD DIFFUSION")

    schedule = make_diffusion_schedule(DEVICE)

    clean = torch.randn(
        4,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    noise = torch.randn_like(clean)

    timesteps = torch.tensor(
        [0, 10, 50, DIFFUSION_STEPS - 1],
        device=DEVICE,
        dtype=torch.long,
    )

    noisy = q_sample(clean, timesteps, noise, schedule)

    assert noisy.shape == clean.shape
    assert noisy.device == clean.device
    assert torch.isfinite(noisy).all()

    print("PASS")


def test_cross_attention():

    print("\nTEST 7 — CROSS-ATTENTION")

    attention = TextCrossAttention(
        image_channels=64,
        text_dim=TEXT_EMBED_DIM,
        num_heads=4,
    ).to(DEVICE)

    image_features = torch.randn(
        2, 64, 4, 4, device=DEVICE
    )

    text_features = torch.randn(
        2, MAX_TEXT_LENGTH, TEXT_EMBED_DIM, device=DEVICE
    )

    masks = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
        ],
        device=DEVICE,
    )

    output = attention(image_features, text_features, masks)

    assert output.shape == image_features.shape
    assert torch.isfinite(output).all()

    print("PASS")


def test_unet_output():

    print("\nTEST 8 — MULTI-RESOLUTION U-NET")

    text_encoder = TextEncoder().to(DEVICE)
    model = TextConditionedUNet().to(DEVICE)

    device = get_model_device(model)

    ids, masks = TOKENIZER.batch_encode(PROMPTS[:2])

    ids = ids.to(device)
    masks = masks.to(device)

    text_features = text_encoder(ids, masks)

    images = torch.randn(
        2,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    timesteps = torch.tensor(
        [10, 50],
        device=device,
        dtype=torch.long,
    )

    output = model(
        images,
        timesteps,
        text_features,
        masks,
    )

    assert output.shape == images.shape
    assert torch.isfinite(output).all()

    print("Output:", tuple(output.shape))
    print("PASS")


def test_gradient_propagation():

    print("\nTEST 9 — GRADIENT PROPAGATION")

    text_encoder = TextEncoder().to(DEVICE)
    model = TextConditionedUNet().to(DEVICE)

    device = check_model_devices(text_encoder, model)

    ids, masks = TOKENIZER.batch_encode(PROMPTS[:2])

    ids = ids.to(device)
    masks = masks.to(device)

    images = torch.randn(
        2,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    timesteps = torch.tensor(
        [5, 30],
        device=device,
        dtype=torch.long,
    )

    text_features = text_encoder(ids, masks)

    output = model(images, timesteps, text_features, masks)

    output.square().mean().backward()

    text_gradients = [
        p.grad
        for p in text_encoder.parameters()
        if p.grad is not None
    ]

    model_gradients = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]

    assert len(text_gradients) > 0
    assert len(model_gradients) > 0

    assert any(g.abs().sum().item() > 0 for g in text_gradients)
    assert any(g.abs().sum().item() > 0 for g in model_gradients)

    print("PASS")


def test_device_consistency():

    print("\nTEST 10 — DEVICE CONSISTENCY")

    text_encoder = TextEncoder().to(DEVICE)
    model = TextConditionedUNet().to(DEVICE)

    actual_device = check_model_devices(text_encoder, model)

    ids, masks = TOKENIZER.batch_encode(PROMPTS[:2])

    ids = ids.to(actual_device)
    masks = masks.to(actual_device)

    output = text_encoder(ids, masks)

    assert output.device == actual_device

    print("Actual model device:", actual_device)
    print("PASS")


def test_dataset_alignment():

    print("\nTEST 11 — DATASET ALIGNMENT")

    images = torch.randn(
        8,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    tokens, masks = TOKENIZER.batch_encode(PROMPTS * 2)

    assert validate_dataset_alignment(images, tokens, masks)

    error_caught = False

    try:
        validate_dataset_alignment(images, tokens[:4], masks)
    except ValueError:
        error_caught = True

    assert error_caught

    print("PASS")


def test_complete_sampler():

    print("\nTEST 12 — COMPLETE DDPM SAMPLER")

    text_encoder = TextEncoder().to(DEVICE)
    model = TextConditionedUNet().to(DEVICE)

    actual_device = check_model_devices(text_encoder, model)

    schedule = make_diffusion_schedule(actual_device)

    ids, masks = TOKENIZER.batch_encode(PROMPTS[:2])

    generated = sample_text_to_image(
        model,
        text_encoder,
        ids,
        masks,
        schedule,
        num_samples=2,
    )

    assert generated.shape == (
        2,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    # Check the actual model device, not the global DEVICE variable.
    assert generated.device == actual_device
    assert torch.isfinite(generated).all()

    print("Generated:", tuple(generated.shape))
    print("Actual device:", generated.device)
    print("PASS")


# =============================================================================
# 22. MAIN
# =============================================================================

def main():

    set_seed()

    print("=" * 80)
    print("LESSON 27 — IMPROVED TEXT-TO-IMAGE DIFFUSION")
    print("=" * 80)

    print("PyTorch:", torch.__version__)
    print("Configured device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Vocabulary size:", VOCAB_SIZE)
    print("Text embedding dimension:", TEXT_EMBED_DIM)
    print("Diffusion steps:", DIFFUSION_STEPS)
    print("Training steps:", TRAIN_STEPS)
    print("Batch size:", BATCH_SIZE)

    # -------------------------------------------------------------------------
    # UNIT TESTS
    # -------------------------------------------------------------------------

    tests = [
        test_tokenizer,
        test_unknown_tokens,
        test_text_encoder,
        test_padding_mask,
        test_diffusion_schedule,
        test_forward_diffusion,
        test_cross_attention,
        test_unet_output,
        test_gradient_propagation,
        test_device_consistency,
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

    images, labels = create_synthetic_dataset()

    print("Images:", tuple(images.shape))
    print("Labels:", tuple(labels.shape))

    sample_indices = []

    for class_id in range(NUM_CLASSES):

        index = torch.where(labels == class_id)[0][0].item()
        sample_indices.append(index)

    show_images(
        images[sample_indices],
        PROMPTS,
        "Training dataset examples",
    )

    # -------------------------------------------------------------------------
    # TEXT DATA
    # -------------------------------------------------------------------------

    print("\nTOKENIZING TRAINING PROMPTS")

    dataset_prompts = [
        PROMPTS[label]
        for label in labels.tolist()
    ]

    token_ids, attention_masks = TOKENIZER.batch_encode(
        dataset_prompts
    )

    validate_dataset_alignment(
        images,
        token_ids,
        attention_masks,
    )

    print("Token IDs:", tuple(token_ids.shape))
    print("Masks:", tuple(attention_masks.shape))

    # -------------------------------------------------------------------------
    # BUILD MODELS
    # -------------------------------------------------------------------------

    print("\nBUILDING TEXT ENCODER")

    text_encoder = TextEncoder().to(DEVICE)

    print(
        "Text encoder parameters:",
        sum(p.numel() for p in text_encoder.parameters()),
    )

    print("\nBUILDING MULTI-RESOLUTION DIFFUSION U-NET")

    diffusion_model = TextConditionedUNet().to(DEVICE)

    print(
        "Diffusion parameters:",
        sum(p.numel() for p in diffusion_model.parameters()),
    )

    # -------------------------------------------------------------------------
    # TRAIN GENERATOR
    # -------------------------------------------------------------------------

    print("\nTRAINING TEXT-TO-IMAGE MODEL")

    start_time = time.time()

    losses = train_text_to_image(
        diffusion_model,
        text_encoder,
        images,
        token_ids,
        attention_masks,
    )

    training_time = time.time() - start_time

    print("\nTRAINING COMPLETE")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Last 200-step average: {np.mean(losses[-200:]):.6f}")

    plot_losses(losses)

    # -------------------------------------------------------------------------
    # TRAIN INDEPENDENT CLASSIFIER
    # -------------------------------------------------------------------------

    print("\nTRAINING INDEPENDENT EVALUATION CLASSIFIER")

    classifier = ShapeClassifier().to(DEVICE)

    classifier = train_independent_classifier(
        classifier,
        images,
        labels,
    )

    # -------------------------------------------------------------------------
    # GENERATE ALL FOUR PROMPTS
    # -------------------------------------------------------------------------

    print("\nGENERATING IMAGES FROM TEXT")

    prompt_ids, prompt_masks = TOKENIZER.batch_encode(PROMPTS)

    device = check_model_devices(
        text_encoder,
        diffusion_model,
    )

    schedule = make_diffusion_schedule(device)

    generated_images = sample_text_to_image(
        diffusion_model,
        text_encoder,
        prompt_ids,
        prompt_masks,
        schedule,
        num_samples=NUM_GENERATED,
    )

    print_image_diagnostics(generated_images)

    # -------------------------------------------------------------------------
    # INDEPENDENT EVALUATION
    # -------------------------------------------------------------------------

    print("\nEVALUATING GENERATED IMAGES")

    predictions, probabilities = evaluate_generated_images(
        classifier,
        generated_images,
    )

    correct = 0

    for index, prompt in enumerate(PROMPTS):

        predicted_class = predictions[index].item()

        confidence = probabilities[
            index,
            predicted_class,
        ].item()

        expected_class = index

        if predicted_class == expected_class:
            correct += 1

        print(
            f"Prompt: {prompt:20s} "
            f"Predicted: {PROMPTS[predicted_class]:20s} "
            f"Confidence: {confidence:.4f}"
        )

    print(
        f"\nDiagnostic classifier agreement: "
        f"{correct}/{NUM_CLASSES}"
    )

    show_images(
        generated_images,
        [
            f"Prompt: {PROMPTS[i]}\n"
            f"Predicted: {PROMPTS[predictions[i].item()]}"
            for i in range(NUM_CLASSES)
        ],
        "Improved generated images and classifier predictions",
    )

    # -------------------------------------------------------------------------
    # SAME INITIAL NOISE / DIFFERENT PROMPTS
    # -------------------------------------------------------------------------

    print("\nSAME INITIAL NOISE / DIFFERENT TEXT PROMPTS")

    shared_noise = torch.randn(
        1,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    shared_noise = shared_noise.expand(
        NUM_CLASSES,
        -1,
        -1,
        -1,
    ).contiguous()

    same_noise_images = sample_text_to_image(
        diffusion_model,
        text_encoder,
        prompt_ids,
        prompt_masks,
        schedule,
        num_samples=NUM_CLASSES,
        initial_noise=shared_noise,
        step_noise_seed=SEED + 100,
    )

    show_images(
        same_noise_images,
        PROMPTS,
        "Same initial noise — different text prompts",
    )

    # -------------------------------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LESSON 27 COMPLETE")
    print("=" * 80)

    print("Tokenizer: custom vocabulary")
    print("Text encoder: Transformer")
    print("Text conditioning: FiLM-style residual modulation")
    print("Cross-attention: 16x16 and 8x8 resolutions")
    print("Generator: image-space DDPM U-Net")
    print("Sampler: complete DDPM reverse process")
    print("Evaluation: independently trained classifier")
    print("Pretrained models: NONE")

    print("\nIMPORTANT:")
    print(
        "Classifier agreement is a diagnostic metric, not proof "
        "that every generated image is correct."
    )

    print("\nNEXT DEVELOPMENT DIRECTIONS:")
    print("1. Classifier-free guidance.")
    print("2. Exponential moving average of model weights.")
    print("3. Improved sampling algorithms.")
    print("4. Larger and more diverse datasets.")
    print("5. More expressive text conditioning.")

    print("\nLESSON 27 FINISHED")


if __name__ == "__main__":
    main()
