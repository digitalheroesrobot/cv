"""
================================================================================
LESSON 13 — VISION TRANSFORMER FROM SCRATCH
================================================================================

PURPOSE
-------
Build and understand a Vision Transformer (ViT) for image classification
without using pretrained models or pretrained weights.

This lesson moves from CNN-style local feature extraction toward
Transformer-style global image reasoning.

The entire lesson is implemented in ONE runnable Python file.

IMPORTANT:
    No pretrained model.
    No pretrained checkpoint.
    All weights are randomly initialized and trained from scratch.


WHAT YOU WILL LEARN
-------------------
1. Why images can be represented as sequences of patches.
2. Patch embedding.
3. Token sequences.
4. The CLS token.
5. Positional embeddings.
6. Query / Key / Value.
7. Scaled dot-product self-attention.
8. Multi-head self-attention.
9. Attention probability normalization.
10. Transformer MLP blocks.
11. Residual connections.
12. Layer normalization.
13. Complete Vision Transformer architecture.
14. Backpropagation through attention.
15. Attention visualization.
16. Positional-information ablation.
17. Practical computer-vision interpretation.


COURSE ROADMAP
--------------
Previous:
    Lesson 01  Images as numerical data
    Lesson 02  Vectors, matrices, tensors
    Lesson 03  Gradient descent
    Lesson 04  Neural networks from scratch
    Lesson 05  Convolution from scratch
    Lesson 06  CNN from scratch
    Lesson 07  Practical CNN training with PyTorch
    Lesson 08  Deep CNN / residual learning
    Lesson 09  Autoencoder
    Lesson 10  Variational Autoencoder
    Lesson 11  GAN from scratch
    Lesson 12  Representation / generative vision foundations

Current:
    Lesson 13  Vision Transformer from scratch

Next:
    Lesson 14  Generative modeling fundamentals
    Lesson 15  Probability distributions for image generation
    Lesson 16  Forward diffusion process
    Lesson 17  Noise schedules
    Lesson 18  Denoising model
    Lesson 19  DDPM
    Lesson 20  U-Net for diffusion
    Lesson 21  Sampling
    Lesson 22  DDIM
    Lesson 23  Latent diffusion
    Lesson 24  Conditioning
    Lesson 25  Cross-attention
    Lesson 26  Text conditioning
    Lesson 27  Own text-to-image model


REAL-WORLD PROBLEM
------------------
CNNs are excellent at local spatial processing.

However, many vision problems require relationships between distant image
regions.

Example:

    A robot sees a cup on the left side of an image and a hand/object on
    the right side.

The meaning of one region can depend on another distant region.

CNN:
    local receptive fields gradually grow through depth.

Transformer:
    self-attention can directly compare every token with every other token.

For robotics and VLA systems, this is important because an observation may
contain:

    robot body
    object
    workspace
    obstacles
    target
    human hand
    other objects

and their relationships matter.


CORE IDEA
---------
Instead of processing an image only as a 2-D grid:

        IMAGE
          |
          v
    split into patches
          |
          v
    patch tokens
          |
          v
    Transformer
          |
          v
    image representation
          |
          v
    classification


PATCH REPRESENTATION
--------------------
Suppose:

    image = 32 x 32
    patch = 8 x 8

Then:

    patches_per_side = 32 / 8 = 4

and:

    number_of_patches = 4 x 4 = 16

Each 8 x 8 patch becomes one token.


PATCH EMBEDDING
---------------
For grayscale images:

    patch shape = 8 x 8 x 1

A learnable projection converts each patch into:

    embedding dimension = 64

Therefore:

    image
        -> 16 patches
        -> 16 tokens
        -> each token has 64 values

Tensor:

    [B, N, D]

where:

    B = batch size
    N = number of patches
    D = embedding dimension


CLS TOKEN
---------
A learnable classification token is added:

    [CLS], patch_1, patch_2, ..., patch_N

Therefore sequence length becomes:

    N + 1

For this lesson:

    N = 16

so:

    sequence length = 17


POSITIONAL EMBEDDING
--------------------
Self-attention itself does not know where a token originated.

For example:

    patch at top-left

and

    patch at bottom-right

would otherwise have no inherent positional identity.

Therefore we add learnable positional embeddings:

    X_positioned = X + P

where:

    X = token embeddings
    P = positional embeddings


SELF-ATTENTION
--------------
Given token matrix X:

    X -> Q
    X -> K
    X -> V

using learned linear projections:

    Q = X W_Q
    K = X W_K
    V = X W_V


ATTENTION SCORE
---------------
Similarity between query and key:

    score = Q K^T


SCALED ATTENTION
----------------
Large embedding dimensions can create large dot products.

Therefore:

    AttentionScore = Q K^T / sqrt(d_k)

where:

    d_k = key dimension


SOFTMAX
-------
Convert scores into probabilities:

    A = softmax(AttentionScore)

For each query token:

    sum_j A[i,j] = 1


IMPORTANT TEST
--------------
The attention normalization test must inspect the TRUE probability matrix
BEFORE dropout.

During training:

    attention_probs
          |
          v
       dropout
          |
          v
    attention_used

Dropout intentionally removes some probabilities.

Therefore the post-dropout matrix does NOT necessarily have rows summing
to exactly 1.

The mathematically normalized object is:

    attention_probs = softmax(scores)

This lesson returns that tensor for testing and visualization.


MULTI-HEAD ATTENTION
--------------------
Instead of learning one attention relationship, use several attention heads.

For:

    embedding dimension = 64
    heads = 4

each head has:

    head dimension = 64 / 4 = 16

Each head can learn different relationships.

Conceptually:

    Head 1 -> local relationship
    Head 2 -> object relationship
    Head 3 -> shape relationship
    Head 4 -> global relationship

The actual learned behavior is determined by training.


TRANSFORMER BLOCK
-----------------
A standard pre-LayerNorm Transformer block is:

    x = x + Attention(LayerNorm(x))

    x = x + MLP(LayerNorm(x))

This combines:

    LayerNorm
    Self-Attention
    Residual connection
    MLP
    Residual connection


MLP
---
The Transformer feed-forward network is:

    D -> 4D -> D

with:

    Linear
    GELU
    Dropout
    Linear
    Dropout


VISION TRANSFORMER
------------------
Complete architecture:

    Image
      |
      v
    Patch Embedding
      |
      v
    Patch Tokens
      |
      +---- CLS token
      |
      +---- Positional embedding
      |
      v
    Transformer Block
      |
      v
    Transformer Block
      |
      v
    Transformer Block
      |
      v
    LayerNorm
      |
      v
    CLS token
      |
      v
    Classification Head


MATHEMATICAL SUMMARY
--------------------
Patch embedding:

    Z = PatchEmbed(X)

Token sequence:

    X_0 = [CLS ; Z] + P

Attention:

    Q = X W_Q
    K = X W_K
    V = X W_V

    S = Q K^T / sqrt(d_k)

    A = softmax(S)

    O = A V

Multi-head attention:

    MHA(X) = Concat(head_1, ..., head_h) W_O

Transformer block:

    X' = X + MHA(LN(X))

    X'' = X' + MLP(LN(X'))

Classification:

    logits = Head(LN(X'')[:, 0])


REAL-WORLD APPLICATION
----------------------
A Vision Transformer can be used for:

    image classification
    object recognition
    visual inspection
    medical imaging
    satellite image analysis
    robot perception
    visual navigation
    multimodal models
    VLA systems


REAL-WORLD EXAMPLE
------------------
Imagine a warehouse robot.

Camera observation:

    shelf
    box
    robot gripper
    target container

A Transformer can create relationships between image regions.

For example:

    gripper <-> box
    box <-> container
    box <-> shelf

These relationships can later become useful for:

    manipulation
    navigation
    planning
    VLA action prediction


RESULT / WHAT YOU SHOULD SEE
----------------------------
The program should:

    1. Run all unit tests.
    2. Train a Vision Transformer.
    3. Show training/validation curves.
    4. Show a confusion matrix.
    5. Show predictions.
    6. Visualize patch structure.
    7. Visualize CLS attention.
    8. Run a positional-embedding ablation.
    9. Report test accuracy.
    10. Verify attention probability normalization.


IMPORTANT DEBUGGING NOTE
------------------------
There are TWO different objects:

    attention_probs
    attention_dropped

The first is the true probability distribution.

The second is the probability distribution after dropout.

Therefore:

    attention_probs.sum(dim=-1) ~= 1

but:

    attention_dropped.sum(dim=-1)

is NOT guaranteed to equal 1 during training.

This distinction is important in real Transformer implementations.


================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import math
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 32
CHANNELS = 1

PATCH_SIZE = 8
NUM_PATCHES_PER_SIDE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = NUM_PATCHES_PER_SIDE ** 2

EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
MLP_RATIO = 4

NUM_CLASSES = 3

TRAIN_SAMPLES = 3000
TEST_SAMPLES = 600

BATCH_SIZE = 64
EPOCHS = 20

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1

NUM_GENERATED_VISUALIZATION = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    """
    Make the experiment as reproducible as practical.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is useful for a teaching experiment.
    # It can reduce performance slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# SYNTHETIC IMAGE DATASET
# =============================================================================

class SyntheticShapesDataset(Dataset):
    """
    Three-class synthetic vision dataset.

    Class 0:
        horizontal bar

    Class 1:
        vertical bar

    Class 2:
        diagonal bar

    Random translation and Gaussian noise are added.

    The purpose is not to create a difficult benchmark.

    The purpose is to create a controlled visual problem where we can inspect
    what the Vision Transformer is learning.
    """

    def __init__(
        self,
        num_samples,
        image_size=IMAGE_SIZE,
        seed=SEED,
    ):
        self.num_samples = num_samples
        self.image_size = image_size

        rng = np.random.default_rng(seed)

        self.images = []
        self.labels = []

        for index in range(num_samples):
            label = index % NUM_CLASSES

            image = self._generate_image(
                label=label,
                rng=rng,
            )

            self.images.append(image)
            self.labels.append(label)

        self.images = np.stack(self.images).astype(np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def _generate_image(self, label, rng):
        h = self.image_size
        w = self.image_size

        image = np.zeros((h, w), dtype=np.float32)

        # Random translation.
        offset_x = int(rng.integers(-4, 5))
        offset_y = int(rng.integers(-4, 5))

        if label == 0:
            # Horizontal bar.
            center_y = h // 2 + offset_y
            thickness = 4

            y0 = max(0, center_y - thickness // 2)
            y1 = min(h, center_y + thickness // 2 + 1)

            x0 = max(0, 4 + offset_x)
            x1 = min(w, w - 4 + offset_x)

            image[y0:y1, x0:x1] = 1.0

        elif label == 1:
            # Vertical bar.
            center_x = w // 2 + offset_x
            thickness = 4

            x0 = max(0, center_x - thickness // 2)
            x1 = min(w, center_x + thickness // 2 + 1)

            y0 = max(0, 4 + offset_y)
            y1 = min(h, h - 4 + offset_y)

            image[y0:y1, x0:x1] = 1.0

        elif label == 2:
            # Diagonal bar.
            thickness = 3

            for y in range(h):
                x_center = int(
                    (y / max(1, h - 1)) * (w - 1)
                    + offset_x
                )

                for dx in range(-thickness, thickness + 1):
                    x = x_center + dx

                    yy = y + offset_y

                    if 0 <= x < w and 0 <= yy < h:
                        image[yy, x] = 1.0

        # Add small Gaussian sensor noise.
        noise = rng.normal(
            loc=0.0,
            scale=0.08,
            size=(h, w),
        ).astype(np.float32)

        image = image + noise

        image = np.clip(image, 0.0, 1.0)

        return image[None, :, :]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, label


# =============================================================================
# PATCH EMBEDDING
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert an image into a sequence of patch embeddings.

    A Conv2d with:

        kernel_size = patch_size
        stride = patch_size

    is mathematically equivalent to extracting non-overlapping patches and
    linearly projecting them into the embedding space.

    Input:

        [B, C, H, W]

    Output:

        [B, N, D]

    where:

        N = number of patches
        D = embedding dimension
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # [B, D, H/P, W/P]
        x = self.projection(x)

        # [B, D, N]
        x = x.flatten(2)

        # [B, N, D]
        x = x.transpose(1, 2)

        return x


# =============================================================================
# MULTI-HEAD SELF-ATTENTION
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.

    Input:

        [B, N, D]

    Output:

        [B, N, D]

    Attention map:

        [B, heads, N, N]

    The returned attention map is the TRUE softmax probability matrix
    BEFORE dropout.

    This is important for mathematical verification and visualization.
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3,
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.shape

        # ---------------------------------------------------------------
        # QKV projection
        # ---------------------------------------------------------------
        qkv = self.qkv(x)

        # [B, N, 3D]
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        # [3, B, heads, N, head_dim]
        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        q, k, v = qkv.unbind(0)

        # ---------------------------------------------------------------
        # Scaled dot-product attention
        # ---------------------------------------------------------------
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores * self.scale

        # ---------------------------------------------------------------
        # TRUE attention probabilities
        # ---------------------------------------------------------------
        attention_probs = torch.softmax(
            scores,
            dim=-1,
        )

        # ---------------------------------------------------------------
        # Dropout is applied AFTER softmax.
        #
        # Therefore attention_dropped is NOT guaranteed to sum to 1.
        #
        # attention_probs is returned for mathematical testing.
        # ---------------------------------------------------------------
        attention_dropped = self.attn_dropout(
            attention_probs
        )

        attended = torch.matmul(
            attention_dropped,
            v,
        )

        # [B, heads, N, head_dim]
        attended = attended.transpose(1, 2)

        # [B, N, D]
        attended = attended.reshape(
            batch_size,
            sequence_length,
            embed_dim,
        )

        output = self.projection(attended)

        output = self.projection_dropout(output)

        return output, attention_probs


# =============================================================================
# TRANSFORMER MLP
# =============================================================================

class TransformerMLP(nn.Module):
    """
    Feed-forward network inside a Transformer block.

    D -> 4D -> D

    GELU is used as the nonlinear activation.
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    ):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(
            embed_dim,
            hidden_dim,
        )

        self.activation = nn.GELU()

        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(
            hidden_dim,
            embed_dim,
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.dropout2(x)

        return x


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block.

        x = x + Attention(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = TransformerMLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x):
        attention_output, attention_probs = self.attention(
            self.norm1(x)
        )

        x = x + attention_output

        x = x + self.mlp(
            self.norm2(x)
        )

        return x, attention_probs


# =============================================================================
# VISION TRANSFORMER
# =============================================================================

class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer.

    Architecture:

        Image
          |
          v
        PatchEmbedding
          |
          v
        Patch tokens
          |
          +---- CLS token
          |
          +---- positional embedding
          |
          v
        Transformer blocks
          |
          v
        LayerNorm
          |
          v
        CLS token
          |
          v
        Classification head
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_ratio=MLP_RATIO,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embedding.num_patches

        # Learnable CLS token.
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # One positional embedding for CLS + every patch.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim,
            )
        )

        self.position_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(
            embed_dim,
            num_classes,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x,
        return_attention=False,
    ):
        batch_size = x.shape[0]

        # ---------------------------------------------------------------
        # Image -> patch tokens
        # ---------------------------------------------------------------
        x = self.patch_embedding(x)

        # ---------------------------------------------------------------
        # Expand CLS token for current batch.
        # ---------------------------------------------------------------
        cls_token = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        # ---------------------------------------------------------------
        # [CLS] + patch tokens
        # ---------------------------------------------------------------
        x = torch.cat(
            [cls_token, x],
            dim=1,
        )

        # ---------------------------------------------------------------
        # Add positional information.
        # ---------------------------------------------------------------
        x = x + self.position_embedding

        x = self.position_dropout(x)

        attention_maps = []

        # ---------------------------------------------------------------
        # Transformer blocks
        # ---------------------------------------------------------------
        for block in self.blocks:

            x, attention_probs = block(x)

            if return_attention:
                attention_maps.append(
                    attention_probs
                )

        # ---------------------------------------------------------------
        # Final normalization
        # ---------------------------------------------------------------
        x = self.norm(x)

        cls_representation = x[:, 0]

        logits = self.head(
            cls_representation
        )

        if return_attention:
            return (
                logits,
                attention_maps,
                cls_representation,
            )

        return logits


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def show_dataset_examples(
    dataset,
    num_examples=9,
):
    """
    Show examples from the synthetic dataset.
    """

    num_examples = min(
        num_examples,
        len(dataset),
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(8, 8),
    )

    axes = axes.flatten()

    for i in range(num_examples):

        image, label = dataset[i]

        axes[i].imshow(
            image.squeeze(0),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[i].set_title(
            f"class = {label}"
        )

        axes[i].axis("off")

    for i in range(
        num_examples,
        len(axes),
    ):
        axes[i].axis("off")

    plt.suptitle(
        "Synthetic Vision Transformer Dataset"
    )

    plt.tight_layout()
    plt.show()


def show_patch_grid(
    image,
    patch_size=PATCH_SIZE,
):
    """
    Visualize how an image is divided into patches.
    """

    image = image.squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    height, width = image.shape

    for x in range(
        0,
        width + 1,
        patch_size,
    ):
        ax.axvline(
            x - 0.5,
            linewidth=1,
        )

    for y in range(
        0,
        height + 1,
        patch_size,
    ):
        ax.axhline(
            y - 0.5,
            linewidth=1,
        )

    ax.set_title(
        "Vision Transformer Patch Partition"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot training and validation loss.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history["train_loss"],
        label="train loss",
    )

    ax.plot(
        history["val_loss"],
        label="validation loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_title("Vision Transformer Training")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_accuracy_history(history):
    """
    Plot training and validation accuracy.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history["train_accuracy"],
        label="train accuracy",
    )

    ax.plot(
        history["val_accuracy"],
        label="validation accuracy",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Vision Transformer Accuracy")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    confusion,
):
    """
    Plot confusion matrix.
    """

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(confusion)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_title(
        "Vision Transformer Confusion Matrix"
    )

    ax.set_xticks(
        range(NUM_CLASSES)
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):

            ax.text(
                j,
                i,
                int(confusion[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.show()


def visualize_cls_attention(
    model,
    image,
    device=DEVICE,
):
    """
    Visualize how the CLS token attends to image patches.

    For each Transformer layer:

        attention[:, CLS, patch]

    is extracted.

    The CLS token itself is excluded from the spatial visualization.
    """

    model.eval()

    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():

        _, attention_maps, _ = model(
            image_batch,
            return_attention=True,
        )

    num_layers = len(attention_maps)

    fig, axes = plt.subplots(
        1,
        num_layers + 1,
        figsize=(4 * (num_layers + 1), 4),
    )

    axes = np.asarray(axes).reshape(-1)

    axes[0].imshow(
        image.squeeze(0).cpu(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title("Input")
    axes[0].axis("off")

    for layer_index, attention in enumerate(
        attention_maps
    ):

        # attention:
        # [B, heads, tokens, tokens]
        #
        # CLS query:
        # token index 0
        #
        # Patch tokens:
        # token indices 1:17

        cls_to_patches = attention[
            0,
            :,
            0,
            1:,
        ]

        # Average attention across heads.
        cls_to_patches = cls_to_patches.mean(
            dim=0
        )

        spatial_size = NUM_PATCHES_PER_SIDE

        attention_map = cls_to_patches.reshape(
            spatial_size,
            spatial_size,
        )

        axes[layer_index + 1].imshow(
            attention_map.cpu(),
            cmap="viridis",
            interpolation="nearest",
        )

        axes[layer_index + 1].set_title(
            f"Layer {layer_index + 1}"
        )

        axes[layer_index + 1].set_xticks([])
        axes[layer_index + 1].set_yticks([])

    plt.suptitle(
        "CLS-to-Patch Attention"
    )

    plt.tight_layout()
    plt.show()


def visualize_predictions(
    model,
    dataset,
    device=DEVICE,
    num_examples=12,
):
    """
    Display predictions made by the trained ViT.
    """

    model.eval()

    num_examples = min(
        num_examples,
        len(dataset),
    )

    indices = np.linspace(
        0,
        len(dataset) - 1,
        num_examples,
        dtype=int,
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(10, 8),
    )

    axes = axes.flatten()

    with torch.no_grad():

        for plot_index, data_index in enumerate(
            indices
        ):

            image, label = dataset[data_index]

            logits = model(
                image.unsqueeze(0).to(device)
            )

            prediction = logits.argmax(
                dim=1
            ).item()

            axes[plot_index].imshow(
                image.squeeze(0),
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            axes[plot_index].set_title(
                f"true={label}, pred={prediction}"
            )

            axes[plot_index].axis("off")

    for i in range(
        num_examples,
        len(axes),
    ):
        axes[i].axis("off")

    plt.suptitle(
        "Vision Transformer Predictions"
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

        for true_label, predicted_label in zip(
            labels.cpu().numpy(),
            predictions.cpu().numpy(),
        ):
            confusion[
                true_label,
                predicted_label,
            ] += 1

    average_loss = total_loss / total
    accuracy = correct / total

    return (
        average_loss,
        accuracy,
        confusion,
    )


def train_model(
    model,
    train_loader,
    test_loader,
    device=DEVICE,
    epochs=EPOCHS,
):
    """
    Train the Vision Transformer.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    print()
    print("=" * 80)
    print("TRAINING VISION TRANSFORMER")
    print("=" * 80)

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_accuracy, _ = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f} | "
            f"val acc={val_accuracy:.4f}"
        )

    return history


# =============================================================================
# UNIT TEST 1 — PATCH EMBEDDING
# =============================================================================

def test_patch_embedding():
    """
    Verify image -> patch-token transformation.
    """

    print()
    print("TEST 1 — Patch Embedding")

    model = PatchEmbedding()

    x = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    y = model(x)

    expected_shape = (
        2,
        NUM_PATCHES,
        EMBED_DIM,
    )

    assert y.shape == expected_shape, (
        f"Expected {expected_shape}, "
        f"got {tuple(y.shape)}"
    )

    assert torch.isfinite(y).all()

    print(
        f"PASS: image {tuple(x.shape)} -> "
        f"tokens {tuple(y.shape)}"
    )


# =============================================================================
# UNIT TEST 2 — TRANSFORMER BLOCK
# =============================================================================

def test_transformer_block():
    """
    Verify Transformer block shape and numerical stability.
    """

    print()
    print("TEST 2 — Transformer Block")

    block = TransformerBlock()

    x = torch.randn(
        2,
        NUM_PATCHES + 1,
        EMBED_DIM,
    )

    y, attention = block(x)

    assert y.shape == x.shape

    expected_attention_shape = (
        2,
        NUM_HEADS,
        NUM_PATCHES + 1,
        NUM_PATCHES + 1,
    )

    assert attention.shape == (
        expected_attention_shape
    )

    assert torch.isfinite(y).all()
    assert torch.isfinite(attention).all()

    print(
        "PASS: Transformer block preserves "
        "token shape and produces valid attention."
    )


# =============================================================================
# UNIT TEST 3 — COMPLETE VIT
# =============================================================================

def test_complete_vit():
    """
    Verify the complete Vision Transformer.
    """

    print()
    print("TEST 3 — Complete Vision Transformer")

    model = VisionTransformer()

    x = torch.randn(
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    logits = model(x)

    expected_shape = (
        4,
        NUM_CLASSES,
    )

    assert logits.shape == expected_shape

    assert torch.isfinite(logits).all()

    print(
        f"PASS: ViT output shape = {tuple(logits.shape)}"
    )


# =============================================================================
# UNIT TEST 4 — GRADIENT FLOW
# =============================================================================

def test_gradient_flow():
    """
    Verify that gradients flow through the entire ViT.
    """

    print()
    print("TEST 4 — Gradient Flow")

    model = VisionTransformer()

    x = torch.randn(
        4,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    labels = torch.tensor(
        [0, 1, 2, 0],
        dtype=torch.long,
    )

    criterion = nn.CrossEntropyLoss()

    logits = model(x)

    loss = criterion(
        logits,
        labels,
    )

    loss.backward()

    gradient_count = 0
    nonzero_gradient_count = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            assert parameter.grad is not None

            gradient_count += 1

            if torch.any(
                parameter.grad != 0
            ):
                nonzero_gradient_count += 1

    assert gradient_count > 0
    assert nonzero_gradient_count > 0

    print(
        f"PASS: gradients reached "
        f"{nonzero_gradient_count}/{gradient_count} "
        f"trainable parameter tensors."
    )


# =============================================================================
# UNIT TEST 5 — ATTENTION NORMALIZATION
# =============================================================================

def test_attention_normalization():
    """
    Verify the mathematical normalization of TRUE attention probabilities.

    IMPORTANT:
        The model returns attention_probs BEFORE attention dropout.

    Therefore each row must sum to approximately 1:

        sum_j softmax(score[i,j]) = 1

    We intentionally do NOT test the post-dropout matrix because dropout
    changes the individual probabilities during training.
    """

    print()
    print("TEST 5 — Attention Normalization")

    model = VisionTransformer(
        num_layers=2,
    )

    x = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    _, attention_maps, _ = model(
        x,
        return_attention=True,
    )

    assert len(attention_maps) == 2

    for layer_index, attention in enumerate(
        attention_maps
    ):

        # attention shape:
        #
        # [B, heads, query_tokens, key_tokens]
        #
        # The final dimension is the probability distribution.

        row_sums = attention.sum(
            dim=-1
        )

        expected = torch.ones_like(
            row_sums
        )

        assert torch.allclose(
            row_sums,
            expected,
            atol=1e-5,
            rtol=1e-5,
        ), (
            f"Layer {layer_index}: "
            f"attention rows do not sum to 1."
        )

        assert torch.all(
            attention >= 0
        )

        assert torch.isfinite(
            attention
        ).all()

    print(
        "PASS: every true attention-probability "
        "row sums to approximately 1."
    )


# =============================================================================
# UNIT TEST 6 — DATASET
# =============================================================================

def test_dataset():
    """
    Verify dataset dimensions, labels, and numerical validity.
    """

    print()
    print("TEST 6 — Dataset")

    dataset = SyntheticShapesDataset(
        num_samples=30,
        seed=SEED,
    )

    assert len(dataset) == 30

    image, label = dataset[0]

    expected_shape = (
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert image.shape == expected_shape

    assert label.item() in range(
        NUM_CLASSES
    )

    assert torch.isfinite(
        image
    ).all()

    assert image.min() >= 0.0
    assert image.max() <= 1.0

    labels = [
        dataset[index][1].item()
        for index in range(len(dataset))
    ]

    assert set(labels) == {
        0,
        1,
        2,
    }

    print(
        "PASS: dataset shape, labels, range, "
        "and numerical values are valid."
    )


# =============================================================================
# POSITION ABLATION MODEL
# =============================================================================

class ViTWithoutPosition(nn.Module):
    """
    Ablation model with positional embeddings removed.

    This is useful for understanding why positional information matters.

    The Transformer still sees the same patch content, but it is no longer
    explicitly told the spatial position of each patch.
    """

    def __init__(self):
        super().__init__()

        self.patch_embedding = PatchEmbedding()

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                EMBED_DIM,
            )
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

        self.head = nn.Linear(
            EMBED_DIM,
            NUM_CLASSES,
        )

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

    def forward(self, x):

        batch_size = x.shape[0]

        x = self.patch_embedding(x)

        cls_token = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            [cls_token, x],
            dim=1,
        )

        for block in self.blocks:

            x, _ = block(x)

        x = self.norm(x)

        cls_representation = x[:, 0]

        return self.head(
            cls_representation
        )


def run_position_ablation(
    train_loader,
    test_loader,
    device=DEVICE,
):
    """
    Compare a model with positional embeddings against one without them.

    This is an educational experiment rather than a rigorous benchmark.
    """

    print()
    print("=" * 80)
    print("POSITIONAL EMBEDDING ABLATION")
    print("=" * 80)

    model = ViTWithoutPosition().to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    ablation_epochs = 5

    for epoch in range(
        ablation_epochs
    ):

        model.train()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

    _, accuracy, _ = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"ViT WITHOUT positional embeddings "
        f"after {ablation_epochs} epochs: "
        f"{accuracy:.4f}"
    )

    return accuracy


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    set_seed()

    print("=" * 80)
    print("LESSON 13 — VISION TRANSFORMER FROM SCRATCH")
    print("=" * 80)

    print()
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Number of patches: {NUM_PATCHES}")
    print(f"Embedding dimension: {EMBED_DIM}")
    print(f"Attention heads: {NUM_HEADS}")
    print(f"Transformer layers: {NUM_LAYERS}")
    print(f"Classes: {NUM_CLASSES}")

    # -------------------------------------------------------------------------
    # UNIT TESTS
    #
    # IMPORTANT BUG FIX:
    #
    # Do NOT create a local variable named "test_dataset" inside main().
    #
    # There is already a function named:
    #
    #     test_dataset()
    #
    # If we write:
    #
    #     test_dataset = SyntheticShapesDataset(...)
    #
    # Python considers test_dataset a local variable throughout main().
    #
    # Then:
    #
    #     test_dataset()
    #
    # attempts to call the uninitialized local variable.
    #
    # We therefore use:
    #
    #     train_ds
    #     test_ds
    #
    # for dataset objects.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING UNIT TESTS")
    print("=" * 80)

    test_patch_embedding()
    test_transformer_block()
    test_complete_vit()
    test_gradient_flow()
    test_attention_normalization()
    test_dataset()

    print()
    print("=" * 80)
    print("ALL UNIT TESTS PASSED")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    train_ds = SyntheticShapesDataset(
        num_samples=TRAIN_SAMPLES,
        seed=SEED,
    )

    test_ds = SyntheticShapesDataset(
        num_samples=TEST_SAMPLES,
        seed=SEED + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------------------------------------------------------
    # DATA VISUALIZATION
    # -------------------------------------------------------------------------

    show_dataset_examples(
        train_ds
    )

    sample_image, sample_label = train_ds[0]

    print()
    print(
        f"Example image label: {sample_label}"
    )

    show_patch_grid(
        sample_image
    )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    model = VisionTransformer().to(
        DEVICE
    )

    print()
    print("=" * 80)
    print("MODEL")
    print("=" * 80)

    print(model)

    parameter_count = count_parameters(
        model
    )

    print()
    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        epochs=EPOCHS,
    )

    # -------------------------------------------------------------------------
    # TRAINING CURVES
    # -------------------------------------------------------------------------

    plot_training_history(
        history
    )

    plot_accuracy_history(
        history
    )

    # -------------------------------------------------------------------------
    # FINAL EVALUATION
    # -------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy, confusion = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE,
    )

    print()
    print("=" * 80)
    print("FINAL TEST RESULT")
    print("=" * 80)

    print(
        f"Test loss     : {test_loss:.4f}"
    )

    print(
        f"Test accuracy : {test_accuracy:.4f}"
    )

    # -------------------------------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------------------------------

    print()
    print("Confusion Matrix:")
    print(confusion)

    plot_confusion_matrix(
        confusion
    )

    # -------------------------------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------------------------------

    visualize_predictions(
        model=model,
        dataset=test_ds,
        device=DEVICE,
        num_examples=12,
    )

    # -------------------------------------------------------------------------
    # ATTENTION VISUALIZATION
    # -------------------------------------------------------------------------

    attention_image, attention_label = test_ds[0]

    print()
    print(
        f"Attention visualization sample label: "
        f"{attention_label}"
    )

    visualize_cls_attention(
        model=model,
        image=attention_image,
        device=DEVICE,
    )

    # -------------------------------------------------------------------------
    # POSITIONAL EMBEDDING ABLATION
    # -------------------------------------------------------------------------

    positionless_accuracy = run_position_ablation(
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
    )

    # -------------------------------------------------------------------------
    # FINAL NUMERICAL VERIFICATION
    # -------------------------------------------------------------------------

    assert len(history["train_loss"]) == EPOCHS
    assert len(history["val_loss"]) == EPOCHS
    assert len(history["train_accuracy"]) == EPOCHS
    assert len(history["val_accuracy"]) == EPOCHS

    assert np.isfinite(
        np.asarray(
            history["train_loss"]
        )
    ).all()

    assert np.isfinite(
        np.asarray(
            history["val_loss"]
        )
    ).all()

    assert np.isfinite(
        np.asarray(
            history["train_accuracy"]
        )
    ).all()

    assert np.isfinite(
        np.asarray(
            history["val_accuracy"]
        )
    ).all()

    # Training should generally reduce the loss on this simple dataset.
    assert (
        history["train_loss"][-1]
        <= history["train_loss"][0]
    )

    # The synthetic problem is intentionally simple.
    # A healthy implementation should achieve substantially better than
    # random guessing.
    assert test_accuracy > 0.75, (
        "Test accuracy is unexpectedly low. "
        "Check the training process or implementation."
    )

    # -------------------------------------------------------------------------
    # REAL-WORLD INTERPRETATION
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("REAL-WORLD APPLICATION")
    print("=" * 80)

    print(
        """
A Vision Transformer can be used as the visual backbone of a robot.

Example:

    Camera
       |
       v
    Image
       |
       v
    Patch Embedding
       |
       v
    Transformer
       |
       v
    Visual Representation
       |
       +--------------------+
       |                    |
       v                    v
    Object              Robot/VLA
    Recognition         Policy
                            |
                            v
                         Action

For a manipulation robot, visual tokens can represent:

    object
    gripper
    table
    obstacle
    target
    workspace

Self-attention allows the network to learn relationships between these
different regions.

This is particularly relevant to modern VLA systems because the visual
representation can later be combined with language and robot state.
"""
    )

    # -------------------------------------------------------------------------
    # REAL-WORLD RESULT
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("REAL-WORLD RESULT")
    print("=" * 80)

    print(
        f"""
The model learned to classify three visual patterns from raw images.

Final test accuracy:
    {test_accuracy:.4f}

The important engineering result is not merely the accuracy.

The experiment demonstrated the complete pipeline:

    image
      ->
    patches
      ->
    token sequence
      ->
    positional information
      ->
    self-attention
      ->
    Transformer representation
      ->
    classification

The attention normalization test also verified the mathematical property:

    softmax(scores).sum(dim=-1) = 1

This is the foundation required before moving into generative Transformer
and diffusion architectures.
"""
    )

    # -------------------------------------------------------------------------
    # LESSON RESULT
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LESSON RESULT")
    print("=" * 80)

    print(
        """
SUCCESS.

You have now implemented a Vision Transformer from scratch using PyTorch
building blocks.

You understand:

    - image patchification
    - patch embeddings
    - token sequences
    - CLS token
    - positional embeddings
    - Q/K/V
    - scaled dot-product attention
    - multi-head attention
    - attention probability normalization
    - attention dropout
    - Transformer MLP
    - LayerNorm
    - residual connections
    - complete ViT training
    - gradient flow
    - attention visualization
    - positional ablation

Most importantly:

    You are no longer limited to CNN-style local image processing.

The next major step is to understand how neural networks can model
probability distributions and generate new data.

That takes us directly toward diffusion models.
"""
    )

    # -------------------------------------------------------------------------
    # NEXT LESSON
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("NEXT LESSON")
    print("=" * 80)

    print(
        """
LESSON 14 — GENERATIVE MODELING FUNDAMENTALS

We will move from:

    "What class is this image?"

to:

    "How can a neural network learn the distribution of images?"

We will study:

    p(x)
    probability distributions
    likelihood
    latent variables
    sampling
    score intuition
    noise
    denoising
    generative processes

This is the conceptual bridge from:

    Vision Transformer
            |
            v
    Generative Modeling
            |
            v
    Diffusion
            |
            v
    Image Generation
            |
            v
    Text-to-Image
"""
    )

    # -------------------------------------------------------------------------
    # POSITION ABLATION RESULT
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("POSITION ABLATION RESULT")
    print("=" * 80)

    print(
        f"""
ViT without positional embeddings:

    accuracy = {positionless_accuracy:.4f}

This experiment demonstrates an important Transformer concept:

    Self-attention describes relationships between tokens,
    but it does not inherently encode image coordinates.

Therefore positional information is an important component of Vision
Transformers when spatial arrangement matters.
"""
    )

    print()
    print("=" * 80)
    print("LESSON 13 COMPLETE")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()