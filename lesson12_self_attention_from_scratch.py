"""
================================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 12 — SELF-ATTENTION FROM SCRATCH
================================================================================

Filename:
    lesson12_self_attention_from_scratch.py

================================================================================
PURPOSE
================================================================================

Learn how self-attention works internally before using Transformer or ViT
architectures.

The goal is to understand the mathematical mechanism that allows one image
token to dynamically determine which other tokens are important.

This lesson implements:

    1. Query / Key / Value projections
    2. Scaled dot-product attention
    3. Numerically stable softmax
    4. Attention masks
    5. Multi-head self-attention
    6. Attention visualization
    7. Gradient verification
    8. A small image-patch attention experiment

No pretrained model, checkpoint, or pretrained weight is used.

Everything is randomly initialized and learned/verified directly in this file.


================================================================================
WHAT YOU WILL LEARN
================================================================================

After this lesson you should understand:

    Input tokens
        |
        +----> Query Q
        |
        +----> Key K
        |
        +----> Value V
        |
        v
    Q K^T
        |
        v
    Scale by sqrt(d_k)
        |
        v
    Softmax
        |
        v
    Attention weights
        |
        v
    Weighted sum of V
        |
        v
    Attention output


The central equation is:

    Attention(Q, K, V)
        = softmax(Q K^T / sqrt(d_k)) V


You will also understand why attention is important for computer vision:

    CNN:
        mostly local interaction

    Self-attention:
        potentially global interaction

For example, an image patch containing one part of an object can directly
attend to another distant patch containing another part of the same object.


================================================================================
COURSE ROADMAP
================================================================================

Phase 0  : Mathematical foundations
Phase 1  : Image representation and tensors
Phase 2  : Neural networks and backpropagation
Phase 3  : CNNs
Phase 4  : Autoencoders / VAE
Phase 5  : GANs
Phase 6  : Attention / Transformers / ViT       <-- CURRENT
Phase 7  : Generative modeling fundamentals
Phase 8  : Diffusion from scratch
Phase 9  : Latent diffusion
Phase 10 : Conditioning / cross-attention
Phase 11 : Own text-to-image model
Phase 12 : Advanced diffusion
Phase 13 : Scaling and training infrastructure


Current progression:

    Lesson 11
        GAN
          |
          v
    Lesson 12
        Self-attention                  <-- YOU ARE HERE
          |
          v
    Lesson 13
        Transformer / Vision Transformer
          |
          v
    Lesson 14+
        Generative modeling
          |
          v
        Diffusion
          |
          v
        U-Net / DDPM
          |
          v
        Latent diffusion
          |
          v
        Cross-attention
          |
          v
        Text-conditioned image generation


================================================================================
REAL-WORLD PROBLEM
================================================================================

Consider a robot looking at a table.

The image contains:

    [robot hand] [cup] [table] [background] [other objects]


A CNN processes local neighborhoods.

A deep CNN can eventually obtain a large receptive field, but information
must travel through multiple layers.

Self-attention provides a different mechanism.

Suppose one image patch contains:

    "cup handle"

Another distant patch contains:

    "cup body"

The attention mechanism can learn:

    handle patch
        |
        +--------------------+
                             |
                             v
                         body patch

This gives the model a direct mechanism for modeling relationships between
distant image regions.


================================================================================
1. TOKENS
================================================================================

A Transformer does not directly operate on an image in H x W x C form.

Instead, the image is converted into tokens.

For example:

    Image: 32 x 32

Split into:

    8 x 8 patches

Then:

    32 / 8 = 4

So we obtain:

    4 x 4 = 16 image patches

Each patch becomes one token.

Therefore:

    image
       |
       v
    16 patches
       |
       v
    16 tokens


If each token is represented by 64 numbers:

    X shape = [batch, tokens, embedding_dim]

    X shape = [B, 16, 64]


================================================================================
2. QUERY / KEY / VALUE
================================================================================

Given input:

    X

we calculate:

    Q = X W_Q
    K = X W_K
    V = X W_V


Interpretation:

    Query:
        "What information am I looking for?"

    Key:
        "What information do I contain?"

    Value:
        "What information should I provide?"


The model compares queries against keys.

For tokens i and j:

    similarity(i,j) = Q_i dot K_j


Large similarity means:

    token i considers token j important.


================================================================================
3. SCALED DOT-PRODUCT ATTENTION
================================================================================

Raw attention score:

    S = Q K^T


If:

    Q shape = [B, N, D]

    K shape = [B, N, D]

then:

    K^T shape = [B, D, N]

and:

    Q K^T

has shape:

    [B, N, N]


Therefore every token can compare itself against every token.

The score is scaled:

    S_scaled = Q K^T / sqrt(D)


Why?

Because the magnitude of dot products grows with dimensionality.

Without scaling, softmax can become too sharp and gradients can become
unnecessarily small.

Then:

    A = softmax(S_scaled)


where A is the attention matrix.

Finally:

    output = A V


Shape:

    A       = [B, N, N]

    V       = [B, N, D]

    output  = [B, N, D]


================================================================================
4. ATTENTION MATRIX
================================================================================

The attention matrix is extremely important.

For N tokens:

                KEY
             0   1   2   3
          +-----------------
        0 | .   .   .   .
        1 | .   .   .   .
QUERY   2 | .   .   .   .
        3 | .   .   .   .


Each row represents one query token.

For example:

    A[0, 2] = 0.65

means:

    token 0 gives approximately 65% attention weight to token 2.


Every row should sum to approximately:

    1.0


================================================================================
5. NUMERICAL STABILITY
================================================================================

Softmax:

    softmax(x_i) = exp(x_i) / sum_j exp(x_j)


Directly computing exp(x) can overflow for large x.

Instead:

    softmax(x)
        =
    exp(x - max(x))
        /
    sum(exp(x - max(x)))


Subtracting the maximum does not change the result mathematically, but
dramatically improves numerical stability.


================================================================================
6. MASKING
================================================================================

Attention can optionally prevent certain tokens from attending to others.

For example:

    causal attention


Token 0 can see:

    token 0

Token 1 can see:

    token 0, token 1

Token 2 can see:

    token 0, token 1, token 2


Mask:

    [1, 0, 0]
    [1, 1, 0]
    [1, 1, 1]


In practice, invalid positions receive:

    -infinity

before softmax.

Then:

    exp(-infinity) = 0

so those positions receive zero attention.


For vision Transformers, we often use non-causal attention, where all image
tokens can interact with all other image tokens.


================================================================================
7. MULTI-HEAD SELF-ATTENTION
================================================================================

One attention operation provides one type of relationship.

Multi-head attention allows multiple relationship spaces.

For example:

    Head 1:
        local spatial relationship

    Head 2:
        object-part relationship

    Head 3:
        long-range relationship

    Head 4:
        texture relationship


If:

    embedding dimension = 64
    number of heads = 4

then:

    head dimension = 64 / 4
                  = 16


Each head performs:

    Attention(Q_h, K_h, V_h)


Then:

    concatenate(head_1, head_2, ..., head_H)

followed by:

    output projection


================================================================================
8. COMPUTATIONAL COMPLEXITY
================================================================================

For N tokens:

    attention matrix = N x N


Therefore attention has approximately:

    O(N^2)

token interaction complexity.

This is one of the major challenges in Vision Transformers.

Example:

    16 tokens:
        16^2 = 256 interactions

    256 tokens:
        256^2 = 65,536 interactions

    1024 tokens:
        1024^2 = 1,048,576 interactions


This is why image tokenization and efficient attention mechanisms are
important for large vision systems.


================================================================================
9. IMPLEMENTATION
================================================================================

We use PyTorch for tensor operations and automatic differentiation.

Important:

    No pretrained model is used.

    No pretrained weights are used.

    All projection layers start from random initialization.


================================================================================
10. REAL-WORLD COMPUTER VISION CONNECTION
================================================================================

A Vision Transformer approximately follows:

    Image
      |
      v
    Patch extraction
      |
      v
    Patch embeddings
      |
      v
    Positional information
      |
      v
    Transformer blocks
      |
      v
    Vision representation
      |
      v
    Classification / detection / segmentation / generation


Today's lesson implements the core mechanism inside the Transformer block.

Lesson 13 will assemble these ideas into a Transformer / Vision Transformer.


================================================================================
11. EXPERIMENT
================================================================================

We create a synthetic image containing several spatial structures.

The image is divided into patches.

Then we construct patch embeddings and run self-attention.

We visualize:

    1. Original image
    2. Patch grid
    3. Attention matrix
    4. Attention from selected query patches


This lets us directly observe the relationship between image patches.


================================================================================
12. VERIFICATION / TESTS
================================================================================

The script verifies:

    - tensor shapes
    - attention rows sum to one
    - output is finite
    - softmax numerical stability
    - causal mask behavior
    - multi-head shape
    - gradients exist
    - gradients are finite
    - numerical gradient agreement


================================================================================
13. REAL-WORLD RESULT
================================================================================

After this lesson, we have a working implementation of the central
relationship mechanism used by modern Transformer-based vision systems.

The important conceptual transition is:

    CNN:
        "look around nearby pixels"

to:

    Self-attention:
        "decide which tokens are relevant to this token"


This mechanism is fundamental to:

    Vision Transformers
    multimodal models
    image generation
    diffusion Transformers
    video Transformers
    robot perception
    VLA models
    language-conditioned vision


================================================================================
14. LESSON RESULT
================================================================================

You should now be able to explain:

    What Q, K and V mean
    Why QK^T is computed
    Why sqrt(d_k) is used
    Why softmax is required
    What an attention matrix means
    What masking does
    Why multiple heads are useful
    Why attention costs O(N^2)
    Why attention is useful for computer vision


================================================================================
15. CONNECTION TO NEXT LESSON
================================================================================

Next:

    LESSON 13 — TRANSFORMER AND VISION TRANSFORMER FROM SCRATCH


We will combine:

    Patch embedding
    Positional embedding
    Multi-head self-attention
    Layer normalization
    Residual connections
    Feed-forward network
    Transformer blocks
    Classification head


The resulting architecture will be a small Vision Transformer trained from
scratch on a synthetic image classification problem.

After that, the course will move toward:

    generative modeling
        ->
    diffusion
        ->
    DDPM
        ->
    U-Net
        ->
    latent diffusion
        ->
    cross-attention
        ->
    text-conditioned generation


================================================================================
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 32
PATCH_SIZE = 8
CHANNELS = 1

NUM_PATCHES_PER_SIDE = IMAGE_SIZE // PATCH_SIZE
NUM_TOKENS = NUM_PATCHES_PER_SIDE ** 2

EMBED_DIM = 64
NUM_HEADS = 4

BATCH_SIZE = 4
SEQ_LEN = NUM_TOKENS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# NUMERICALLY STABLE SOFTMAX
# =============================================================================

def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax.

    softmax(x) =
        exp(x - max(x))
        ----------------
        sum(exp(x - max(x)))
    """

    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(shifted)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# =============================================================================
# SINGLE-HEAD SCALED DOT-PRODUCT ATTENTION
# =============================================================================

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    """
    Compute:

        Attention(Q,K,V)
            = softmax(QK^T / sqrt(d_k)) V

    Expected shapes:

        q: [B, N, D]
        k: [B, N, D]
        v: [B, N, D]

    Output:

        [B, N, D]

    Attention matrix:

        [B, N, N]
    """

    d_k = q.shape[-1]

    scores = torch.matmul(q, k.transpose(-2, -1))

    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention = stable_softmax(scores, dim=-1)

    output = torch.matmul(attention, v)

    return output, attention


# =============================================================================
# MULTI-HEAD SELF-ATTENTION
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.

    Input:

        x = [B, N, D]

    Projection:

        Q = XW_Q
        K = XW_K
        V = XW_V

    Split:

        D -> H heads

    Each head:

        Attention(Q_h, K_h, V_h)

    Concatenate heads and apply output projection.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, N, D]

        becomes:

        [B, H, N, D_head]
        """

        batch_size, num_tokens, _ = x.shape

        x = x.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2)

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, H, N, D_head]

        becomes:

        [B, N, D]
        """

        batch_size, _, num_tokens, _ = x.shape

        x = x.transpose(1, 2)

        x = x.reshape(
            batch_size,
            num_tokens,
            self.embed_dim,
        )

        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        output, attention = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
        )

        output = self._merge_heads(output)

        output = self.out_proj(output)

        return output, attention


# =============================================================================
# PATCH EXTRACTION
# =============================================================================

def image_to_patches(
    images: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Convert:

        [B, C, H, W]

    into:

        [B, N, patch_dim]

    where:

        N = number of patches
        patch_dim = C * patch_size * patch_size
    """

    batch_size, channels, height, width = images.shape

    assert height % patch_size == 0
    assert width % patch_size == 0

    patches_per_side_h = height // patch_size
    patches_per_side_w = width // patch_size

    patches = images.unfold(
        dimension=2,
        size=patch_size,
        step=patch_size,
    )

    patches = patches.unfold(
        dimension=3,
        size=patch_size,
        step=patch_size,
    )

    # [B, C, H_p, W_p, P, P]

    patches = patches.permute(
        0,
        2,
        3,
        1,
        4,
        5,
    )

    # [B, H_p, W_p, C, P, P]

    patches = patches.contiguous()

    patches = patches.reshape(
        batch_size,
        patches_per_side_h * patches_per_side_w,
        channels * patch_size * patch_size,
    )

    return patches


# =============================================================================
# PATCH EMBEDDING
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert image patches into Transformer token embeddings.

        image
          |
          v
        patches
          |
          v
        linear projection
          |
          v
        tokens
    """

    def __init__(
        self,
        channels: int,
        patch_size: int,
        embed_dim: int,
    ):
        super().__init__()

        patch_dim = channels * patch_size * patch_size

        self.projection = nn.Linear(
            patch_dim,
            embed_dim,
        )

    def forward(self, images: torch.Tensor):
        patches = image_to_patches(
            images,
            patch_size=PATCH_SIZE,
        )

        tokens = self.projection(patches)

        return tokens


# =============================================================================
# POSITIONAL EMBEDDING
# =============================================================================

def sinusoidal_position_embedding(
    num_tokens: int,
    embed_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding.

    For position p and dimension i:

        PE(p, 2i)
            = sin(p / 10000^(2i/D))

        PE(p, 2i+1)
            = cos(p / 10000^(2i/D))

    Shape:

        [1, num_tokens, embed_dim]
    """

    position = torch.arange(
        num_tokens,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1)

    dimension = torch.arange(
        embed_dim,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    div_term = torch.exp(
        -(dimension // 2)
        * math.log(10000.0)
        / embed_dim
    )

    angles = position * div_term

    pe = torch.zeros(
        num_tokens,
        embed_dim,
        device=device,
    )

    pe[:, 0::2] = torch.sin(
        angles[:, 0::2]
    )

    pe[:, 1::2] = torch.cos(
        angles[:, 1::2]
    )

    return pe.unsqueeze(0)


# =============================================================================
# SYNTHETIC IMAGE GENERATION
# =============================================================================

def create_attention_image(
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Create a synthetic image with several spatial structures.

    The image intentionally contains:

        - horizontal structure
        - vertical structure
        - diagonal structure
        - bright object

    This gives the attention mechanism spatial patterns to process.
    """

    image = np.zeros(
        (image_size, image_size),
        dtype=np.float32,
    )

    # Horizontal structure.
    image[8:12, 4:28] = 0.8

    # Vertical structure.
    image[4:28, 20:24] = 0.9

    # Diagonal structure.
    for i in range(5, 25):
        j = i

        if 0 <= j < image_size:
            image[i, j] = 1.0

        if 0 <= j + 1 < image_size:
            image[i, j + 1] = 1.0

    # Bright square object.
    image[22:28, 5:11] = 0.7

    # Small amount of noise.
    noise = np.random.normal(
        0.0,
        0.03,
        size=image.shape,
    ).astype(np.float32)

    image += noise

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return image


# =============================================================================
# ATTENTION MASK
# =============================================================================

def create_causal_mask(
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create lower-triangular causal mask.

    Example for N=4:

        True  False False False
        True  True  False False
        True  True  True  False
        True  True  True  True
    """

    return torch.tril(
        torch.ones(
            num_tokens,
            num_tokens,
            dtype=torch.bool,
            device=device,
        )
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_input_and_patches(
    image: np.ndarray,
):
    """
    Visualize the image and its patch grid.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title("Synthetic Image")
    axes[0].axis("off")

    axes[1].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    for x in range(
        0,
        IMAGE_SIZE + 1,
        PATCH_SIZE,
    ):
        axes[1].axvline(
            x - 0.5,
            linewidth=1,
        )

    for y in range(
        0,
        IMAGE_SIZE + 1,
        PATCH_SIZE,
    ):
        axes[1].axhline(
            y - 0.5,
            linewidth=1,
        )

    axes[1].set_title(
        f"Patch Grid ({NUM_TOKENS} tokens)"
    )

    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()


def visualize_attention_matrix(
    attention: torch.Tensor,
    head_index: int = 0,
):
    """
    Visualize one attention matrix.

    attention:

        [B, H, N, N]
    """

    matrix = attention[
        0,
        head_index,
    ].detach().cpu().numpy()

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        matrix,
        cmap="viridis",
        aspect="auto",
    )

    plt.colorbar(
        label="Attention weight"
    )

    plt.xlabel("Key token")
    plt.ylabel("Query token")

    plt.title(
        f"Self-Attention Matrix — Head {head_index}"
    )

    plt.tight_layout()
    plt.show()


def visualize_query_attention(
    image: np.ndarray,
    attention: torch.Tensor,
    query_index: int,
    head_index: int = 0,
):
    """
    Show which image patches a selected query token attends to.

    The attention vector:

        attention[query_index, :]

    is reshaped back into the spatial patch grid.
    """

    weights = attention[
        0,
        head_index,
        query_index,
    ].detach().cpu().numpy()

    weights = weights.reshape(
        NUM_PATCHES_PER_SIDE,
        NUM_PATCHES_PER_SIDE,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    query_row = query_index // NUM_PATCHES_PER_SIDE
    query_col = query_index % NUM_PATCHES_PER_SIDE

    rect_x = query_col * PATCH_SIZE
    rect_y = query_row * PATCH_SIZE

    rectangle = plt.Rectangle(
        (rect_x - 0.5, rect_y - 0.5),
        PATCH_SIZE,
        PATCH_SIZE,
        fill=False,
        linewidth=3,
    )

    axes[0].add_patch(rectangle)

    axes[0].set_title(
        f"Query Token {query_index}"
    )

    axes[0].axis("off")

    axes[1].imshow(
        weights,
        cmap="viridis",
        interpolation="nearest",
    )

    axes[1].set_title(
        "Where This Token Attends"
    )

    axes[1].set_xlabel("Patch column")
    axes[1].set_ylabel("Patch row")

    plt.tight_layout()
    plt.show()


def visualize_multihead_attention(
    attention: torch.Tensor,
):
    """
    Display attention matrices from all heads.
    """

    num_heads = attention.shape[1]

    fig, axes = plt.subplots(
        1,
        num_heads,
        figsize=(4 * num_heads, 4),
    )

    if num_heads == 1:
        axes = [axes]

    for head_index in range(num_heads):

        matrix = attention[
            0,
            head_index,
        ].detach().cpu().numpy()

        axes[head_index].imshow(
            matrix,
            cmap="viridis",
            aspect="auto",
        )

        axes[head_index].set_title(
            f"Head {head_index}"
        )

        axes[head_index].set_xlabel(
            "Key"
        )

        axes[head_index].set_ylabel(
            "Query"
        )

    plt.tight_layout()
    plt.show()


# =============================================================================
# TEST 1 — STABLE SOFTMAX
# =============================================================================

def test_stable_softmax():

    print("\nTEST 1 — Stable Softmax")

    x = torch.tensor(
        [
            [1000.0, 1001.0, 1002.0],
            [-1000.0, -999.0, -998.0],
        ]
    )

    probabilities = stable_softmax(
        x,
        dim=-1,
    )

    assert torch.isfinite(
        probabilities
    ).all()

    row_sums = probabilities.sum(
        dim=-1
    )

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )

    print(
        "PASS: softmax remains finite "
        "for extreme values."
    )


# =============================================================================
# TEST 2 — PATCH EXTRACTION
# =============================================================================

def test_patch_extraction():

    print("\nTEST 2 — Patch Extraction")

    images = torch.randn(
        2,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    patches = image_to_patches(
        images,
        PATCH_SIZE,
    )

    expected_shape = (
        2,
        NUM_TOKENS,
        CHANNELS * PATCH_SIZE * PATCH_SIZE,
    )

    assert patches.shape == expected_shape

    print(
        "PASS: patch extraction shape =",
        tuple(patches.shape),
    )


# =============================================================================
# TEST 3 — ATTENTION ROW SUM
# =============================================================================

def test_attention_rows():

    print("\nTEST 3 — Attention Rows")

    q = torch.randn(
        BATCH_SIZE,
        SEQ_LEN,
        EMBED_DIM,
    )

    k = torch.randn(
        BATCH_SIZE,
        SEQ_LEN,
        EMBED_DIM,
    )

    v = torch.randn(
        BATCH_SIZE,
        SEQ_LEN,
        EMBED_DIM,
    )

    output, attention = (
        scaled_dot_product_attention(
            q,
            k,
            v,
        )
    )

    assert output.shape == (
        BATCH_SIZE,
        SEQ_LEN,
        EMBED_DIM,
    )

    assert attention.shape == (
        BATCH_SIZE,
        SEQ_LEN,
        SEQ_LEN,
    )

    row_sums = attention.sum(
        dim=-1
    )

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )

    assert torch.isfinite(
        output
    ).all()

    print(
        "PASS: attention output shape and "
        "row normalization are correct."
    )


# =============================================================================
# TEST 4 — CAUSAL MASK
# =============================================================================

def test_causal_mask():

    print("\nTEST 4 — Causal Mask")

    q = torch.randn(
        1,
        5,
        8,
    )

    k = torch.randn(
        1,
        5,
        8,
    )

    v = torch.randn(
        1,
        5,
        8,
    )

    mask = create_causal_mask(
        5,
        q.device,
    )

    _, attention = (
        scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
        )
    )

    upper_triangle = torch.triu(
        attention,
        diagonal=1,
    )

    assert torch.allclose(
        upper_triangle,
        torch.zeros_like(
            upper_triangle
        ),
        atol=1e-7,
    )

    print(
        "PASS: causal mask blocks future tokens."
    )


# =============================================================================
# TEST 5 — MULTI-HEAD SHAPES
# =============================================================================

def test_multihead_shapes():

    print("\nTEST 5 — Multi-Head Attention Shapes")

    model = MultiHeadSelfAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
    )

    x = torch.randn(
        BATCH_SIZE,
        SEQ_LEN,
        EMBED_DIM,
    )

    output, attention = model(x)

    assert output.shape == (
        BATCH_SIZE,
        SEQ_LEN,
        EMBED_DIM,
    )

    assert attention.shape == (
        BATCH_SIZE,
        NUM_HEADS,
        SEQ_LEN,
        SEQ_LEN,
    )

    assert torch.isfinite(
        output
    ).all()

    print(
        "PASS: multi-head shapes are correct."
    )


# =============================================================================
# TEST 6 — GRADIENT FLOW
# =============================================================================

def test_gradient_flow():

    print("\nTEST 6 — Gradient Flow")

    model = MultiHeadSelfAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
    )

    x = torch.randn(
        2,
        SEQ_LEN,
        EMBED_DIM,
        requires_grad=True,
    )

    output, _ = model(x)

    loss = output.pow(2).mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()

    parameter_with_gradient = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            assert parameter.grad is not None

            assert torch.isfinite(
                parameter.grad
            ).all()

            parameter_with_gradient += 1

    assert parameter_with_gradient > 0

    print(
        "PASS: gradients flow through "
        "the complete attention module."
    )


# =============================================================================
# TEST 7 — NUMERICAL GRADIENT CHECK
# =============================================================================

def test_numerical_gradient():

    print("\nTEST 7 — Numerical Gradient Check")

    torch.manual_seed(123)

    q = torch.randn(
        1,
        3,
        4,
        dtype=torch.float64,
        requires_grad=True,
    )

    k = torch.randn(
        1,
        3,
        4,
        dtype=torch.float64,
        requires_grad=True,
    )

    v = torch.randn(
        1,
        3,
        4,
        dtype=torch.float64,
        requires_grad=True,
    )

    output, _ = scaled_dot_product_attention(
        q,
        k,
        v,
    )

    loss = output.sum()

    loss.backward()

    analytical = q.grad[
        0,
        0,
        0,
    ].item()

    epsilon = 1e-5

    q_plus = q.detach().clone()
    q_minus = q.detach().clone()

    q_plus[0, 0, 0] += epsilon
    q_minus[0, 0, 0] -= epsilon

    output_plus, _ = (
        scaled_dot_product_attention(
            q_plus,
            k.detach(),
            v.detach(),
        )
    )

    output_minus, _ = (
        scaled_dot_product_attention(
            q_minus,
            k.detach(),
            v.detach(),
        )
    )

    loss_plus = output_plus.sum()
    loss_minus = output_minus.sum()

    numerical = (
        (loss_plus - loss_minus)
        / (2.0 * epsilon)
    ).item()

    absolute_error = abs(
        analytical - numerical
    )

    print(
        f"Analytical gradient : {analytical:.8f}"
    )

    print(
        f"Numerical gradient  : {numerical:.8f}"
    )

    print(
        f"Absolute error      : {absolute_error:.8e}"
    )

    assert absolute_error < 1e-5

    print(
        "PASS: analytical and numerical "
        "gradients agree."
    )


# =============================================================================
# TEST 8 — POSITIONAL EMBEDDING
# =============================================================================

def test_positional_embedding():

    print("\nTEST 8 — Positional Embedding")

    pe = sinusoidal_position_embedding(
        NUM_TOKENS,
        EMBED_DIM,
        DEVICE,
    )

    assert pe.shape == (
        1,
        NUM_TOKENS,
        EMBED_DIM,
    )

    assert torch.isfinite(pe).all()

    print(
        "PASS: positional embedding shape =",
        tuple(pe.shape),
    )


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def count_parameters(model: nn.Module) -> int:

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =============================================================================
# ATTENTION STATISTICS
# =============================================================================

def print_attention_statistics(
    attention: torch.Tensor,
):

    print("\nATTENTION STATISTICS")

    print(
        "Shape:",
        tuple(attention.shape),
    )

    print(
        "Minimum:",
        attention.min().item(),
    )

    print(
        "Maximum:",
        attention.max().item(),
    )

    print(
        "Mean:",
        attention.mean().item(),
    )

    entropy = -(
        attention
        * torch.log(
            attention.clamp_min(1e-12)
        )
    ).sum(dim=-1)

    print(
        "Mean attention entropy:",
        entropy.mean().item(),
    )


# =============================================================================
# COMPARE ATTENTION WITH AND WITHOUT SCALING
# =============================================================================

def scaling_experiment():

    """
    Demonstrate why sqrt(d_k) scaling is useful.

    Larger dimensions create larger dot products.

    We compare:

        QK^T

    against:

        QK^T / sqrt(d_k)
    """

    print("\nSCALING EXPERIMENT")

    torch.manual_seed(10)

    dimensions = [
        8,
        32,
        128,
    ]

    for d in dimensions:

        q = torch.randn(
            1,
            16,
            d,
        )

        k = torch.randn(
            1,
            16,
            d,
        )

        raw_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scaled_scores = (
            raw_scores
            / math.sqrt(d)
        )

        raw_std = raw_scores.std().item()
        scaled_std = scaled_scores.std().item()

        print(
            f"d_k={d:3d} | "
            f"raw std={raw_std:8.4f} | "
            f"scaled std={scaled_std:8.4f}"
        )


# =============================================================================
# REAL-WORLD PATCH ATTENTION DEMO
# =============================================================================

def run_image_attention_demo():

    print("\n" + "=" * 80)
    print("IMAGE PATCH SELF-ATTENTION DEMO")
    print("=" * 80)

    image = create_attention_image()

    visualize_input_and_patches(
        image
    )

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    patch_embedding = PatchEmbedding(
        channels=CHANNELS,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
    ).to(DEVICE)

    tokens = patch_embedding(
        image_tensor
    )

    position = sinusoidal_position_embedding(
        NUM_TOKENS,
        EMBED_DIM,
        DEVICE,
    )

    tokens = tokens + position

    print(
        "Image tensor:",
        tuple(image_tensor.shape),
    )

    print(
        "Patch tokens:",
        tuple(tokens.shape),
    )

    attention_model = MultiHeadSelfAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
    ).to(DEVICE)

    output, attention = attention_model(
        tokens
    )

    print(
        "Attention output:",
        tuple(output.shape),
    )

    print(
        "Attention tensor:",
        tuple(attention.shape),
    )

    assert output.shape == (
        1,
        NUM_TOKENS,
        EMBED_DIM,
    )

    assert attention.shape == (
        1,
        NUM_HEADS,
        NUM_TOKENS,
        NUM_TOKENS,
    )

    print_attention_statistics(
        attention
    )

    visualize_multihead_attention(
        attention
    )

    visualize_attention_matrix(
        attention,
        head_index=0,
    )

    # Select a patch containing a strong structure.
    query_index = 5

    visualize_query_attention(
        image=image,
        attention=attention,
        query_index=query_index,
        head_index=0,
    )

    return (
        image,
        tokens,
        output,
        attention,
    )


# =============================================================================
# REAL-WORLD INTERPRETATION
# =============================================================================

def print_real_world_interpretation():

    print(
        """
================================================================================
REAL-WORLD INTERPRETATION
================================================================================

The experiment above is intentionally small.

The important result is architectural.

An image was converted into:

    pixels
       |
       v
    patches
       |
       v
    tokens
       |
       v
    Q / K / V
       |
       v
    attention relationships
       |
       v
    contextualized tokens


In a real robot vision system this can become:

    camera
      |
      v
    image
      |
      v
    visual tokens
      |
      v
    Transformer
      |
      v
    object / scene representation
      |
      +--------------------+
      |                    |
      v                    v
    perception          manipulation
                            |
                            v
                         action


For example, a robot may need to understand:

    "this gripper is near the handle of that cup"

rather than treating every image region independently.

Attention provides a mechanism for representing such relationships.

IMPORTANT:

The model in this lesson is NOT trained to solve cup picking.

It demonstrates the computational mechanism that modern Transformer-based
vision systems use to build contextual representations.


================================================================================
"""
    )


# =============================================================================
# LESSON SUMMARY
# =============================================================================

def print_lesson_summary():

    print(
        """
================================================================================
LESSON 12 RESULT
================================================================================

We implemented from scratch:

    [PASS] Numerically stable softmax
    [PASS] Scaled dot-product attention
    [PASS] Query / Key / Value mechanism
    [PASS] Attention masks
    [PASS] Causal masking
    [PASS] Patch extraction
    [PASS] Patch embedding
    [PASS] Positional encoding
    [PASS] Multi-head self-attention
    [PASS] Gradient verification
    [PASS] Attention visualization
    [PASS] Image-patch attention experiment

The central equation is:

    Attention(Q,K,V)
        = softmax(QK^T / sqrt(d_k))V


NEXT LESSON
-----------

LESSON 13 — TRANSFORMER AND VISION TRANSFORMER FROM SCRATCH

We will combine:

    patch embedding
    positional encoding
    multi-head attention
    residual connections
    layer normalization
    feed-forward networks
    Transformer blocks
    classification head

and train a complete small Vision Transformer from random initialization.

After that:

    ViT
     |
     v
    Generative modeling
     |
     v
    Diffusion
     |
     v
    DDPM
     |
     v
    U-Net
     |
     v
    Latent diffusion
     |
     v
    Cross-attention
     |
     v
    Text-to-image generation


The important conceptual transition is:

    Lesson 5-8:
        convolution learns local spatial features

    Lesson 12:
        attention learns token-to-token relationships

    Lesson 13:
        Transformer combines attention + residual learning +
        normalization + feed-forward processing into a complete
        vision architecture.


================================================================================
"""
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    print("=" * 80)
    print("COMPUTER VISION ADVANCED CRASH COURSE")
    print("LESSON 12 — SELF-ATTENTION FROM SCRATCH")
    print("=" * 80)

    print(
        "\nDevice:",
        DEVICE,
    )

    print(
        "Image size:",
        IMAGE_SIZE,
        "x",
        IMAGE_SIZE,
    )

    print(
        "Patch size:",
        PATCH_SIZE,
        "x",
        PATCH_SIZE,
    )

    print(
        "Number of tokens:",
        NUM_TOKENS,
    )

    print(
        "Embedding dimension:",
        EMBED_DIM,
    )

    print(
        "Attention heads:",
        NUM_HEADS,
    )

    # -------------------------------------------------------------------------
    # UNIT TESTS
    # -------------------------------------------------------------------------

    test_stable_softmax()

    test_patch_extraction()

    test_attention_rows()

    test_causal_mask()

    test_multihead_shapes()

    test_gradient_flow()

    test_numerical_gradient()

    test_positional_embedding()

    # -------------------------------------------------------------------------
    # EXPERIMENT
    # -------------------------------------------------------------------------

    scaling_experiment()

    # -------------------------------------------------------------------------
    # IMAGE ATTENTION DEMO
    # -------------------------------------------------------------------------

    (
        image,
        tokens,
        output,
        attention,
    ) = run_image_attention_demo()

    # -------------------------------------------------------------------------
    # MODEL INFORMATION
    # -------------------------------------------------------------------------

    model = MultiHeadSelfAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
    )

    print(
        "\nMulti-head attention parameters:",
        count_parameters(model),
    )

    # -------------------------------------------------------------------------
    # FINAL NUMERICAL CHECKS
    # -------------------------------------------------------------------------

    print(
        "\nFINAL NUMERICAL CHECKS"
    )

    assert torch.isfinite(
        tokens
    ).all()

    assert torch.isfinite(
        output
    ).all()

    assert torch.isfinite(
        attention
    ).all()

    row_sums = attention.sum(
        dim=-1
    )

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )

    print(
        "PASS: all final tensors are finite."
    )

    print(
        "PASS: every attention row sums to 1."
    )

    # -------------------------------------------------------------------------
    # REAL-WORLD APPLICATION
    # -------------------------------------------------------------------------

    print_real_world_interpretation()

    # -------------------------------------------------------------------------
    # LESSON RESULT
    # -------------------------------------------------------------------------

    print_lesson_summary()


if __name__ == "__main__":
    main()