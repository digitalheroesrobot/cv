"""
Filename: lesson08_resnet_residual_learning.py

================================================================================
LESSON 8 — DEEP CNNs AND RESIDUAL LEARNING
================================================================================

PURPOSE
-------
This lesson explains why simply making CNNs deeper can become difficult to
optimize, and how residual connections solve much of that problem.

We will build and train:

    1. A plain deeper CNN
    2. A residual CNN using skip connections

Then we compare:

    - training behavior
    - validation accuracy
    - gradient flow
    - parameter counts
    - learned representations

The central idea is the residual transformation:

    y = F(x) + x

instead of learning:

    y = H(x)

The network learns the residual:

    F(x) = H(x) - x

This idea is the foundation of ResNet-style vision architectures and is
extremely important before moving toward modern computer vision systems.

IMPORTANT COURSE RULE
---------------------
This lesson uses PyTorch, but the model starts with RANDOM initialization.

No pretrained weights are required.

================================================================================
WHAT YOU WILL LEARN
================================================================================

1. Why deeper CNNs are useful.
2. Why deeper networks can become difficult to optimize.
3. Vanishing/exploding gradient intuition.
4. Residual learning.
5. Skip connections.
6. Residual blocks.
7. Projection shortcuts when tensor dimensions change.
8. Building a ResNet-style network.
9. Comparing plain CNN vs residual CNN.
10. Measuring gradient flow.
11. Understanding why ResNet became an important CNN architecture.
12. Connecting residual learning to real-world robot vision.

================================================================================
TECHNICAL / COURSE ROADMAP
================================================================================

LESSON 1
    Images as numerical data

LESSON 2
    Vectors, matrices, tensors

LESSON 3
    Gradient descent

LESSON 4
    Neural networks and backpropagation

LESSON 5
    Convolution from scratch

LESSON 6
    CNN from scratch

LESSON 7
    Practical PyTorch CNN training

LESSON 8  <-- CURRENT
    Deep CNNs + residual learning

LESSON 9
    Representation learning / autoencoders

LESSON 10
    Variational autoencoders

LESSON 11+
    Generative modeling
    GANs
    Attention
    Transformers
    Diffusion
    Text conditioning
    Own text-to-image model

================================================================================
REAL-WORLD PROBLEM
================================================================================

Imagine a robot camera looking at objects on a sorting table.

The robot may need to distinguish:

    plastic bottle
    metal can
    cardboard
    paper
    irregular waste
    human hand
    background

A shallow CNN may not extract enough hierarchical features.

A deeper CNN can learn:

    Layer 1:
        edges

    Layer 2:
        corners / textures

    Layer 3:
        local shapes

    Layer 4:
        object parts

    Layer 5+:
        semantic object structures

However, simply stacking many layers can make optimization harder.

Residual connections allow information and gradients to travel through
shortcut paths.

This is one reason residual networks became extremely important in
real-world computer vision.

================================================================================
THEORY
================================================================================

A normal feed-forward block can be written as:

    y = H(x)

Suppose the desired transformation is:

    H(x)

A residual block instead learns:

    y = F(x) + x

where:

    F(x) = H(x) - x

Therefore:

    H(x) = F(x) + x

The network does not need to learn the entire transformation directly.

It only needs to learn the correction/residual.

This is especially useful when the desired transformation is close to the
identity function.

--------------------------------------------------------------------------------
IDENTITY MAPPING
--------------------------------------------------------------------------------

Suppose:

    H(x) ≈ x

A normal network must learn:

    H(x) ≈ x

A residual network can simply make:

    F(x) ≈ 0

so:

    y = F(x) + x
      ≈ x

This makes identity mappings much easier to represent.

--------------------------------------------------------------------------------
GRADIENT FLOW
--------------------------------------------------------------------------------

For:

    y = F(x) + x

the derivative is:

    dy/dx = dF(x)/dx + I

The identity term:

    I

creates a direct gradient path.

This helps gradients propagate through many layers.

This is one of the key ideas behind residual learning.

--------------------------------------------------------------------------------
DIMENSION MATCHING
--------------------------------------------------------------------------------

The simple shortcut:

    x + F(x)

only works when x and F(x) have the same shape.

Example:

    x:
        [B, 32, 16, 16]

    F(x):
        [B, 32, 16, 16]

can be added directly.

But if we change channels:

    x:
        [B, 32, 16, 16]

    F(x):
        [B, 64, 8, 8]

we cannot add them.

We therefore use a projection shortcut:

    shortcut = Conv2D(32 -> 64, kernel=1, stride=2)

so that:

    shortcut(x):
        [B, 64, 8, 8]

and now:

    y = F(x) + shortcut(x)

================================================================================
MATHEMATICS
================================================================================

Residual block:

    z = Conv1(x)

    z = BatchNorm(z)

    z = ReLU(z)

    z = Conv2(z)

    z = BatchNorm(z)

    y = ReLU(z + shortcut(x))

Shortcut:

    shortcut(x) = x

when dimensions match.

Otherwise:

    shortcut(x) = Conv1x1(x)

The complete residual function is therefore:

    y = ReLU(F(x) + S(x))

where:

    F(x) = learned transformation

and:

    S(x) = identity or projection shortcut.

--------------------------------------------------------------------------------
CROSS ENTROPY
--------------------------------------------------------------------------------

For classification:

    L = -sum_i y_i log(p_i)

where:

    y_i = true class indicator
    p_i = predicted probability

PyTorch's CrossEntropyLoss combines:

    softmax + negative log likelihood

in a numerically stable implementation.

--------------------------------------------------------------------------------
WHY WE USE LOGITS
--------------------------------------------------------------------------------

The final layer outputs:

    logits

not manually normalized probabilities.

For example:

    logits = [2.1, 0.4, -1.0]

CrossEntropyLoss internally performs the appropriate stable operation.

================================================================================
IMPLEMENTATION PLAN
================================================================================

Dataset:

    synthetic 32x32 grayscale images

Classes:

    0 = horizontal structure
    1 = vertical structure
    2 = diagonal structure

Models:

    PlainDeepCNN
    ResidualCNN

The task is intentionally simple.

The purpose is not to build the most powerful classifier.

The purpose is to observe the effect of residual architecture.

================================================================================
"""

import os
import random
import copy
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 32
NUM_CLASSES = 3

TRAIN_SAMPLES = 1800
VAL_SAMPLES = 450
TEST_SAMPLES = 450

BATCH_SIZE = 64

EPOCHS = 25

LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "lesson08_residual_cnn_best.pt"


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


# =============================================================================
# ENVIRONMENT INFORMATION
# =============================================================================

def print_environment():
    """
    Display the computational environment.
    """

    print("=" * 80)
    print("LESSON 8 — DEEP CNN + RESIDUAL LEARNING")
    print("=" * 80)

    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")

    print("=" * 80)


# =============================================================================
# SYNTHETIC COMPUTER-VISION DATASET
# =============================================================================

def create_pattern_image(label, size=IMAGE_SIZE, noise_level=0.12):
    """
    Create a synthetic image.

    Classes:

        0 -> horizontal structure
        1 -> vertical structure
        2 -> diagonal structure

    Small random variations are added so that the CNN must learn a
    representation instead of simply memorizing one exact image.
    """

    image = np.zeros((size, size), dtype=np.float32)

    if label == 0:
        # Horizontal pattern.
        center = np.random.randint(size // 4, 3 * size // 4)
        thickness = np.random.randint(2, 5)

        y0 = max(0, center - thickness)
        y1 = min(size, center + thickness)

        image[y0:y1, :] = 1.0

    elif label == 1:
        # Vertical pattern.
        center = np.random.randint(size // 4, 3 * size // 4)
        thickness = np.random.randint(2, 5)

        x0 = max(0, center - thickness)
        x1 = min(size, center + thickness)

        image[:, x0:x1] = 1.0

    elif label == 2:
        # Diagonal pattern.
        thickness = np.random.randint(1, 3)

        for i in range(size):
            for offset in range(-thickness, thickness + 1):
                j = i + offset

                if 0 <= j < size:
                    image[i, j] = 1.0

    else:
        raise ValueError("Unknown class label.")

    # Random translation.
    shift_y = np.random.randint(-3, 4)
    shift_x = np.random.randint(-3, 4)

    translated = np.zeros_like(image)

    src_y0 = max(0, -shift_y)
    src_y1 = min(size, size - shift_y)

    src_x0 = max(0, -shift_x)
    src_x1 = min(size, size - shift_x)

    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    translated[dst_y0:dst_y1, dst_x0:dst_x1] = \
        image[src_y0:src_y1, src_x0:src_x1]

    image = translated

    # Add Gaussian noise.
    noise = np.random.normal(
        loc=0.0,
        scale=noise_level,
        size=image.shape
    ).astype(np.float32)

    image = image + noise

    image = np.clip(image, 0.0, 1.0)

    return image.astype(np.float32)


def create_dataset(num_samples):
    """
    Generate the complete synthetic dataset.
    """

    images = []
    labels = []

    for i in range(num_samples):
        label = i % NUM_CLASSES

        image = create_pattern_image(label)

        images.append(image)
        labels.append(label)

    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle.
    indices = np.random.permutation(len(images))

    images = images[indices]
    labels = labels[indices]

    return images, labels


# =============================================================================
# DATASET CLASS
# =============================================================================

class PatternDataset(Dataset):
    """
    PyTorch Dataset.

    Converts:

        H x W

    into:

        C x H x W

    because PyTorch Conv2d expects channel-first tensors.
    """

    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float().unsqueeze(1)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# =============================================================================
# RESIDUAL BLOCK
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Basic residual block.

    Main branch:

        Conv
        BatchNorm
        ReLU
        Conv
        BatchNorm

    Shortcut:

        identity

    OR:

        1x1 convolution when dimensions change

    Output:

        ReLU(main + shortcut)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity shortcut when dimensions match.
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()

        else:
            # Projection shortcut.
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = self.relu(out)

        return out


# =============================================================================
# PLAIN DEEP CNN
# =============================================================================

class PlainDeepCNN(nn.Module):
    """
    A deeper CNN without residual connections.

    Architecture:

        1 x 32 x 32
            |
        Conv 16
            |
        Conv 16
            |
        Pool
            |
        Conv 32
            |
        Conv 32
            |
        Pool
            |
        Conv 64
            |
        Conv 64
            |
        Pool
            |
        Linear

    This provides a baseline for comparison.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# RESIDUAL CNN
# =============================================================================

class ResidualCNN(nn.Module):
    """
    ResNet-style CNN.

    Architecture:

        Input
          |
        Conv
          |
        Residual block
        Residual block
          |
        Downsample
          |
        Residual block
        Residual block
          |
        Downsample
          |
        Residual block
        Residual block
          |
        Global Average Pool
          |
        Linear classifier

    The network learns residual transformations instead of relying only on
    sequential transformations.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 16)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64)
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =============================================================================
# VISUALIZE DATASET
# =============================================================================

def visualize_dataset(images, labels):
    """
    Show representative examples from each class.
    """

    fig, axes = plt.subplots(
        1,
        NUM_CLASSES,
        figsize=(9, 3)
    )

    for class_id in range(NUM_CLASSES):

        index = np.where(labels == class_id)[0][0]

        axes[class_id].imshow(
            images[index],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[class_id].set_title(
            f"Class {class_id}"
        )

        axes[class_id].axis("off")

    plt.suptitle(
        "Synthetic Vision Dataset"
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):
    """
    Train for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = logits.argmax(dim=1)

        total_correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total_samples += images.size(0)

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return average_loss, accuracy


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):
    """
    Evaluate without gradient computation.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = logits.argmax(dim=1)

        total_correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total_samples += images.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return (
        average_loss,
        accuracy,
        np.array(all_predictions),
        np.array(all_labels)
    )


# =============================================================================
# TRAIN A MODEL
# =============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    checkpoint_path=None
):
    """
    Complete training loop.

    Returns:

        history
        best_model_state
    """

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_accuracy = -float("inf")

    best_state = None

    for epoch in range(1, epochs + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state = copy.deepcopy(
                model.state_dict()
            )

            if checkpoint_path is not None:
                torch.save(
                    best_state,
                    checkpoint_path
                )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train Acc={train_accuracy:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Acc={val_accuracy:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_state


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def compute_confusion_matrix(
    labels,
    predictions,
    num_classes=NUM_CLASSES
):
    """
    Build a confusion matrix using NumPy only.
    """

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    for true_label, predicted_label in zip(
        labels,
        predictions
    ):
        matrix[true_label, predicted_label] += 1

    return matrix


def plot_confusion_matrix(matrix, title):
    """
    Visualize confusion matrix.
    """

    plt.figure(figsize=(5, 4))

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(title)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(range(NUM_CLASSES))
    plt.yticks(range(NUM_CLASSES))

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.show()


# =============================================================================
# TRAINING CURVES
# =============================================================================

def plot_training_history(
    plain_history,
    residual_history
):
    """
    Compare training behavior.
    """

    epochs_plain = range(
        1,
        len(plain_history["train_loss"]) + 1
    )

    epochs_residual = range(
        1,
        len(residual_history["train_loss"]) + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs_plain,
        plain_history["val_accuracy"],
        label="Plain CNN"
    )

    plt.plot(
        epochs_residual,
        residual_history["val_accuracy"],
        label="Residual CNN"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")

    plt.title(
        "Plain CNN vs Residual CNN"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# =============================================================================
# GRADIENT FLOW ANALYSIS
# =============================================================================

def measure_gradient_flow(model, loader):
    """
    Measure gradient magnitude after one backward pass.

    This is not a complete scientific benchmark.

    It is a teaching diagnostic.

    We examine:

        mean(abs(parameter.grad))

    for trainable parameters.

    Residual networks provide shorter gradient paths because the shortcut
    connection allows gradients to propagate through addition.
    """

    model = model.to(DEVICE)

    model.train()

    criterion = nn.CrossEntropyLoss()

    images, labels = next(iter(loader))

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    model.zero_grad()

    logits = model(images)

    loss = criterion(
        logits,
        labels
    )

    loss.backward()

    results = []

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        gradient_mean = (
            parameter.grad.detach()
            .abs()
            .mean()
            .item()
        )

        results.append(
            (
                name,
                gradient_mean
            )
        )

    return results


def print_gradient_flow(results, title):
    """
    Print selected gradient statistics.
    """

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for name, value in results:

        print(
            f"{name:60s} "
            f"mean|grad|={value:.8e}"
        )


# =============================================================================
# NUMERICAL TESTS
# =============================================================================

def test_residual_identity_behavior():
    """
    Verify the mathematical idea behind an identity shortcut.

    We construct:

        output = residual + identity

    and verify the tensor shape and exact addition.
    """

    x = torch.randn(
        2,
        16,
        8,
        8
    )

    residual = torch.randn_like(x)

    output = residual + x

    assert output.shape == x.shape

    assert torch.allclose(
        output,
        residual + x
    )

    print(
        "[PASS] Residual addition test"
    )


def test_projection_shortcut():
    """
    Verify that a residual block can change tensor dimensions.
    """

    block = ResidualBlock(
        in_channels=16,
        out_channels=32,
        stride=2
    )

    x = torch.randn(
        2,
        16,
        16,
        16
    )

    with torch.no_grad():
        y = block(x)

    expected_shape = (
        2,
        32,
        8,
        8
    )

    assert y.shape == expected_shape

    print(
        "[PASS] Projection shortcut shape test"
    )


def test_model_forward():
    """
    Verify complete model forward passes.
    """

    x = torch.randn(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    plain_model = PlainDeepCNN().to(DEVICE)
    residual_model = ResidualCNN().to(DEVICE)

    with torch.no_grad():

        plain_output = plain_model(x)
        residual_output = residual_model(x)

    assert plain_output.shape == (
        4,
        NUM_CLASSES
    )

    assert residual_output.shape == (
        4,
        NUM_CLASSES
    )

    print(
        "[PASS] Model forward-pass test"
    )


def test_gradient_exists():
    """
    Verify that residual network parameters receive gradients.
    """

    model = ResidualCNN().to(DEVICE)

    x = torch.randn(
        4,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    labels = torch.randint(
        0,
        NUM_CLASSES,
        (4,)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    logits = model(x)

    loss = criterion(
        logits,
        labels
    )

    loss.backward()

    gradients_found = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            assert parameter.grad is not None

            assert torch.isfinite(
                parameter.grad
            ).all()

            gradients_found += 1

    assert gradients_found > 0

    print(
        "[PASS] Gradient existence test"
    )


# =============================================================================
# PREDICTION VISUALIZATION
# =============================================================================

@torch.no_grad()
def visualize_predictions(
    model,
    dataset,
    num_examples=12
):
    """
    Display predictions made by the residual CNN.
    """

    model.eval()

    indices = np.random.choice(
        len(dataset),
        size=num_examples,
        replace=False
    )

    rows = 3
    cols = 4

    plt.figure(
        figsize=(10, 7)
    )

    for plot_index, dataset_index in enumerate(indices):

        image, label = dataset[dataset_index]

        x = image.unsqueeze(0).to(DEVICE)

        logits = model(x)

        prediction = logits.argmax(
            dim=1
        ).item()

        plt.subplot(
            rows,
            cols,
            plot_index + 1
        )

        plt.imshow(
            image.squeeze(0).numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        plt.title(
            f"True={label}, Pred={prediction}"
        )

        plt.axis("off")

    plt.suptitle(
        "Residual CNN Predictions"
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# REAL-WORLD INTERPRETATION
# =============================================================================

def real_world_application(
    test_accuracy,
    parameter_count
):
    """
    Explain what the experiment means for a real robot-vision system.
    """

    print()
    print("=" * 80)
    print("REAL-WORLD APPLICATION")
    print("=" * 80)

    print(
        """
A real robot camera may need to classify or recognize objects under
different lighting, positions, textures, and viewpoints.

A deeper CNN provides a hierarchy of visual features:

    pixels
      |
    edges
      |
    textures
      |
    shapes
      |
    object parts
      |
    object representation

Residual connections make deeper feature extractors easier to optimize.

For a waste-sorting robot, this can support:

    camera
       |
       v
    CNN feature extractor
       |
       +--> object classification
       |
       +--> object detection
       |
       +--> segmentation
       |
       +--> grasping policy input

The same principle is relevant to robotic manipulation systems where
visual observations must be transformed into useful representations.

In a production system, the synthetic dataset used here would be replaced
by real camera data containing:

    - lighting variation
    - camera noise
    - object deformation
    - occlusion
    - clutter
    - multiple object instances
    - real backgrounds
    """
    )

    print(
        f"Measured residual CNN test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Residual CNN trainable parameters: "
        f"{parameter_count:,}"
    )


# =============================================================================
# LESSON EXPERIMENT
# =============================================================================

def run_parameter_experiment():
    """
    Demonstrate that model depth and architecture can be changed without
    changing the basic training interface.

    This is intentionally lightweight.

    The important lesson is architectural experimentation.
    """

    print()
    print("=" * 80)
    print("ARCHITECTURE EXPERIMENT")
    print("=" * 80)

    plain = PlainDeepCNN()

    residual = ResidualCNN()

    print(
        f"Plain CNN parameters    : "
        f"{count_parameters(plain):,}"
    )

    print(
        f"Residual CNN parameters : "
        f"{count_parameters(residual):,}"
    )

    ratio = (
        count_parameters(residual)
        / count_parameters(plain)
    )

    print(
        f"Residual / Plain parameter ratio: "
        f"{ratio:.3f}"
    )

    print(
        """
Important observation:

Residual learning is not simply "adding more parameters."

Its main architectural contribution is the shortcut path:

        x --------------------+
        |                     |
        v                     |
        F(x) ---------------->+
                              |
                              v
                            F(x)+x

The shortcut changes how information and gradients move through the network.
"""
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    print_environment()

    # -------------------------------------------------------------------------
    # STEP 1 — BASIC TESTS BEFORE TRAINING
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 1 — ARCHITECTURE TESTS")
    print("=" * 80)

    test_residual_identity_behavior()

    test_projection_shortcut()

    test_model_forward()

    test_gradient_exists()

    # -------------------------------------------------------------------------
    # STEP 2 — CREATE DATASET
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 2 — CREATE SYNTHETIC VISION DATASET")
    print("=" * 80)

    train_images, train_labels = create_dataset(
        TRAIN_SAMPLES
    )

    val_images, val_labels = create_dataset(
        VAL_SAMPLES
    )

    test_images, test_labels = create_dataset(
        TEST_SAMPLES
    )

    print(
        f"Training images   : {train_images.shape}"
    )

    print(
        f"Validation images : {val_images.shape}"
    )

    print(
        f"Test images       : {test_images.shape}"
    )

    visualize_dataset(
        train_images,
        train_labels
    )

    # -------------------------------------------------------------------------
    # STEP 3 — DATA LOADERS
    # -------------------------------------------------------------------------

    train_dataset = PatternDataset(
        train_images,
        train_labels
    )

    val_dataset = PatternDataset(
        val_images,
        val_labels
    )

    test_dataset = PatternDataset(
        test_images,
        test_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -------------------------------------------------------------------------
    # STEP 4 — CREATE MODELS
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 4 — CREATE MODELS")
    print("=" * 80)

    plain_model = PlainDeepCNN()

    residual_model = ResidualCNN()

    print(
        f"Plain CNN parameters    : "
        f"{count_parameters(plain_model):,}"
    )

    print(
        f"Residual CNN parameters : "
        f"{count_parameters(residual_model):,}"
    )

    # -------------------------------------------------------------------------
    # STEP 5 — TRAIN PLAIN CNN
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 5 — TRAIN PLAIN CNN")
    print("=" * 80)

    plain_history, _ = train_model(
        plain_model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # -------------------------------------------------------------------------
    # STEP 6 — TRAIN RESIDUAL CNN
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 6 — TRAIN RESIDUAL CNN")
    print("=" * 80)

    residual_history, residual_state = train_model(
        residual_model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        checkpoint_path=CHECKPOINT_PATH
    )

    # -------------------------------------------------------------------------
    # STEP 7 — TEST BOTH MODELS
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 7 — TEST SET")
    print("=" * 80)

    criterion = nn.CrossEntropyLoss()

    plain_test_loss, plain_test_accuracy, \
        plain_predictions, plain_true = evaluate(
            plain_model,
            test_loader,
            criterion
        )

    residual_test_loss, residual_test_accuracy, \
        residual_predictions, residual_true = evaluate(
            residual_model,
            test_loader,
            criterion
        )

    print(
        f"Plain CNN    Test Loss={plain_test_loss:.4f} "
        f"Accuracy={plain_test_accuracy:.4f}"
    )

    print(
        f"Residual CNN Test Loss={residual_test_loss:.4f} "
        f"Accuracy={residual_test_accuracy:.4f}"
    )

    # -------------------------------------------------------------------------
    # STEP 8 — CONFUSION MATRICES
    # -------------------------------------------------------------------------

    plain_matrix = compute_confusion_matrix(
        plain_true,
        plain_predictions
    )

    residual_matrix = compute_confusion_matrix(
        residual_true,
        residual_predictions
    )

    print()
    print("Plain CNN confusion matrix:")
    print(plain_matrix)

    print()
    print("Residual CNN confusion matrix:")
    print(residual_matrix)

    plot_confusion_matrix(
        plain_matrix,
        "Plain CNN Confusion Matrix"
    )

    plot_confusion_matrix(
        residual_matrix,
        "Residual CNN Confusion Matrix"
    )

    # -------------------------------------------------------------------------
    # STEP 9 — TRAINING CURVES
    # -------------------------------------------------------------------------

    plot_training_history(
        plain_history,
        residual_history
    )

    # -------------------------------------------------------------------------
    # STEP 10 — GRADIENT FLOW
    # -------------------------------------------------------------------------

    plain_gradients = measure_gradient_flow(
        plain_model,
        train_loader
    )

    residual_gradients = measure_gradient_flow(
        residual_model,
        train_loader
    )

    print_gradient_flow(
        plain_gradients,
        "PLAIN CNN GRADIENT FLOW"
    )

    print_gradient_flow(
        residual_gradients,
        "RESIDUAL CNN GRADIENT FLOW"
    )

    # -------------------------------------------------------------------------
    # STEP 11 — PREDICTION VISUALIZATION
    # -------------------------------------------------------------------------

    visualize_predictions(
        residual_model,
        test_dataset
    )

    # -------------------------------------------------------------------------
    # STEP 12 — ARCHITECTURE EXPERIMENT
    # -------------------------------------------------------------------------

    run_parameter_experiment()

    # -------------------------------------------------------------------------
    # STEP 13 — REAL-WORLD APPLICATION
    # -------------------------------------------------------------------------

    real_world_application(
        residual_test_accuracy,
        count_parameters(residual_model)
    )

    # -------------------------------------------------------------------------
    # STEP 14 — FINAL TESTS
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)

    assert len(plain_history["train_loss"]) == EPOCHS

    assert len(residual_history["train_loss"]) == EPOCHS

    assert np.isfinite(
        plain_test_loss
    )

    assert np.isfinite(
        residual_test_loss
    )

    assert np.isfinite(
        plain_test_accuracy
    )

    assert np.isfinite(
        residual_test_accuracy
    )

    # The synthetic problem is intentionally simple.
    # A correctly functioning CNN should achieve high accuracy.
    assert residual_test_accuracy > 0.85, (
        "Residual CNN test accuracy is unexpectedly low."
    )

    print(
        "[PASS] Training history test"
    )

    print(
        "[PASS] Finite test-loss verification"
    )

    print(
        "[PASS] Accuracy verification"
    )

    # -------------------------------------------------------------------------
    # FINAL LESSON RESULT
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LESSON RESULT")
    print("=" * 80)

    print(
        """
You have now moved from a basic CNN to deep residual CNN design.

The most important conceptual result is:

        Deep network
             |
             v
        difficult optimization
             |
             v
        residual connection
             |
             v
        F(x) + x
             |
             v
        easier information / gradient flow

You should now understand:

    [x] why CNNs become deeper
    [x] why depth can create optimization problems
    [x] what residual learning means
    [x] what a skip connection does
    [x] when identity shortcuts work
    [x] when projection shortcuts are required
    [x] how to implement a residual block
    [x] how to build a ResNet-style CNN
    [x] how to measure gradient flow
    [x] how residual CNNs fit into robot vision

This is a major transition from "CNN as a classifier" toward
"deep visual representation learning."
"""
    )

    # -------------------------------------------------------------------------
    # CONNECTION TO NEXT LESSON
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CONNECTION TO LESSON 9")
    print("=" * 80)

    print(
        """
LESSON 9 will move beyond supervised classification.

Instead of asking:

    "What class is this image?"

we will ask:

    "Can a neural network learn a useful representation of the image
     without directly predicting its class?"

This leads to:

    representation learning
            |
            v
        autoencoders
            |
            v
        encoder
            |
            v
        latent representation
            |
            v
        decoder
            |
            v
        reconstructed image

This is the conceptual bridge toward generative modeling.

The latent representation learned by an encoder will later become
particularly important when we study:

    VAE
      |
    latent diffusion
      |
    conditional generation
      |
    text-to-image generation

So the progression is:

    CNN
      ↓
    Deep CNN
      ↓
    Residual CNN
      ↓
    Representation Learning
      ↓
    Autoencoder
      ↓
    VAE
      ↓
    Generative Models
      ↓
    Diffusion
      ↓
    Text-to-Image
"""
    )

    print()
    print("=" * 80)
    print("END OF LESSON 8")
    print("=" * 80)


if __name__ == "__main__":
    main()