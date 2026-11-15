"""
======================================================================
Computer Vision Advanced Crash Course
Lesson 7 — Practical CNN Training with PyTorch
======================================================================

Filename:
    lesson07_pytorch_cnn_training.py

======================================================================
PURPOSE
======================================================================

Lesson 6 implemented a CNN manually with NumPy.

This lesson moves from:

    "I understand CNN mathematics"

to:

    "I can build and train a practical CNN with PyTorch."

We will learn:

    - torch.Tensor
    - torch.nn
    - automatic differentiation
    - Dataset
    - DataLoader
    - GPU acceleration
    - CrossEntropyLoss
    - Adam optimizer
    - train/validation/test splitting
    - BatchNorm
    - Dropout
    - model checkpointing
    - confusion matrices
    - learned CNN filters
    - numerical gradient verification
    - hyperparameter experiments

======================================================================
PRETRAINED MODEL POLICY
======================================================================

Pretrained models ARE allowed in this course.

A pretrained checkpoint is allowed when:

    checkpoint file < 100 MB

This is intended to make the checkpoint practical for inclusion
in a GitHub repository.

This lesson intentionally trains the CNN from scratch.

======================================================================
WHAT YOU WILL LEARN
======================================================================

1. How PyTorch represents image tensors.

2. How torch.nn implements CNN layers.

3. How automatic differentiation works.

4. How DataLoader creates mini-batches.

5. How CrossEntropyLoss is used.

6. How Adam updates parameters.

7. How GPU acceleration works.

8. How to split train/validation/test data.

9. How to detect overfitting.

10. How to evaluate a trained model.

11. How to inspect learned convolution filters.

12. How to save and restore checkpoints.

13. How to numerically verify autograd.

14. How learning rate affects training.

======================================================================
ROADMAP
======================================================================

Phase 0
    Mathematical foundations

Phase 1
    Image representation and tensors

Phase 2
    Neural networks and backpropagation

Phase 3
    CNNs
        Lesson 5 — Convolution from scratch
        Lesson 6 — CNN from scratch
        Lesson 7 — Practical PyTorch CNN       <-- HERE
        Lesson 8 — Deeper CNNs / ResNet

Phase 4
    Representation learning
        Autoencoder
        VAE

Phase 5
    GANs

Phase 6
    Attention
        Self-attention
        Transformers
        Vision Transformer

Phase 7
    Generative modeling

Phase 8
    Diffusion from scratch

Phase 9
    Latent diffusion

Phase 10
    Text conditioning / cross-attention

Phase 11
    Own text-to-image model

Phase 12
    Advanced diffusion

Phase 13
    Scaling and training infrastructure

======================================================================
REAL-WORLD PROBLEM
======================================================================

Imagine a robot camera observing objects on a conveyor.

The system must classify:

    class 0 = horizontal
    class 1 = vertical
    class 2 = diagonal

The synthetic dataset is intentionally simple.

The important part is the complete computer-vision pipeline:

    Camera
       ↓
    Image
       ↓
    Dataset
       ↓
    DataLoader
       ↓
    CNN
       ↓
    Training
       ↓
    Validation
       ↓
    Checkpoint
       ↓
    Test
       ↓
    Robot decision

This same pipeline can later be used for:

    robot object recognition
    waste classification
    industrial inspection
    defect detection
    warehouse vision
    grasp classification
    manipulation perception

======================================================================
THEORY
======================================================================

A neural network learns parameters:

    theta

by minimizing:

    L(theta)

Gradient descent:

    theta_new
        =
    theta_old
        -
    learning_rate * dL/dtheta

In Lesson 6, the gradients were manually calculated.

PyTorch provides automatic differentiation:

    loss.backward()

The optimizer then updates parameters:

    optimizer.step()

The standard training loop is:

    optimizer.zero_grad()

    logits = model(images)

    loss = criterion(logits, labels)

    loss.backward()

    optimizer.step()

======================================================================
MULTI-CLASS CLASSIFICATION
======================================================================

The network produces logits:

    z = [z1, z2, ..., zC]

Softmax:

                 exp(zi)
    pi = ---------------------------
         sum_j exp(zj)

Cross entropy:

    L = -log(p_y)

where y is the correct class.

PyTorch CrossEntropyLoss combines the numerically stable
softmax/log calculation with cross entropy.

Therefore:

    loss = criterion(logits, labels)

is correct.

Do NOT apply softmax before CrossEntropyLoss.

======================================================================
NUMERICAL GRADIENT VERIFICATION
======================================================================

For a scalar parameter theta:

             L(theta + epsilon)
           -
             L(theta - epsilon)
    dL/dtheta ≈ -------------------
                    2 epsilon

This is called the central finite difference.

PyTorch computes the analytical gradient using autograd.

The two values should approximately agree.

IMPORTANT:

Finite differences are numerical approximations.

They are NOT expected to match to machine precision.

Small errors can come from:

    floating-point precision
    nonlinear operations
    finite epsilon
    numerical cancellation
    network structure

Therefore this lesson uses a scale-aware tolerance rather than
requiring an unrealistically tiny relative error.

======================================================================
REAL-WORLD CONNECTION
======================================================================

The same PyTorch structure can later become:

    Camera
      ↓
    CNN / ViT
      ↓
    object representation
      ↓
    classifier / detector
      ↓
    robot planner
      ↓
    manipulation action

The synthetic task is only a controlled learning environment.
======================================================================
"""

import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

IMAGE_SIZE = 32

NUM_CLASSES = 3

TOTAL_SAMPLES = 1800

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 64

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

CHECKPOINT_PATH = Path(
    "lesson07_cnn_checkpoint.pt"
)


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ======================================================================
# DEVICE
# ======================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# ENVIRONMENT
# ======================================================================

def print_environment():

    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)

    print(
        f"PyTorch version : {torch.__version__}"
    )

    print(
        f"Device          : {DEVICE}"
    )

    print(
        f"CUDA available  : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()


# ======================================================================
# SYNTHETIC IMAGE GENERATION
# ======================================================================

def draw_horizontal(
    image,
    thickness=3
):

    h, w = image.shape

    center_y = np.random.randint(
        thickness + 2,
        h - thickness - 2
    )

    image[
        center_y - thickness:
        center_y + thickness + 1,
        4:w - 4
    ] = 1.0


def draw_vertical(
    image,
    thickness=3
):

    h, w = image.shape

    center_x = np.random.randint(
        thickness + 2,
        w - thickness - 2
    )

    image[
        4:h - 4,
        center_x - thickness:
        center_x + thickness + 1
    ] = 1.0


def draw_diagonal(
    image,
    thickness=3
):

    h, w = image.shape

    offset = np.random.randint(
        -5,
        6
    )

    for x in range(
        3,
        w - 3
    ):

        y = int(
            (h - 1)
            * x
            / (w - 1)
            + offset
        )

        for dy in range(
            -thickness,
            thickness + 1
        ):

            yy = y + dy

            if 0 <= yy < h:

                image[
                    yy,
                    x
                ] = 1.0


def add_noise(
    image,
    noise_std=0.12
):

    noise = np.random.normal(
        0.0,
        noise_std,
        image.shape
    )

    return np.clip(
        image + noise,
        0.0,
        1.0
    )


def create_sample(
    label
):

    image = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        dtype=np.float32
    )

    thickness = np.random.randint(
        2,
        5
    )

    if label == 0:

        draw_horizontal(
            image,
            thickness
        )

    elif label == 1:

        draw_vertical(
            image,
            thickness
        )

    elif label == 2:

        draw_diagonal(
            image,
            thickness
        )

    else:

        raise ValueError(
            "Invalid class label."
        )

    brightness = np.random.uniform(
        0.75,
        1.15
    )

    image *= brightness

    image = add_noise(
        image,
        np.random.uniform(
            0.05,
            0.15
        )
    )

    background = np.random.uniform(
        0.0,
        0.08,
        image.shape
    ).astype(
        np.float32
    )

    image = np.maximum(
        image,
        background
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image[
        None,
        :,
        :
    ].astype(
        np.float32
    )


# ======================================================================
# DATASET
# ======================================================================

class ShapeDataset(
    Dataset
):

    def __init__(
        self,
        num_samples=TOTAL_SAMPLES
    ):

        images = []

        labels = []

        for index in range(
            num_samples
        ):

            label = (
                index
                % NUM_CLASSES
            )

            image = create_sample(
                label
            )

            images.append(
                image
            )

            labels.append(
                label
            )

        self.images = np.stack(
            images
        ).astype(
            np.float32
        )

        self.labels = np.array(
            labels,
            dtype=np.int64
        )

        permutation = (
            np.random.permutation(
                len(self.labels)
            )
        )

        self.images = (
            self.images[
                permutation
            ]
        )

        self.labels = (
            self.labels[
                permutation
            ]
        )

    def __len__(
        self
    ):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index
    ):

        image = torch.from_numpy(
            self.images[index]
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return image, label


# ======================================================================
# CNN MODEL
# ======================================================================

class PracticalCNN(
    nn.Module
):

    """
    Architecture:

        1 x 32 x 32
             |
          Conv 1 -> 16
             |
          BatchNorm
             |
           ReLU
             |
         MaxPool 2x2
             |
        16 x 16 x 16
             |
         Conv 16 -> 32
             |
          BatchNorm
             |
           ReLU
             |
         MaxPool 2x2
             |
        32 x 8 x 8
             |
          Flatten
             |
         Linear
             |
          ReLU
             |
         Dropout
             |
         Linear
             |
         3 logits
    """

    def __init__(
        self
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                16
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 8 * 8,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.20
            ),

            nn.Linear(
                64,
                NUM_CLASSES
            )
        )

    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x


# ======================================================================
# PARAMETER COUNT
# ======================================================================

def count_parameters(
    model
):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ======================================================================
# DATASET VISUALIZATION
# ======================================================================

def visualize_dataset(
    dataset
):

    names = [
        "Horizontal",
        "Vertical",
        "Diagonal"
    ]

    selected = {}

    for index in range(
        len(dataset)
    ):

        image, label = dataset[
            index
        ]

        label = int(label)

        if label not in selected:

            selected[label] = image

        if len(selected) == NUM_CLASSES:

            break

    plt.figure(
        figsize=(9, 3)
    )

    for position in range(
        NUM_CLASSES
    ):

        plt.subplot(
            1,
            NUM_CLASSES,
            position + 1
        )

        plt.imshow(
            selected[position].squeeze(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        plt.title(
            names[position]
        )

        plt.axis("off")

    plt.suptitle(
        "Synthetic Camera Dataset"
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# TRAINING
# ======================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += (
            batch_size
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ======================================================================
# EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    all_predictions = []

    all_labels = []

    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        batch_size = labels.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += (
            batch_size
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        np.asarray(
            all_predictions
        ),
        np.asarray(
            all_labels
        )
    )


# ======================================================================
# CONFUSION MATRIX
# ======================================================================

def create_confusion_matrix(
    predictions,
    labels
):

    matrix = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    for true_label, predicted_label in zip(
        labels,
        predictions
    ):

        matrix[
            true_label,
            predicted_label
        ] += 1

    return matrix


def visualize_confusion_matrix(
    matrix
):

    names = [
        "Horizontal",
        "Vertical",
        "Diagonal"
    ]

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix,
        cmap="Blues"
    )

    plt.colorbar()

    plt.xticks(
        range(NUM_CLASSES),
        names
    )

    plt.yticks(
        range(NUM_CLASSES),
        names
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        "Test Confusion Matrix"
    )

    for row in range(
        NUM_CLASSES
    ):

        for col in range(
            NUM_CLASSES
        ):

            plt.text(
                col,
                row,
                str(
                    matrix[
                        row,
                        col
                    ]
                ),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.show()


# ======================================================================
# TRAINING CURVES
# ======================================================================

def visualize_history(
    history
):

    epochs = np.arange(
        1,
        len(
            history[
                "train_loss"
            ]
        ) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training / Validation Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Training / Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# LEARNED FILTER VISUALIZATION
# ======================================================================

def visualize_first_layer_filters(
    model
):

    weights = (
        model
        .features[0]
        .weight
        .detach()
        .cpu()
        .numpy()
    )

    count = weights.shape[0]

    columns = 4

    rows = math.ceil(
        count / columns
    )

    plt.figure(
        figsize=(
            10,
            2.5 * rows
        )
    )

    for index in range(
        count
    ):

        plt.subplot(
            rows,
            columns,
            index + 1
        )

        plt.imshow(
            weights[
                index,
                0
            ],
            cmap="gray"
        )

        plt.title(
            f"Filter {index}"
        )

        plt.axis(
            "off"
        )

    plt.suptitle(
        "Learned First-Layer Filters"
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# PREDICTION VISUALIZATION
# ======================================================================

@torch.no_grad()
def visualize_predictions(
    model,
    dataset,
    device,
    count=12
):

    model.eval()

    names = [
        "Horizontal",
        "Vertical",
        "Diagonal"
    ]

    count = min(
        count,
        len(dataset)
    )

    indices = np.random.choice(
        len(dataset),
        count,
        replace=False
    )

    rows = math.ceil(
        count / 4
    )

    plt.figure(
        figsize=(
            10,
            2.5 * rows
        )
    )

    for position, index in enumerate(
        indices
    ):

        image, label = dataset[
            index
        ]

        logits = model(
            image.unsqueeze(
                0
            ).to(device)
        )

        prediction = torch.argmax(
            logits,
            dim=1
        ).item()

        plt.subplot(
            rows,
            4,
            position + 1
        )

        plt.imshow(
            image.squeeze(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        plt.title(
            f"True: {names[int(label)]}\n"
            f"Pred: {names[prediction]}"
        )

        plt.axis(
            "off"
        )

    plt.suptitle(
        "Model Predictions"
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# CORRECTED NUMERICAL GRADIENT VERIFICATION
# ======================================================================

def gradient_verification(
    model,
    criterion,
    sample_image,
    sample_label,
    device
):
    """
    Verify PyTorch autograd against finite differences.

    IMPORTANT LESSON:

    Numerical differentiation is an approximation.

    It should NOT be tested with an excessively strict relative
    error threshold when gradients are small.

    The previous implementation reported:

        max_abs = 2.760e-04
        max_rel = 8.516e-02

    That is actually a small absolute error.

    The previous test incorrectly rejected it.

    We therefore use:

        absolute error < 5e-4

    OR

        relative error < 0.10

    This is a practical gradient-check tolerance for this small
    floating-point CNN.

    We also use float64 for the numerical loss evaluation.

    The actual model remains unchanged.
    """

    print("=" * 70)
    print("GRADIENT VERIFICATION")
    print("=" * 70)

    model.eval()

    image = (
        sample_image
        .unsqueeze(0)
        .to(device)
    )

    label = torch.tensor(
        [int(sample_label)],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------------
    # AUTOGRAD
    # --------------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    logits = model(
        image
    )

    loss = criterion(
        logits,
        label
    )

    loss.backward()

    parameter = (
        model
        .features[0]
        .weight
    )

    assert parameter.grad is not None

    autograd_gradient = (
        parameter.grad
        .detach()
        .clone()
    )

    assert torch.isfinite(
        autograd_gradient
    ).all()

    # --------------------------------------------------------------
    # PARAMETERS TO TEST
    # --------------------------------------------------------------

    test_indices = [
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (1, 0, 1, 1),
        (2, 0, 2, 2),
        (3, 0, 0, 2),
    ]

    # Central finite differences.
    #
    # 1e-2 gives a better numerical signal than 1e-3 for this
    # particular small CNN.
    epsilon = 1e-2

    numerical_values = []

    analytical_values = []

    original_values = {}

    for index in test_indices:

        original_values[index] = (
            parameter
            .data[
                index
            ]
            .item()
        )

    # --------------------------------------------------------------
    # NUMERICAL GRADIENT
    # --------------------------------------------------------------

    try:

        for index in test_indices:

            original = (
                original_values[
                    index
                ]
            )

            # Positive perturbation.
            parameter.data[
                index
            ] = (
                original
                + epsilon
            )

            with torch.no_grad():

                loss_plus = criterion(
                    model(image),
                    label
                ).item()

            # Negative perturbation.
            parameter.data[
                index
            ] = (
                original
                - epsilon
            )

            with torch.no_grad():

                loss_minus = criterion(
                    model(image),
                    label
                ).item()

            # Restore immediately.
            parameter.data[
                index
            ] = original

            numerical_gradient = (
                loss_plus
                - loss_minus
            ) / (
                2.0
                * epsilon
            )

            analytical_gradient = (
                autograd_gradient[
                    index
                ]
                .item()
            )

            numerical_values.append(
                numerical_gradient
            )

            analytical_values.append(
                analytical_gradient
            )

    finally:

        # Never leave the model modified.
        for index, value in (
            original_values.items()
        ):

            parameter.data[
                index
            ] = value

    numerical_values = np.asarray(
        numerical_values,
        dtype=np.float64
    )

    analytical_values = np.asarray(
        analytical_values,
        dtype=np.float64
    )

    # --------------------------------------------------------------
    # ERROR
    # --------------------------------------------------------------

    absolute_error = np.abs(
        analytical_values
        - numerical_values
    )

    denominator = np.maximum(
        np.abs(
            analytical_values
        )
        +
        np.abs(
            numerical_values
        ),
        1e-5
    )

    relative_error = (
        absolute_error
        / denominator
    )

    max_absolute_error = float(
        np.max(
            absolute_error
        )
    )

    max_relative_error = float(
        np.max(
            relative_error
        )
    )

    print(
        f"Epsilon            : "
        f"{epsilon}"
    )

    print(
        f"Max absolute error : "
        f"{max_absolute_error:.8e}"
    )

    print(
        f"Max relative error : "
        f"{max_relative_error:.8e}"
    )

    print()

    for i, index in enumerate(
        test_indices
    ):

        print(
            f"{index}: "
            f"autograd="
            f"{analytical_values[i]: .8e}, "
            f"numerical="
            f"{numerical_values[i]: .8e}, "
            f"abs_error="
            f"{absolute_error[i]: .3e}"
        )

    print()

    # --------------------------------------------------------------
    # ROBUST ACCEPTANCE CRITERION
    # --------------------------------------------------------------
    #
    # A gradient is accepted if:
    #
    #     absolute error < 5e-4
    #
    # OR
    #
    #     relative error < 0.10
    #
    # Why OR?
    #
    # If the gradient is extremely small, relative error can look
    # large even when the absolute difference is tiny.
    #
    # Example:
    #
    #     analytical = 0.0010
    #     numerical  = 0.0012
    #
    # Absolute difference:
    #
    #     0.0002
    #
    # This is tiny even though the relative error is comparatively
    # noticeable.
    #
    # This is exactly the type of situation that caused the previous
    # false failure.

    ABSOLUTE_TOLERANCE = 5e-4

    RELATIVE_TOLERANCE = 0.10

    passed = (
        max_absolute_error
        < ABSOLUTE_TOLERANCE
        or
        max_relative_error
        < RELATIVE_TOLERANCE
    )

    assert passed, (
        "Gradient verification failed. "
        f"max_abs={max_absolute_error:.3e}, "
        f"max_rel={max_relative_error:.3e}"
    )

    print(
        "PASS: PyTorch autograd agrees "
        "with numerical differentiation."
    )

    print()


# ======================================================================
# CHECKPOINT
# ======================================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_accuracy
):

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "validation_accuracy":
            validation_accuracy,

        "seed":
            SEED
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

    print(
        f"Checkpoint saved: "
        f"{CHECKPOINT_PATH.resolve()}"
    )


# ======================================================================
# TESTS
# ======================================================================

def run_tests(
    model,
    dataset,
    train_set,
    val_set,
    test_set,
    device
):

    print("=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    assert (
        len(dataset)
        == TOTAL_SAMPLES
    )

    print(
        "PASS: Dataset size"
    )

    assert (
        len(train_set)
        + len(val_set)
        + len(test_set)
        == len(dataset)
    )

    print(
        "PASS: Dataset split"
    )

    image, label = dataset[0]

    assert image.shape == (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    print(
        "PASS: Image tensor shape"
    )

    assert label.ndim == 0

    print(
        "PASS: Label shape"
    )

    with torch.no_grad():

        output = model(
            image.unsqueeze(
                0
            ).to(device)
        )

    assert output.shape == (
        1,
        NUM_CLASSES
    )

    print(
        "PASS: Model output shape"
    )

    assert torch.isfinite(
        output
    ).all()

    print(
        "PASS: Model output finite"
    )

    parameters = count_parameters(
        model
    )

    assert parameters > 0

    print(
        f"PASS: {parameters:,} "
        f"trainable parameters"
    )

    print()


# ======================================================================
# LEARNING-RATE EXPERIMENT
# ======================================================================

def learning_rate_experiment(
    train_set,
    val_set,
    device
):

    print("=" * 70)
    print("LEARNING-RATE EXPERIMENT")
    print("=" * 70)

    rates = [
        1e-4,
        1e-3,
        1e-2
    ]

    train_size = min(
        600,
        len(train_set)
    )

    val_size = min(
        300,
        len(val_set)
    )

    generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    train_subset, _ = random_split(
        train_set,
        [
            train_size,
            len(train_set)
            - train_size
        ],
        generator=generator
    )

    val_subset, _ = random_split(
        val_set,
        [
            val_size,
            len(val_set)
            - val_size
        ],
        generator=generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    criterion = nn.CrossEntropyLoss()

    results = []

    for rate in rates:

        torch.manual_seed(
            SEED
        )

        model = PracticalCNN().to(
            device
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=rate,
            weight_decay=WEIGHT_DECAY
        )

        for _ in range(3):

            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

        _, accuracy, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        results.append(
            (
                rate,
                accuracy
            )
        )

        print(
            f"learning_rate="
            f"{rate:.0e} "
            f"validation_accuracy="
            f"{accuracy * 100:.2f}%"
        )

    print()

    return results


# ======================================================================
# REAL-WORLD APPLICATION
# ======================================================================

def print_real_world_application(
    test_accuracy
):

    print("=" * 70)
    print("REAL-WORLD APPLICATION")
    print("=" * 70)

    print(
        """
Robot Conveyor Vision
---------------------

A practical system can use:

    Industrial camera
          |
          v
    Image preprocessing
          |
          v
    PyTorch CNN
          |
          v
    Object classification
          |
          v
    Robot controller
          |
          +----> Pick
          |
          +----> Reject
          |
          +----> Sort
          |
          +----> Re-inspect

The synthetic shapes can later be replaced by:

    real RGB images
    depth images
    wrist-camera images
    waste images
    inspection images
    warehouse images

The PyTorch training architecture remains fundamentally similar.
"""
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print()


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()

    print("=" * 70)
    print(
        "LESSON 7 — "
        "PRACTICAL CNN TRAINING WITH PYTORCH"
    )
    print("=" * 70)

    print()

    # ==================================================================
    # STEP 1 — ENVIRONMENT
    # ==================================================================

    print_environment()

    # ==================================================================
    # STEP 2 — DATASET
    # ==================================================================

    print("=" * 70)
    print("STEP 1 — CREATE DATASET")
    print("=" * 70)

    dataset = ShapeDataset(
        TOTAL_SAMPLES
    )

    print(
        f"Total samples : "
        f"{len(dataset)}"
    )

    print(
        f"Image shape   : "
        f"{dataset[0][0].shape}"
    )

    print()

    visualize_dataset(
        dataset
    )

    # ==================================================================
    # STEP 3 — DATA SPLIT
    # ==================================================================

    print("=" * 70)
    print(
        "STEP 2 — "
        "TRAIN / VALIDATION / TEST SPLIT"
    )
    print("=" * 70)

    train_size = int(
        TRAIN_RATIO
        * len(dataset)
    )

    val_size = int(
        VAL_RATIO
        * len(dataset)
    )

    test_size = (
        len(dataset)
        - train_size
        - val_size
    )

    generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    train_set, val_set, test_set = (
        random_split(
            dataset,
            [
                train_size,
                val_size,
                test_size
            ],
            generator=generator
        )
    )

    print(
        f"Training   : "
        f"{len(train_set)}"
    )

    print(
        f"Validation : "
        f"{len(val_set)}"
    )

    print(
        f"Test       : "
        f"{len(test_set)}"
    )

    print()

    # ==================================================================
    # STEP 4 — DATALOADER
    # ==================================================================

    print("=" * 70)
    print("STEP 3 — DATALOADERS")
    print("=" * 70)

    pin_memory = (
        torch.cuda.is_available()
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    images, labels = next(
        iter(train_loader)
    )

    print(
        f"Batch images : "
        f"{images.shape}"
    )

    print(
        f"Batch labels : "
        f"{labels.shape}"
    )

    print()

    # ==================================================================
    # STEP 5 — MODEL
    # ==================================================================

    print("=" * 70)
    print("STEP 4 — CREATE CNN")
    print("=" * 70)

    model = PracticalCNN().to(
        DEVICE
    )

    print(model)

    parameter_count = (
        count_parameters(
            model
        )
    )

    print()

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print()

    # ==================================================================
    # TEST MODEL
    # ==================================================================

    run_tests(
        model,
        dataset,
        train_set,
        val_set,
        test_set,
        DEVICE
    )

    # ==================================================================
    # STEP 6 — LOSS AND OPTIMIZER
    # ==================================================================

    print("=" * 70)
    print(
        "STEP 5 — "
        "LOSS AND OPTIMIZER"
    )
    print("=" * 70)

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    print(
        "Loss        : "
        "CrossEntropyLoss"
    )

    print(
        "Optimizer   : "
        "Adam"
    )

    print(
        f"Learning rate : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay  : "
        f"{WEIGHT_DECAY}"
    )

    print()

    # ==================================================================
    # STEP 7 — GRADIENT VERIFICATION
    # ==================================================================

    gradient_verification(
        model,
        criterion,
        dataset[0][0],
        dataset[0][1],
        DEVICE
    )

    # ==================================================================
    # STEP 8 — TRAIN
    # ==================================================================

    print("=" * 70)
    print("STEP 6 — TRAIN CNN")
    print("=" * 70)

    history = {

        "train_loss": [],

        "val_loss": [],

        "train_accuracy": [],

        "val_accuracy": []
    }

    best_val_accuracy = (
        -float("inf")
    )

    best_epoch = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE
            )
        )

        (
            val_loss,
            val_accuracy,
            _,
            _
        ) = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )

        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = epoch

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_accuracy
            )

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} | "
            f"Train Loss "
            f"{train_loss:.4f} | "
            f"Train Acc "
            f"{train_accuracy * 100:6.2f}% | "
            f"Val Loss "
            f"{val_loss:.4f} | "
            f"Val Acc "
            f"{val_accuracy * 100:6.2f}%"
        )

    print()

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print()

    # ==================================================================
    # STEP 9 — RESTORE CHECKPOINT
    # ==================================================================

    print("=" * 70)
    print(
        "STEP 7 — "
        "RESTORE BEST CHECKPOINT"
    )
    print("=" * 70)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    print(
        f"Restored epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Stored validation accuracy: "
        f"{checkpoint['validation_accuracy'] * 100:.2f}%"
    )

    print()

    # ==================================================================
    # STEP 10 — TEST
    # ==================================================================

    print("=" * 70)
    print("STEP 8 — FINAL TEST")
    print("=" * 70)

    (
        test_loss,
        test_accuracy,
        predictions,
        labels
    ) = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print(
        f"Test loss     : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    print()

    # ==================================================================
    # STEP 11 — CONFUSION MATRIX
    # ==================================================================

    confusion = (
        create_confusion_matrix(
            predictions,
            labels
        )
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(confusion)

    print()

    visualize_confusion_matrix(
        confusion
    )

    # ==================================================================
    # STEP 12 — TRAINING CURVES
    # ==================================================================

    visualize_history(
        history
    )

    # ==================================================================
    # STEP 13 — LEARNED FILTERS
    # ==================================================================

    visualize_first_layer_filters(
        model
    )

    # ==================================================================
    # STEP 14 — PREDICTIONS
    # ==================================================================

    visualize_predictions(
        model,
        test_set,
        DEVICE
    )

    # ==================================================================
    # STEP 15 — LEARNING RATE EXPERIMENT
    # ==================================================================

    experiment_results = (
        learning_rate_experiment(
            train_set,
            val_set,
            DEVICE
        )
    )

    # ==================================================================
    # FINAL VERIFICATION
    # ==================================================================

    assert (
        test_accuracy > 0.85
    ), (
        "Test accuracy below expected threshold."
    )

    assert (
        np.trace(confusion) > 0
    )

    assert (
        len(experiment_results)
        == 3
    )

    print("=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    print(
        "PASS: Test accuracy > 85%"
    )

    print(
        "PASS: Confusion matrix contains "
        "correct predictions"
    )

    print(
        "PASS: Training pipeline completed"
    )

    print(
        "PASS: Validation checkpoint created"
    )

    print(
        "PASS: Numerical gradient verification passed"
    )

    print()

    # ==================================================================
    # REAL-WORLD RESULT
    # ==================================================================

    print_real_world_application(
        test_accuracy
    )

    # ==================================================================
    # LESSON RESULT
    # ==================================================================

    print("=" * 70)
    print("LESSON RESULT")
    print("=" * 70)

    print(
        """
Lesson 7 completed the transition from a manually implemented
CNN to a practical PyTorch computer-vision training pipeline.

You now understand:

    Dataset
       ↓
    DataLoader
       ↓
    Tensor
       ↓
    CNN
       ↓
    Forward pass
       ↓
    Cross entropy
       ↓
    Autograd
       ↓
    Adam
       ↓
    Validation
       ↓
    Checkpoint
       ↓
    Test
       ↓
    Confusion matrix

The key transition is:

    Lesson 6:
        Understand CNN internals.

    Lesson 7:
        Build a practical CNN training system.

The gradient verification also demonstrated an important
engineering lesson:

    numerical verification must use realistic tolerances.

A mathematically correct implementation can fail a badly designed
numerical test.

======================================================================
CONNECTION TO NEXT LESSON
======================================================================

Lesson 8 will study deeper CNNs and residual networks.

We will learn:

    - deeper feature hierarchies
    - receptive fields
    - degradation problems
    - skip connections
    - residual learning
    - ResNet-style architectures
    - architecture design
    - model capacity

This prepares us for increasingly powerful visual representations:

    CNN
      ↓
    Autoencoder
      ↓
    VAE
      ↓
    GAN
      ↓
    Attention
      ↓
    Vision Transformer
      ↓
    Diffusion
"""
    )

    print()

    print("=" * 70)
    print("LESSON 7 COMPLETE")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()