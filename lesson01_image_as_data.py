"""
======================================================================
COMPUTER VISION ADVANCED CRASH COURSE
LESSON 01 — IMAGES ARE NUMERICAL DATA
======================================================================

PURPOSE
-------

This is the first lesson of the Computer Vision Advanced Crash Course.

The purpose is to understand the most fundamental idea in computer
vision:

                    AN IMAGE IS DATA.

A human looks at an image and sees:

    "a red object"
    "a green object"
    "a blue object"

A computer does not see those concepts directly.

A computer receives numbers.

For an RGB image, those numbers can be represented as:

                    H × W × 3

where:

    H = image height
    W = image width
    3 = RGB color channels

This lesson intentionally contains:

    - NO pretrained model
    - NO neural network
    - NO pretrained weights
    - NO computer-vision model

We are starting from the raw numerical representation.

This is important because every advanced model we build later will
ultimately operate on numerical tensors.

The progression will eventually become:

    IMAGE
      ↓
    NUMERICAL TENSOR
      ↓
    MATHEMATICAL OPERATIONS
      ↓
    CNN
      ↓
    REPRESENTATION LEARNING
      ↓
    TRANSFORMER
      ↓
    VAE
      ↓
    DIFFUSION
      ↓
    LATENT DIFFUSION
      ↓
    CONDITIONING
      ↓
    YOUR OWN GENERATIVE MODEL


----------------------------------------------------------------------
WHAT YOU WILL LEARN
----------------------------------------------------------------------

1. What a pixel is.
2. What RGB channels are.
3. How an image is represented numerically.
4. What H × W × C means.
5. How to access individual pixels.
6. How to modify image regions numerically.
7. How image values are normalized.
8. How to calculate image statistics.
9. How to separate color channels.
10. How a simple real-world vision task can be performed directly
    from pixel values.


----------------------------------------------------------------------
REAL-WORLD PROBLEM
----------------------------------------------------------------------

Imagine a robot has an RGB camera.

The camera produces something like:

    640 × 480 × 3

The robot software receives numerical values.

Suppose we want the robot to detect a red object.

A very simple vision pipeline could be:

    CAMERA
       ↓
    RGB FRAME
       ↓
    PIXEL VALUES
       ↓
    RED-PIXEL DETECTION
       ↓
    RED OBJECT REGION
       ↓
    APPROXIMATE OBJECT LOCATION

This is obviously much simpler than a modern object detector.

But it is a REAL computer-vision operation.

Before using CNNs, YOLO, Vision Transformers, or diffusion models,
we need to understand this fundamental pipeline.


----------------------------------------------------------------------
MATHEMATICAL REPRESENTATION
----------------------------------------------------------------------

A grayscale image can be represented as:

    I ∈ R^(H × W)

An RGB image can be represented as:

    I ∈ R^(H × W × 3)

A single pixel is:

    I[y, x] = [R, G, B]

For uint8 RGB images:

    0   = minimum intensity
    255 = maximum intensity

Examples:

    [255,   0,   0] = red
    [  0, 255,   0] = green
    [  0,   0, 255] = blue
    [255, 255, 255] = white
    [  0,   0,   0] = black

Therefore an image is essentially a large collection of RGB
vectors arranged according to spatial coordinates.


----------------------------------------------------------------------
IMAGE COORDINATES
----------------------------------------------------------------------

We access an image using:

    image[y, x]

where:

    y = vertical coordinate
    x = horizontal coordinate

For an RGB image:

    image[y, x, channel]

with:

    channel = 0 -> Red
    channel = 1 -> Green
    channel = 2 -> Blue

This coordinate convention is extremely important.

Later, CNNs and other vision models operate on the same spatial
structure.


----------------------------------------------------------------------
NORMALIZATION
----------------------------------------------------------------------

Camera images commonly use:

    0 ... 255

Neural networks often work with floating-point values.

A simple normalization is:

    x_normalized = x / 255

Therefore:

    0   → 0.0
    128 → approximately 0.502
    255 → 1.0

Normalization does not change the underlying visual information.

It changes the numerical scale.

Later we will see why numerical scale becomes extremely important
for neural-network training.


----------------------------------------------------------------------
REAL-WORLD APPLICATION
----------------------------------------------------------------------

The same image representation appears in:

    • robot cameras
    • drone cameras
    • autonomous vehicles
    • industrial inspection
    • surveillance systems
    • medical imaging
    • mobile-phone cameras
    • object detection
    • image segmentation
    • image generation
    • diffusion models

For example:

    RGB camera
        ↓
    1920 × 1080 × 3
        ↓
    computer vision system

The camera does not send the computer the concept:

    "There is a red object."

It sends pixel values.

The vision algorithm must extract useful information from those
numbers.


----------------------------------------------------------------------
LESSON EXPERIMENT
----------------------------------------------------------------------

We will construct a synthetic camera frame.

The frame will contain:

    • black background
    • red object
    • green object
    • blue object
    • white object

Then we will pretend that our "robot vision system" needs to find
the red object.

We will:

    1. Create the image.
    2. Inspect the raw pixel values.
    3. Normalize the image.
    4. Extract RGB channels.
    5. Detect red pixels.
    6. Calculate the red-object bounding box.
    7. Calculate its center.
    8. Visualize the result.

This gives us our first complete computer-vision pipeline.


----------------------------------------------------------------------
EXPECTED REAL-WORLD RESULT
----------------------------------------------------------------------

The program should identify approximately:

    red object bounding box
    red object center
    number of red pixels

The visualization should show:

    original camera frame
             +
    detected red region
             +
    estimated center


======================================================================
"""

# ======================================================================
# IMPORTS
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

# We use a 256 × 256 RGB image.
#
# In a real system this could be:
#
#     640 × 480 × 3
#     1280 × 720 × 3
#     1920 × 1080 × 3
#
# We use a smaller image so that the numerical representation is easy
# to understand.

HEIGHT = 256
WIDTH = 256
CHANNELS = 3


# ======================================================================
# 1. CREATE A SYNTHETIC CAMERA FRAME
# ======================================================================

# Create an image filled with zeros.
#
# Shape:
#
#     (HEIGHT, WIDTH, CHANNELS)
#
# Since all values are zero, the initial image is completely black.

image = np.zeros(
    (HEIGHT, WIDTH, CHANNELS),
    dtype=np.uint8
)


# ======================================================================
# 2. CREATE A RED OBJECT
# ======================================================================

# Select a rectangular region:
#
#     y = 50 ... 119
#     x = 40 ... 119
#
# Then set the RED channel to 255.
#
# Pixel:
#
#     [255, 0, 0]
#
# represents pure red.

image[50:120, 40:120, 0] = 255


# ======================================================================
# 3. CREATE A GREEN OBJECT
# ======================================================================

# Pixel:
#
#     [0, 255, 0]
#
# represents pure green.

image[50:120, 140:220, 1] = 255


# ======================================================================
# 4. CREATE A BLUE OBJECT
# ======================================================================

# Pixel:
#
#     [0, 0, 255]
#
# represents pure blue.

image[150:220, 40:120, 2] = 255


# ======================================================================
# 5. CREATE A WHITE OBJECT
# ======================================================================

# White requires maximum intensity in all three channels:
#
#     [255, 255, 255]

image[150:220, 140:220, :] = 255


# ======================================================================
# 6. INSPECT IMAGE STRUCTURE
# ======================================================================

print("=" * 70)
print("IMAGE STRUCTURE")
print("=" * 70)

print(f"Shape:              {image.shape}")
print(f"Height:             {image.shape[0]}")
print(f"Width:              {image.shape[1]}")
print(f"Channels:           {image.shape[2]}")
print(f"Data type:          {image.dtype}")
print(f"Number of values:   {image.size}")
print(f"Minimum value:      {image.min()}")
print(f"Maximum value:      {image.max()}")


# ======================================================================
# 7. UNDERSTAND PIXELS
# ======================================================================

# Let's inspect representative pixels.
#
# Coordinates:
#
#     background → (10, 10)
#     red        → (80, 80)
#     green      → (80, 180)
#     blue       → (180, 80)
#     white      → (180, 180)

background_pixel = image[10, 10]
red_pixel = image[80, 80]
green_pixel = image[80, 180]
blue_pixel = image[180, 80]
white_pixel = image[180, 180]

print()
print("=" * 70)
print("PIXEL VALUES")
print("=" * 70)

print(f"Background: {background_pixel}")
print(f"Red:        {red_pixel}")
print(f"Green:      {green_pixel}")
print(f"Blue:       {blue_pixel}")
print(f"White:      {white_pixel}")


# ======================================================================
# 8. VERIFY THE PIXELS
# ======================================================================

# We do not simply assume that our image construction worked.
#
# We explicitly test it.
#
# This habit will become extremely important later when we implement:
#
#     neural networks
#     backpropagation
#     diffusion
#     sampling
#     optimization

assert np.array_equal(
    background_pixel,
    np.array([0, 0, 0], dtype=np.uint8)
)

assert np.array_equal(
    red_pixel,
    np.array([255, 0, 0], dtype=np.uint8)
)

assert np.array_equal(
    green_pixel,
    np.array([0, 255, 0], dtype=np.uint8)
)

assert np.array_equal(
    blue_pixel,
    np.array([0, 0, 255], dtype=np.uint8)
)

assert np.array_equal(
    white_pixel,
    np.array([255, 255, 255], dtype=np.uint8)
)

print()
print("Pixel verification: PASSED")


# ======================================================================
# 9. NORMALIZE THE IMAGE
# ======================================================================

# Convert from:
#
#     uint8
#     0 ... 255
#
# to:
#
#     float32
#     0.0 ... 1.0

normalized_image = image.astype(np.float32) / 255.0

print()
print("=" * 70)
print("NORMALIZED IMAGE")
print("=" * 70)

print(f"Data type:       {normalized_image.dtype}")
print(f"Minimum:         {normalized_image.min()}")
print(f"Maximum:         {normalized_image.max()}")
print(f"Red pixel:       {normalized_image[80, 80]}")


# ======================================================================
# 10. VERIFY NORMALIZATION
# ======================================================================

assert normalized_image.dtype == np.float32

assert normalized_image.min() >= 0.0

assert normalized_image.max() <= 1.0

assert np.allclose(
    normalized_image[80, 80],
    np.array([1.0, 0.0, 0.0])
)

print("Normalization verification: PASSED")


# ======================================================================
# 11. SEPARATE RGB CHANNELS
# ======================================================================

# An RGB image is actually three 2D numerical matrices.
#
#     Red channel
#     Green channel
#     Blue channel
#
# Each channel has shape:
#
#     HEIGHT × WIDTH
#
# This becomes extremely important when we study convolution.

red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

print()
print("=" * 70)
print("CHANNEL SHAPES")
print("=" * 70)

print(f"Red channel:      {red_channel.shape}")
print(f"Green channel:    {green_channel.shape}")
print(f"Blue channel:     {blue_channel.shape}")


# ======================================================================
# 12. SIMPLE REAL-WORLD RED OBJECT DETECTION
# ======================================================================

"""
REAL-WORLD COMPUTER VISION TASK
-------------------------------

Imagine this is a frame from a robot camera.

The robot needs to identify a red object.

A simple classical vision strategy is:

    red_value is high
    green_value is low
    blue_value is low

For our synthetic image we can define:

    R > 200
    G < 50
    B < 50

This produces a binary mask:

    True  = red pixel
    False = not red pixel

This is a very simple form of image segmentation.

Modern systems can learn much more complicated segmentation rules,
but the fundamental idea is the same:

    IMAGE
      ↓
    PIXEL ANALYSIS
      ↓
    REGION OF INTEREST
"""

red_mask = (
    (image[:, :, 0] > 200)
    &
    (image[:, :, 1] < 50)
    &
    (image[:, :, 2] < 50)
)


# ======================================================================
# 13. COUNT DETECTED RED PIXELS
# ======================================================================

red_pixel_count = np.sum(red_mask)

print()
print("=" * 70)
print("RED OBJECT DETECTION")
print("=" * 70)

print(f"Detected red pixels: {red_pixel_count}")


# ======================================================================
# 14. FIND RED OBJECT COORDINATES
# ======================================================================

# np.where returns the coordinates where the mask is True.

red_y, red_x = np.where(red_mask)

print(f"Minimum X: {red_x.min()}")
print(f"Maximum X: {red_x.max()}")
print(f"Minimum Y: {red_y.min()}")
print(f"Maximum Y: {red_y.max()}")


# ======================================================================
# 15. CALCULATE BOUNDING BOX
# ======================================================================

# A bounding box is a common computer-vision representation.
#
# We calculate:
#
#     x_min
#     y_min
#     x_max
#     y_max

x_min = int(red_x.min())
x_max = int(red_x.max())

y_min = int(red_y.min())
y_max = int(red_y.max())

print()
print(f"Bounding box:")
print(f"    x_min = {x_min}")
print(f"    y_min = {y_min}")
print(f"    x_max = {x_max}")
print(f"    y_max = {y_max}")


# ======================================================================
# 16. CALCULATE OBJECT CENTER
# ======================================================================

# The center of the detected region is approximately:
#
#     center_x = (x_min + x_max) / 2
#     center_y = (y_min + y_max) / 2

center_x = (x_min + x_max) / 2.0
center_y = (y_min + y_max) / 2.0

print()
print(f"Detected object center:")
print(f"    X = {center_x:.2f}")
print(f"    Y = {center_y:.2f}")


# ======================================================================
# 17. VERIFY THE REAL-WORLD DETECTION
# ======================================================================

# We know approximately where we placed the red object.
#
# The rectangle was:
#
#     x = 40 ... 119
#     y = 50 ... 119
#
# Therefore the detected bounding box should correspond to that region.

assert x_min == 40
assert x_max == 119
assert y_min == 50
assert y_max == 119

expected_pixel_count = 80 * 70

assert red_pixel_count == expected_pixel_count

assert np.isclose(
    center_x,
    (40 + 119) / 2.0
)

assert np.isclose(
    center_y,
    (50 + 119) / 2.0
)

print()
print("Real-world red-object detection: PASSED")


# ======================================================================
# 18. VISUALIZE THE ORIGINAL IMAGE
# ======================================================================

plt.figure(figsize=(7, 7))

plt.imshow(image)

plt.title(
    "Original RGB Camera Frame"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ======================================================================
# 19. VISUALIZE THE RED MASK
# ======================================================================

plt.figure(figsize=(7, 7))

plt.imshow(
    red_mask,
    cmap="gray"
)

plt.title(
    "Detected Red-Pixel Mask"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ======================================================================
# 20. VISUALIZE THE DETECTED OBJECT
# ======================================================================

plt.figure(figsize=(7, 7))

plt.imshow(image)

# Draw the bounding box.
#
# We use matplotlib plotting primitives instead of OpenCV because
# the purpose of this lesson is to understand the underlying data.

plt.plot(
    [
        x_min,
        x_max,
        x_max,
        x_min,
        x_min
    ],
    [
        y_min,
        y_min,
        y_max,
        y_max,
        y_min
    ],
    linewidth=3
)

# Draw the estimated center.

plt.scatter(
    center_x,
    center_y,
    s=100
)

plt.title(
    "Real-World Example: Red Object Detection"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ======================================================================
# 21. DISPLAY THE THREE COLOR CHANNELS
# ======================================================================

# Seeing the individual channels is important.
#
# The original RGB image can be decomposed into:
#
#     R(x, y)
#     G(x, y)
#     B(x, y)
#
# These are simply three separate numerical images.

plt.figure(figsize=(6, 6))

plt.imshow(
    red_channel,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Red Channel")

plt.axis("off")

plt.tight_layout()

plt.show()


plt.figure(figsize=(6, 6))

plt.imshow(
    green_channel,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Green Channel")

plt.axis("off")

plt.tight_layout()

plt.show()


plt.figure(figsize=(6, 6))

plt.imshow(
    blue_channel,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Blue Channel")

plt.axis("off")

plt.tight_layout()

plt.show()


# ======================================================================
# 22. IMAGE STATISTICS
# ======================================================================

# Calculate basic statistics.
#
# These will become useful later when monitoring neural-network
# activations, gradients, latent representations, and diffusion
# variables.

print()
print("=" * 70)
print("IMAGE STATISTICS")
print("=" * 70)

print(
    f"Global mean:       "
    f"{normalized_image.mean():.6f}"
)

print(
    f"Global std:        "
    f"{normalized_image.std():.6f}"
)

print(
    f"Global minimum:    "
    f"{normalized_image.min():.6f}"
)

print(
    f"Global maximum:    "
    f"{normalized_image.max():.6f}"
)


# ======================================================================
# 23. FINAL STRUCTURAL TESTS
# ======================================================================

# Verify the complete image representation.

assert image.ndim == 3

assert image.shape == (
    HEIGHT,
    WIDTH,
    CHANNELS
)

assert CHANNELS == 3

assert image.dtype == np.uint8

assert normalized_image.shape == image.shape

assert normalized_image.dtype == np.float32

assert normalized_image.min() >= 0.0

assert normalized_image.max() <= 1.0

print()
print("=" * 70)
print("FINAL TESTS")
print("=" * 70)

print("3D image representation:     PASSED")
print("H × W × 3 shape:             PASSED")
print("RGB channel extraction:       PASSED")
print("Pixel inspection:             PASSED")
print("Normalization:                PASSED")
print("Red segmentation:             PASSED")
print("Bounding box calculation:     PASSED")
print("Object center calculation:    PASSED")


# ======================================================================
# 24. LESSON RESULT
# ======================================================================

"""
======================================================================
LESSON RESULT
======================================================================

We have completed the first computer-vision pipeline.

We started with nothing except numerical arrays.

We constructed:

                    RGB IMAGE
                        ↓
                 PIXEL VALUES
                        ↓
                RGB CHANNELS
                        ↓
                 NORMALIZATION
                        ↓
                PIXEL ANALYSIS
                        ↓
                RED SEGMENTATION
                        ↓
                 BOUNDING BOX
                        ↓
                 OBJECT CENTER


THE IMPORTANT CONCEPT
---------------------

A computer does not fundamentally receive:

                    "a red object"

It receives:

                    numbers.

Our first vision algorithm therefore directly operated on those
numbers.

This is the foundation of every later computer-vision model.


REAL-WORLD RESULT
-----------------

We simulated a basic robot-camera task:

                    Camera
                       ↓
                   RGB frame
                       ↓
                 pixel analysis
                       ↓
                red object mask
                       ↓
                object bounding box
                       ↓
                 object center


In a real robot, the input would come from a physical camera rather
than our synthetic image.

The downstream system could then use the detected position for tasks
such as:

    • object tracking
    • robotic grasping
    • sorting
    • inspection
    • navigation
    • visual servoing

For example:

                    Camera
                       ↓
                detect red object
                       ↓
              estimate image center
                       ↓
               robot controller
                       ↓
                 move toward object


LIMITATION OF THIS METHOD
-------------------------

Our algorithm only understands a very simple rule:

    "red pixels = object"

It does NOT understand:

    object shape
    object identity
    lighting variation
    shadows
    texture
    occlusion
    perspective
    background complexity

This is exactly why we eventually need learned computer-vision
models.

A CNN can learn useful visual features rather than relying only on
hand-written rules.


WHAT WE HAVE ACTUALLY LEARNED
-----------------------------

We now understand:

    • pixel
    • RGB
    • image tensor
    • channel
    • H × W × C
    • normalization
    • binary mask
    • bounding box
    • object center
    • basic image statistics


CONNECTION TO THE NEXT LESSON
-----------------------------

In this lesson we manipulated individual pixels.

That is useful, but modern vision systems require millions of
numerical operations.

The next question is:

    How do we mathematically manipulate large collections
    of numbers efficiently?

That leads directly to:

                    VECTORS
                       ↓
                    MATRICES
                       ↓
                    TENSORS
                       ↓
             MATRIX MULTIPLICATION
                       ↓
                 NEURAL NETWORKS


NEXT LESSON
-----------

LESSON 02 — TENSORS, VECTORS, AND MATRIX OPERATIONS

We will build the mathematical machinery that eventually becomes:

    Linear Layer
        ↓
    Neural Network
        ↓
    CNN
        ↓
    Attention
        ↓
    Diffusion Model


======================================================================
LESSON 01 COMPLETE
======================================================================
"""

print()
print("=" * 70)
print("COMPUTER VISION ADVANCED CRASH COURSE")
print("LESSON 01 COMPLETE")
print("=" * 70)
print()
print("PURPOSE:       UNDERSTAND IMAGES AS NUMERICAL DATA")
print("REAL RESULT:   DETECTED A RED OBJECT FROM PIXEL VALUES")
print("NEXT LESSON:   TENSORS, VECTORS, AND MATRIX OPERATIONS")
print()