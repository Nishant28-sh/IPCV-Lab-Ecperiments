# ============================================================
# Experiment 2: Sampling & Quantization
# ============================================================
# This program demonstrates two key operations in digital images:
# 1) Sampling  : changing SPATIAL resolution (number of pixels)
# 2) Quantization : changing INTENSITY resolution (number of levels)
# ============================================================

# Pathlib is used for clean, OS-independent path handling.
# Why needed?
# - It avoids mistakes with slashes: \ vs /
# - It lets us write paths like: Path("images") / "input.jpg"
from pathlib import Path

# OpenCV (cv2) is used for:
# - Reading images from disk (cv2.imread)
# - Writing images to disk (cv2.imwrite)
# - Color conversion (cv2.cvtColor)
# - Resizing images (cv2.resize)
import cv2

# NumPy is used because images in OpenCV are stored as NumPy arrays.
# We use NumPy for:
# - Array slicing (sampling/downsampling)
# - Bitwise operations (quantization)
# - Type hints (np.ndarray)
import numpy as np

# Matplotlib is used for displaying images easily in VS Code.
# We use it instead of cv2.imshow() because matplotlib works reliably.
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Function 1: Find the input image file automatically
# ------------------------------------------------------------
def find_input_image(images_dir: Path, stem: str = "input") -> Path:
    """
    images_dir : Path object pointing to the images folder (e.g., Path("images"))
    stem       : base filename without extension, default is "input"
                 Example: if file is input.jpeg -> stem="input"

    Why this function?
    - In Windows, file extensions are often hidden.
    - Your file could be input.jpg, input.jpeg, input.png, etc.
    - This function finds the correct one automatically.
    """

    # Common extensions that usually store images
    # Why list needed?
    # Because we want to check multiple possible formats.
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    # Loop through every extension to check if a file exists
    for ext in exts:
        # Construct file path like: images/input.jpg or images/input.jpeg
        p = images_dir / f"{stem}{ext}"

        # p.exists() checks if this file is actually present on disk
        if p.exists():
            return p  # return the first valid file found

    # If not found in common list, try matching any extension using glob:
    # glob("input.*") means: any file starting with "input."
    matches = sorted(images_dir.glob(f"{stem}.*"))

    # If we found at least one file, return it
    if matches:
        return matches[0]

    # If nothing is found, stop program and show a clear error.
    raise FileNotFoundError(
        f"No input image found in {images_dir} with name '{stem}.*'"
    )


# ------------------------------------------------------------
# Function 2: Downsample (Sampling)
# ------------------------------------------------------------
def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """
    img    : image as NumPy array (H x W x C)
    factor : downsampling factor (2, 4, 8, ...)

    What is done?
    - Pixel decimation: keep every factor-th pixel
    - This reduces resolution:
      if factor = 2 => take rows 0,2,4,... and cols 0,2,4,...
      if factor = 4 => take rows 0,4,8,... and cols 0,4,8,...

    Why slicing works?
    - img[::factor, ::factor] means:
      ::factor in rows  -> step through rows by 'factor'
      ::factor in cols  -> step through columns by 'factor'
    """

    # Return reduced image (smaller H and W)
    return img[::factor, ::factor]


# ------------------------------------------------------------
# Function 3: Upsample back to original size (for comparison)
# ------------------------------------------------------------
def upsample_to_original(img_small: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    img_small : downsampled image (smaller size)
    target_w  : original width
    target_h  : original height

    Why this function?
    - After downsampling, the image becomes smaller.
    - For fair visualization side-by-side with original,
      we resize it back to original display size.

    How resizing works?
    - cv2.resize(image, (width, height))
    - interpolation=cv2.INTER_NEAREST:
        * Nearest-neighbor interpolation
        * This keeps blocky pixel look (clearly shows sampling effect)
        * If we used linear/cubic, it would smooth and hide sampling artifacts
    """

    return cv2.resize(
        img_small,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )


# ------------------------------------------------------------
# Function 4: Quantization (Bit depth reduction)
# ------------------------------------------------------------
def quantize_bits(img: np.ndarray, bits: int) -> np.ndarray:
    """
    img  : 8-bit image (uint8), values 0..255 per channel
    bits : number of bits to keep (1..8)

    What is quantization here?
    - Reduce intensity levels.
    - 8-bit => 256 levels (0..255)
    - 4-bit => 16 levels
    - 2-bit => 4 levels

    How do we implement?
    - Using bit truncation:
      Keep the top 'bits' and remove (zero) the lower bits.

    Example:
    Suppose a pixel value is 201 (binary: 11001001)
    If bits=4 => keep top 4 bits: 1100xxxx -> 11000000 (192)
    So value becomes one of 16 possible levels.
    """

    # Safety check: bits must be valid
    if not (1 <= bits <= 8):
        raise ValueError("bits must be between 1 and 8")

    # shift tells how many lower bits to remove
    # bits=6 => shift=2 (remove 2 lower bits)
    # bits=4 => shift=4 (remove 4 lower bits)
    # bits=2 => shift=6 (remove 6 lower bits)
    shift = 8 - bits

    # Bitwise operations:
    # img >> shift  : move bits right -> removes lower bits (like division by 2^shift)
    # (img >> shift) << shift : put it back left -> lower bits become 0
    return (img >> shift) << shift


# ------------------------------------------------------------
# MAIN function: executes the experiment
# ------------------------------------------------------------
def main():
    # images_dir is the folder where input image is stored
    images_dir = Path("Images")

    # out_dir is where outputs will be saved
    out_dir = Path("lab-1-2-output")

    # Create outputs folder if it doesn't exist
    out_dir.mkdir(exist_ok=True)

    # Find the actual file path of input image automatically
    # Find the input image by stem only; the helper will resolve the extension.
    input_path = find_input_image(images_dir, "website")

    # Read input image
    # cv2.imread returns image as BGR NumPy array
    img_bgr = cv2.imread(str(input_path))

    # If reading fails, img_bgr becomes None
    if img_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read: {input_path}")

    # Get original height and width
    h, w = img_bgr.shape[:2]

    # Print experiment header and image info
    print("=== Experiment 2: Sampling & Quantization ===")
    print("Input file:", input_path)
    print("Original size (W x H):", w, "x", h)

    # ========================================================
    # Part A: Sampling (Downsampling)
    # ========================================================

    # Sampling factors we want to test
    sampling_factors = [2, 4, 8]

    # We'll store results as a list of tuples for display later
    sampled_results = []

    # Loop over each sampling factor
    for f in sampling_factors:
        # Downsample image (reduce resolution)
        ds = downsample(img_bgr, f)

        # Upsample back for fair display comparison
        ds_up = upsample_to_original(ds, w, h)

        # Store (factor, small_image, display_image)
        sampled_results.append((f, ds, ds_up))

        # Save:
        # 1) actual downsampled small image
        cv2.imwrite(str(out_dir / f"exp02_downsample_x{f}_small.png"), ds)

        # 2) upsampled image for visualization
        cv2.imwrite(str(out_dir / f"exp02_downsample_x{f}_upsampled.png"), ds_up)

    # ========================================================
    # Part B: Quantization (Bit reduction)
    # ========================================================

    # Bit levels we want to test
    quant_bits = [6, 4, 2]

    # Store quantized outputs for display later
    quant_results = []

    for b in quant_bits:
        # Quantize to 'b' bits per channel
        q = quantize_bits(img_bgr, b)

        # Store (bits, quantized_image)
        quant_results.append((b, q))

        # Save quantized output image
        cv2.imwrite(str(out_dir / f"exp02_quantized_{b}bit.png"), q)

    # ========================================================
    # Display section using Matplotlib
    # ========================================================

    # Convert BGR to RGB because Matplotlib expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ------------------ Display 1: Sampling ------------------
    plt.figure(figsize=(10, 6))

    # subplot(rows, cols, index) -> 2x2 grid, first cell
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    # enumerate gives:
    # idx = subplot index, starting from 2
    # (f, _, ds_up) = factor, ignore small ds, show upsampled ds_up
    for idx, (f, _, ds_up) in enumerate(sampled_results, start=2):
        plt.subplot(2, 2, idx)
        plt.imshow(cv2.cvtColor(ds_up, cv2.COLOR_BGR2RGB))
        plt.title(f"Sampling x{f} (down+upsample)")
        plt.axis("off")

    # Automatically adjust spacing so titles/images don't overlap
    plt.tight_layout()
    plt.show()

    # ---------------- Display 2: Quantization ----------------
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original (8-bit)")
    plt.axis("off")

    # enumerate through quantized results
    for idx, (b, q) in enumerate(quant_results, start=2):
        plt.subplot(2, 2, idx)
        plt.imshow(cv2.cvtColor(q, cv2.COLOR_BGR2RGB))
        plt.title(f"Quantized to {b}-bit")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ========================================================
    # Print output filenames so user can verify saved results
    # ========================================================
    print("Saved outputs in:", out_dir.resolve())

    print("Sampling outputs:")
    for f in sampling_factors:
        print(" -", f"exp02_downsample_x{f}_small.png")
        print(" -", f"exp02_downsample_x{f}_upsampled.png")

    print("Quantization outputs:")
    for b in quant_bits:
        print(" -", f"exp02_quantized_{b}bit.png")


# This makes sure main() runs only when this file is executed directly.
if __name__ == "__main__":
    main()