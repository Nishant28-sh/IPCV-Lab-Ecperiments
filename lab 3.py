"""
Experiment 5 (Unit-2): Noise Models + Restoration using Spatial Filtering
-----------------------------------------------------------------------
AIM:
1) Understand what "noise" is in an image.
2) Generate two common noise types:
   (a) Gaussian noise (grainy noise everywhere)
   (b) Salt-and-pepper noise (random black/white dots)
3) Restore (denoise) the noisy image using spatial filters:
   - Mean filter (box blur)
   - Gaussian filter
   - Median filter (best for salt-and-pepper)

WHY THIS EXPERIMENT MATTERS (non-technical explanation):
- When you take photos in low light, images become grainy => Gaussian-like noise.
- When a scanner/cable has errors, you get black/white dots => salt-and-pepper.
- Restoration means "cleaning" the noisy image while trying to keep details.

REQUIREMENTS:
pip install opencv-python numpy matplotlib

RUN:
python exp/exp05_noise_models_spatial_restoration.py --input "C:\\Users\\Dr. Mansi Kajal\\Documents\\h.jpg"
(or you can keep input image inside ipcv-lab/images/input.jpg and run without --input)
"""

# ------------------------- IMPORTS -------------------------
# Pathlib Path: helps manage file/folder paths in a safe OS-independent way
from pathlib import Path

# cv2: OpenCV library for reading/writing images, filtering, etc.
import cv2

# numpy: numerical library (images become arrays, noise generation uses numpy)
import numpy as np

# matplotlib: optional display of results in a nice grid (not compulsory)
import matplotlib.pyplot as plt

# argparse: lets us pass input image path from terminal (--input)
import argparse


# ------------------------- HELPER: Find input in images/ folder -------------------------
def find_input_image(images_dir: Path, stem: str = "input") -> Path:
    """
    This function searches for an input image inside the 'images' folder.
    It tries common extensions like input.jpg, input.png, input.jpeg, etc.

    images_dir: Path object pointing to the "images" folder
    stem: base filename (default: "input")

    returns: Path to the found image
    """
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():           # .exists() checks if file is present
            return p

    # If exact extension not found, try any file with the same stem
    matches = sorted(images_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0]

    # If nothing found, raise error
    raise FileNotFoundError(f"No input image found in {images_dir} with name '{stem}.*'")


# ------------------------- METRICS: MSE and PSNR -------------------------
def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Mean Squared Error:
    Measures average squared difference between two images.
    - Lower MSE means images are more similar.
    """
    # Convert to float so subtraction doesn't overflow (uint8 can wrap)
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) in dB:
    Higher PSNR => better restoration quality (closer to original)

    Formula:
    PSNR = 10 * log10( (MAX^2) / MSE )
    """
    m = mse(img1, img2)
    if m == 0:
        return float("inf")  # Perfect match
    return 10.0 * np.log10((max_val * max_val) / m)


# ------------------------- NOISE MODEL 1: Gaussian Noise -------------------------
def add_gaussian_noise(gray: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """
    Adds Gaussian noise to a grayscale image.
    - Gaussian noise looks like grain across the image.
    - sigma controls noise strength:
        small sigma = little noise
        large sigma = heavy noise

    Steps:
    1) Generate random values from normal distribution N(0, sigma^2)
    2) Add to image
    3) Clip into valid range 0..255
    """
    # Create random noise array same size as image (float)
    noise = np.random.normal(loc=0.0, scale=sigma, size=gray.shape).astype(np.float32)

    # Add noise to image (convert image to float first)
    noisy = gray.astype(np.float32) + noise

    # Clip values to [0, 255] because images cannot go outside this range
    noisy = np.clip(noisy, 0, 255)

    # Convert back to uint8 (standard image datatype)
    return noisy.astype(np.uint8)


# ------------------------- NOISE MODEL 2: Salt & Pepper -------------------------
def add_salt_pepper_noise(gray: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """
    Adds Salt-and-Pepper noise:
    - 'Salt' = white dots (255)
    - 'Pepper' = black dots (0)

    amount = fraction of pixels to corrupt
    Example: amount=0.02 means 2% pixels will be changed.

    Steps:
    1) Create random matrix between 0..1
    2) If random < amount/2 => pepper
    3) If random > 1 - amount/2 => salt
    """
    noisy = gray.copy()

    # Random values in [0,1] for every pixel
    r = np.random.rand(*gray.shape)

    # Pepper: set to 0 where r is very small
    noisy[r < (amount / 2.0)] = 0

    # Salt: set to 255 where r is very large
    noisy[r > (1.0 - amount / 2.0)] = 255

    return noisy


# ------------------------- RESTORATION FILTERS -------------------------
def mean_filter(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Mean filter (box blur):
    - Replace each pixel by average of neighbors.
    - Good for mild Gaussian noise, but blurs edges.
    """
    return cv2.blur(gray, (ksize, ksize))


def gaussian_filter(gray: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian filter:
    - Weighted average (center has higher weight)
    - Usually better than mean filter for Gaussian noise.
    """
    return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)


def median_filter(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Median filter:
    - Replace each pixel by the median of neighborhood.
    - BEST for salt-and-pepper noise because median ignores extreme outliers (0 or 255 dots).
    """
    return cv2.medianBlur(gray, ksize)


# ------------------------- MAIN PROGRAM -------------------------
def main():
    # ---- Parse command-line argument for input image ----
    parser = argparse.ArgumentParser(description="Exp-5: Noise Models + Spatial Restoration")
    parser.add_argument("--input", type=str, default=None,
                        help="Full path to input image (optional). If not given, uses images/input.*")
    args = parser.parse_args()

    # ---- Set project folders ----
    images_dir = Path("Images")        # input folder in project
    out_dir = Path("lab-3-output")          # output folder in project
    out_dir.mkdir(exist_ok=True)       # create outputs/ if not exists

    # ---- Decide input path ----
    if args.input:
        input_path = Path(args.input)  # use user-provided path
    else:
        input_path = find_input_image(images_dir, "website")  # auto-find Images/website.*

    # ---- Read image as grayscale ----
    # cv2.IMREAD_GRAYSCALE ensures we work with single-channel intensity (0..255)
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

    # If OpenCV fails to read, it returns None
    if gray is None:
        raise FileNotFoundError(f"OpenCV could not read input image: {input_path}")

    print("=== Experiment 5: Noise Models + Spatial Restoration ===")
    print("Input:", input_path)
    print("Image size (H x W):", gray.shape[0], "x", gray.shape[1])

    # ----------------- Step 1: Create noisy images -----------------
    # Gaussian noise image (grainy)
    g_noisy = add_gaussian_noise(gray, sigma=20.0)

    # Salt & pepper noise image (dots)
    sp_noisy = add_salt_pepper_noise(gray, amount=0.03)

    # Save noisy outputs
    cv2.imwrite(str(out_dir / "exp05_gaussian_noisy.png"), g_noisy)
    cv2.imwrite(str(out_dir / "exp05_saltpepper_noisy.png"), sp_noisy)

    # ----------------- Step 2: Restore Gaussian noisy image -----------------
    g_mean = mean_filter(g_noisy, ksize=3)
    g_gauss = gaussian_filter(g_noisy, ksize=5, sigma=1.0)
    g_median = median_filter(g_noisy, ksize=3)

    # Save restored Gaussian outputs
    cv2.imwrite(str(out_dir / "exp05_gaussian_restored_mean.png"), g_mean)
    cv2.imwrite(str(out_dir / "exp05_gaussian_restored_gaussian.png"), g_gauss)
    cv2.imwrite(str(out_dir / "exp05_gaussian_restored_median.png"), g_median)

    # ----------------- Step 3: Restore Salt-Pepper noisy image -----------------
    sp_mean = mean_filter(sp_noisy, ksize=3)
    sp_gauss = gaussian_filter(sp_noisy, ksize=5, sigma=1.0)
    sp_median = median_filter(sp_noisy, ksize=3)

    # Save restored Salt-Pepper outputs
    cv2.imwrite(str(out_dir / "exp05_sp_restored_mean.png"), sp_mean)
    cv2.imwrite(str(out_dir / "exp05_sp_restored_gaussian.png"), sp_gauss)
    cv2.imwrite(str(out_dir / "exp05_sp_restored_median.png"), sp_median)

    # ----------------- Step 4: Compute quality metrics (PSNR) -----------------
    # We compare restored images against original 'gray'
    print("\n--- PSNR Results (Higher is better) ---")

    print("Gaussian noisy vs original:", round(psnr(gray, g_noisy), 2), "dB")
    print("Gaussian mean restored:", round(psnr(gray, g_mean), 2), "dB")
    print("Gaussian gaussian restored:", round(psnr(gray, g_gauss), 2), "dB")
    print("Gaussian median restored:", round(psnr(gray, g_median), 2), "dB")

    print("\nSalt-pepper noisy vs original:", round(psnr(gray, sp_noisy), 2), "dB")
    print("Salt-pepper mean restored:", round(psnr(gray, sp_mean), 2), "dB")
    print("Salt-pepper gaussian restored:", round(psnr(gray, sp_gauss), 2), "dB")
    print("Salt-pepper median restored:", round(psnr(gray, sp_median), 2), "dB")

    # ----------------- Step 5: Display results in one figure -----------------
    # Matplotlib is used only for viewing; output images already saved
    plt.figure(figsize=(12, 8))

    # Row 1: Original + noisy versions
    plt.subplot(2, 4, 1); plt.imshow(gray, cmap="gray"); plt.title("Original"); plt.axis("off")
    plt.subplot(2, 4, 2); plt.imshow(g_noisy, cmap="gray"); plt.title("Gaussian Noisy"); plt.axis("off")
    plt.subplot(2, 4, 3); plt.imshow(sp_noisy, cmap="gray"); plt.title("Salt-Pepper Noisy"); plt.axis("off")
    plt.subplot(2, 4, 4); plt.axis("off"); plt.title("")

    # Row 2: Best restoration choices
    plt.subplot(2, 4, 5); plt.imshow(g_gauss, cmap="gray"); plt.title("Gaussian Restored (Gaussian)"); plt.axis("off")
    plt.subplot(2, 4, 6); plt.imshow(g_mean, cmap="gray"); plt.title("Gaussian Restored (Mean)"); plt.axis("off")
    plt.subplot(2, 4, 7); plt.imshow(sp_median, cmap="gray"); plt.title("Salt-Pepper Restored (Median)"); plt.axis("off")
    plt.subplot(2, 4, 8); plt.imshow(sp_gauss, cmap="gray"); plt.title("Salt-Pepper Restored (Gaussian)"); plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("\nSaved outputs in:", out_dir.resolve())


# Standard Python entry point:
# If we run this file directly, main() executes.
# If we import this file as a module, main() does not auto-run.
if __name__ == "__main__":
    main()