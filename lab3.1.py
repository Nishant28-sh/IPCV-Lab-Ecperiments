"""
Experiment 6 (Unit-2): Image Degradation (Motion Blur) + Restoration
--------------------------------------------------------------------
AIM:
1) Create a blurred image using a motion blur PSF (degradation function).
2) Restore it using:
   (a) Inverse filtering (simple but noise-sensitive)
   (b) Wiener filtering (practical, controls noise amplification)

WHY THIS IS IMPORTANT (non-technical):
- Motion blur happens when camera shakes or object moves.
- Restoration tries to undo the blur.
- Inverse filter is like "hard undo" (can explode noise).
- Wiener filter is a smart undo that balances blur removal and noise.

REQUIREMENTS:
pip install opencv-python numpy matplotlib
there might be more things we need to install, but these are the main ones so that we have to first install them before running the code. and for later there might be more of these libraries and we will install them as we go along. the open cv library is used for image processing and manipulation, numpy is used for numerical operations and array handling, and matplotlib is used for visualization of images and results.
now we will run this code by providing the path to the input image, and we will see the results of the motion blur degradation and the restoration using both inverse and wiener filters. we will also see the PSNR values for each step to understand the quality of the restoration

after the wiener restoration we can see the its not as good as the original image but its better than the blurred image and the inverse restored image,

RUN:
python exp/exp06_deblur_inverse_wiener.py --input "C:\\Users\\Dr. Mansi Kajal\\Documents\\h.jpg"
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def find_input_image(images_dir: Path, stem: str = "input") -> Path:
    """Same helper as earlier: finds images/input.*"""
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    matches = sorted(images_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No input image found in {images_dir} with name '{stem}.*'")


# ------------------------- METRICS (PSNR) -------------------------
def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    m = mse(img1, img2)
    if m == 0:
        return float("inf")
    return 10.0 * np.log10((max_val * max_val) / m)


# ------------------------- PSF / MOTION BLUR KERNEL -------------------------
def motion_blur_psf(size: int = 21, angle_deg: float = 0.0) -> np.ndarray:
    """
    Create a simple motion blur PSF (Point Spread Function).

    size:
      kernel size (odd number is preferred: 15, 21, 31...)
      bigger => stronger blur

    angle_deg:
      direction of motion blur (0 = horizontal, 90 = vertical)

    HOW IT WORKS (simple):
    - We create an empty kernel (size x size)
    - Draw a bright line in the middle (represents motion path)
    - Rotate that line to the required angle
    - Normalize sum to 1 so brightness stays stable
    """
    # Start with zeros (black)
    psf = np.zeros((size, size), dtype=np.float32)

    # Draw a horizontal line at center row (white line)
    psf[size // 2, :] = 1.0

    # Rotate kernel around its center to create angled motion blur
    center = (size / 2, size / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    psf = cv2.warpAffine(psf, M, (size, size))

    # Normalize kernel so total sum = 1 (keeps brightness roughly same)
    psf_sum = psf.sum()
    if psf_sum != 0:
        psf /= psf_sum

    return psf


# ------------------------- FREQUENCY DOMAIN HELPERS -------------------------
def fft2(img: np.ndarray) -> np.ndarray:
    """2D FFT for image -> frequency domain."""
    return np.fft.fft2(img)


def ifft2(F: np.ndarray) -> np.ndarray:
    """Inverse 2D FFT -> back to spatial domain."""
    return np.fft.ifft2(F)


def inverse_filter(blurred: np.ndarray, psf: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Inverse filtering in frequency domain.

    Model (no noise case):
    G = F * H  =>  F = G / H

    Problem:
    If H is near zero, division becomes huge => noise amplification.
    So we use 'eps' to avoid divide-by-zero.

    blurred: degraded image (grayscale float32)
    psf: blur kernel
    eps: small constant for stability
    """
    H, W = blurred.shape

    # FFT of blurred image
    G = fft2(blurred)

    # FFT of PSF padded to image size using s=(H,W)
    H_uv = np.fft.fft2(psf, s=(H, W))

    # Avoid division by zero: if abs(H_uv) is too small, clamp it
    denom = np.where(np.abs(H_uv) < eps, eps, H_uv)

    # Inverse filter formula
    F_hat = G / denom

    # Back to image using inverse FFT
    f_hat = np.real(ifft2(F_hat))

    # Clip into [0,255]
    return np.clip(f_hat, 0, 255).astype(np.uint8)


def wiener_filter(blurred: np.ndarray, psf: np.ndarray, K: float = 0.01) -> np.ndarray:
    """
    Wiener filtering in frequency domain.

    Practical formula:
    F_hat = (H* / (|H|^2 + K)) * G

    Meaning (simple):
    - H* / |H|^2 acts like inverse filter
    - +K prevents exploding when |H| is small
    - K represents noise-to-signal balance:
        smaller K => more sharpening but more noise
        larger K => safer but less sharp

    blurred: degraded image (grayscale float32)
    psf: blur kernel
    K: noise control parameter
    """
    H, W = blurred.shape

    # FFT of blurred image
    G = fft2(blurred)

    # FFT of PSF
    H_uv = np.fft.fft2(psf, s=(H, W))

    # Conjugate of H (needed for Wiener formula)
    H_conj = np.conj(H_uv)

    # |H|^2 = H * H_conj
    H_mag2 = np.abs(H_uv) ** 2

    # Wiener filter transfer function
    W_uv = H_conj / (H_mag2 + K)

    # Apply Wiener filter
    F_hat = W_uv * G

    # Back to spatial domain
    f_hat = np.real(ifft2(F_hat))

    return np.clip(f_hat, 0, 255).astype(np.uint8)


# ------------------------- MAIN -------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp-6: Motion Blur + Inverse/Wiener Restoration")
    parser.add_argument("--input", type=str, default=None,
                        help="Full path to input image (optional). If not given, uses images/input.*")
    args = parser.parse_args()

    images_dir = Path("Images")
    out_dir = Path("lab-3-1-output")
    out_dir.mkdir(exist_ok=True)

    # Choose input image
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_input_image(images_dir, "website")

    # Read grayscale
    orig = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(f"OpenCV could not read: {input_path}")

    print("=== Experiment 6: Motion Blur Degradation + Restoration ===")
    print("Input:", input_path)
    print("Size (H x W):", orig.shape[0], "x", orig.shape[1])

    # ----------------- Step 1: Create PSF (motion blur kernel) -----------------
    psf = motion_blur_psf(size=21, angle_deg=20.0)

    # Save PSF for visualization (scaled to 0..255)
    psf_vis = (psf / psf.max() * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_dir / "exp06_psf.png"), psf_vis)

    # ----------------- Step 2: Create blurred image using convolution -----------------
    # cv2.filter2D applies the kernel to the image (spatial domain convolution)
    blurred = cv2.filter2D(orig, ddepth=-1, kernel=psf)

    # Optional: add slight Gaussian noise to show why Wiener works better
    sigma = 5.0
    noise = np.random.normal(0, sigma, orig.shape).astype(np.float32)
    blurred_noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Save degraded images
    cv2.imwrite(str(out_dir / "exp06_blurred.png"), blurred)
    cv2.imwrite(str(out_dir / "exp06_blurred_noisy.png"), blurred_noisy)

    # ----------------- Step 3: Restore using Inverse Filter -----------------
    inv_restored = inverse_filter(blurred_noisy.astype(np.float32), psf, eps=1e-3)
    cv2.imwrite(str(out_dir / "exp06_restored_inverse.png"), inv_restored)

    # ----------------- Step 4: Restore using Wiener Filter -----------------
    wien_restored = wiener_filter(blurred_noisy.astype(np.float32), psf, K=0.01)
    cv2.imwrite(str(out_dir / "exp06_restored_wiener.png"), wien_restored)

    # ----------------- Step 5: Metrics -----------------
    print("\n--- PSNR (Higher is better) ---")
    print("Blurred vs Original:", round(psnr(orig, blurred), 2), "dB")
    print("Blurred+Noise vs Original:", round(psnr(orig, blurred_noisy), 2), "dB")
    print("Inverse Restored vs Original:", round(psnr(orig, inv_restored), 2), "dB")
    print("Wiener Restored vs Original:", round(psnr(orig, wien_restored), 2), "dB")

    # ----------------- Step 6: Show results -----------------
    plt.figure(figsize=(12, 7))

    plt.subplot(2, 3, 1); plt.imshow(orig, cmap="gray"); plt.title("Original"); plt.axis("off")
    plt.subplot(2, 3, 2); plt.imshow(blurred, cmap="gray"); plt.title("Blurred (PSF)"); plt.axis("off")
    plt.subplot(2, 3, 3); plt.imshow(blurred_noisy, cmap="gray"); plt.title("Blurred + Noise"); plt.axis("off")

    plt.subplot(2, 3, 4); plt.imshow(psf_vis, cmap="gray"); plt.title("PSF (Motion Kernel)"); plt.axis("off")
    plt.subplot(2, 3, 5); plt.imshow(inv_restored, cmap="gray"); plt.title("Inverse Restored"); plt.axis("off")
    plt.subplot(2, 3, 6); plt.imshow(wien_restored, cmap="gray"); plt.title("Wiener Restored"); plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("\nSaved outputs in:", out_dir.resolve())


if __name__ == "__main__":
    main()