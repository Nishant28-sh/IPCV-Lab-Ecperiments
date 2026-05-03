# ============================================================
# EXPERIMENT 3: Contrast Stretching + Histogram Equalization
# ============================================================

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def contrast_stretch(gray: np.ndarray) -> np.ndarray:
    """
    Perform contrast stretching on a grayscale image
    """

    gmin = np.min(gray)
    gmax = np.max(gray)

    if gmax == gmin:
        return gray.copy()

    gray_f = gray.astype(np.float32)
    stretched = (gray_f - gmin) * 255.0 / (gmax - gmin)

    return stretched.astype(np.uint8)


def main():
    # Images directory
    images_dir = Path("Images")
    input_path = images_dir / "website.jpg"

    # Output directory
    out_dir = Path("lab-2-output")
    out_dir.mkdir(exist_ok=True)

    # Read image
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {input_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # A) Contrast Stretching
    stretched = contrast_stretch(gray)

    # B) Histogram Equalization
    hist_eq = cv2.equalizeHist(gray)

    # Save results
    cv2.imwrite(str(out_dir / "exp03_gray.png"), gray)
    cv2.imwrite(str(out_dir / "exp03_contrast_stretched.png"), stretched)
    cv2.imwrite(str(out_dir / "exp03_hist_equalized.png"), hist_eq)

    # Display results
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original Grayscale")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(stretched, cmap="gray")
    plt.title("Contrast Stretched")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(hist_eq, cmap="gray")
    plt.title("Histogram Equalized")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.hist(gray.ravel(), bins=256)
    plt.title("Histogram of Original Image")

    plt.tight_layout()
    plt.show()

    print("=== Experiment 3 Completed Successfully ===")
    print("Input Image:", input_path)
    print("Outputs saved in:", out_dir)


if __name__ == "__main__":
    main()
