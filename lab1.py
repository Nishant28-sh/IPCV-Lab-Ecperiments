# ---------------------------------------------
# Experiment 1: Acquire and Display Image
# ---------------------------------------------
# Goal:
# 1) Read an image from disk (Acquire)
# 2) Display it on screen (Display)
# 3) Save a copy + grayscale version in outputs/
# ---------------------------------------------

# Path lets us write file paths in a clean + OS-safe way (Windows/Linux/Mac).
# Example: Path("images") / "input.jpg" becomes "images/input.jpg"
from pathlib import Path

# OpenCV (cv2) is the main library for image reading, processing, and saving.
# - cv2.imread() reads image from disk
# - cv2.imwrite() saves image to disk
# - cv2.cvtColor() converts between color spaces (BGR->RGB, BGR->GRAY, etc.)
import cv2

# Matplotlib is used to display images in a window/figure.
# We use matplotlib because cv2.imshow sometimes has GUI issues on some setups.
import matplotlib.pyplot as plt


def find_input_image(images_dir: Path, stem: str = "input") -> Path:
    """
    Purpose:
    - Your Windows hides file extensions sometimes (like .jpg, .jpeg).
    - So we don't want you to manually edit code every time.
    This function automatically searches for images like:
    images/input.jpg, images/input.jpeg, images/input.png, etc.

    Parameters:
    - images_dir: folder where input images are stored (Path object)
    - stem: base name of file (default = "input")

    Returns:
    - full path of the found image file
    """

    # Common image extensions we expect in labs
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    # Try each extension in order:
    # If images/input.jpeg exists, it returns that file immediately.
    for ext in exts:
        p = images_dir / f"{stem}{ext}"  # build path: images/input + ext
        if p.exists():                   # check if file exists on disk
            return p                     # return the first match

    # If not found in common extensions, try ANY file that starts with "input."
    # Example: input.xyz or input.anything (rare, but safe fallback)
    matches = sorted(images_dir.glob(f"{stem}.*"))

    # If we got at least one match, return the first one.
    if matches:
        return matches[0]

    # If nothing found, raise an error with helpful info.
    # This stops the program and tells you what's wrong.
    raise FileNotFoundError(
        f"No image found like {images_dir / (stem + '.jpg')} (or other extensions).\n"
        f"Files currently inside {images_dir}: {[p.name for p in images_dir.glob('*')]}"
    )


def main():
    # --------------------------
    # Step A: Define folders
    # --------------------------

    # "images" folder should contain input image(s)
    images_dir = Path("Images")

    # "outputs" folder is where we will save results
    out_dir = Path("lab-1-output")

    # Create outputs folder if it does not already exist.
    # If outputs already exists, no error occurs because exist_ok=True.
    out_dir.mkdir(exist_ok=True)

    # --------------------------
    # Step B: Find input image
    # --------------------------

    # Automatically locate images/input.(jpg/jpeg/png/...)
    input_path = find_input_image(images_dir, "website")

    # --------------------------
    # Step C: Read image
    # --------------------------

    # cv2.imread reads the image from disk.
    # IMPORTANT: OpenCV reads images in BGR order (Blue, Green, Red) NOT RGB.
    img_bgr = cv2.imread(str(input_path))

    # Sometimes cv2.imread returns None if:
    # - file path wrong
    # - file corrupted
    # - unsupported format
    # So we check and throw a clear error.
    if img_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read: {input_path}")

    # --------------------------
    # Step D: Print image info
    # --------------------------

    # img_bgr.shape gives dimensions:
    # If image is color: shape = (height, width, channels)
    # If image is grayscale: shape = (height, width)
    h, w = img_bgr.shape[:2]  # first two values are always height and width

    # Determine channels:
    # - If grayscale -> img_bgr.ndim == 2
    # - If color     -> img_bgr.ndim == 3 and channels = img_bgr.shape[2]
    ch = 1 if img_bgr.ndim == 2 else img_bgr.shape[2]

    # Print details (useful for lab record + debugging)
    print("=== Experiment 1: Acquire and Display Image ===")
    print("Input file :", input_path)
    print("Width x Height :", w, "x", h)
    print("Channels :", ch)
    print("Datatype :", img_bgr.dtype)  # uint8 is most common: 0-255 per channel

    # --------------------------
    # Step E: Save outputs
    # --------------------------

    # Save a copy of the original image to outputs folder.
    # We save in PNG to avoid JPEG compression artifacts.
    out_original = out_dir / "exp01_original_copy.png"

    # cv2.imwrite writes the image to disk.
    # It expects BGR format (we already have BGR), so directly save img_bgr.
    cv2.imwrite(str(out_original), img_bgr)

    # Convert BGR image to grayscale (single channel).
    # Why do this?
    # - Many image processing operations start with grayscale
    # - It reduces computation (1 channel instead of 3)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Save grayscale output
    out_gray = out_dir / "exp01_gray.png"
    cv2.imwrite(str(out_gray), gray)

    # Print saved file paths so you know where outputs are stored
    print("Saved outputs:")
    print(" -", out_original)
    print(" -", out_gray)

    # --------------------------
    # Step F: Display image
    # --------------------------

    # Matplotlib expects images in RGB order.
    # But OpenCV gives BGR, so we must convert BGR -> RGB.
    # If we don't do this conversion, colors will look wrong (blue/red swapped).
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Create a new figure (a window/canvas for plot)
    plt.figure()

    # Show the RGB image
    plt.imshow(img_rgb)

    # Title shown above the image
    plt.title("Experiment 1: Input Image")

    # Hide X and Y axis ticks for clean display
    plt.axis("off")

    # Display the plot window
    plt.show()


# This block ensures main() runs only when you execute this file directly.
# If you import this file in another script, main() won't automatically run.
if __name__ == "__main__":
    main()