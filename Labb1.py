from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def inputimg(images_dir: Path, stem: str = "input") -> Path:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    for ext in exts:
        p = images_dir / f"{stem}{ext}" #Build file path like:image/hi.jpg
        if p.exists():
            return p #If file exists → return that file immediately.
        
    matches = sorted(images_dir.glob(f"{stem}.*")) #not found above, search any file starting with the name

    if matches:
        return matches[0] #at least one match found → return first one
    raise FileNotFoundError("No matching image found.")


def main():
    images_dir = Path("Images")
    out_dir = Path("Labb-1-output")
    out_dir.mkdir(exist_ok=True)  # create output folder

    input_path = inputimg(images_dir, "website")  # locate input image

    img_bgr = cv2.imread(str(input_path))  # read image
    if img_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read: {input_path}")

    h, w = img_bgr.shape[:2]
    ch = 1 if img_bgr.ndim == 2 else img_bgr.shape[2]

    print("=== Experiment 1 ===")
    print("Input file :", input_path)
    print("Width x Height :", w, "x", h)
    print("Channels :", ch)
    print("Datatype :", img_bgr.dtype)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB

    plt.figure()
    plt.imshow(img_rgb)
    plt.title("Input Image")
    plt.axis("off")  # hide axis
    plt.show()

    out_original = out_dir / "exp01_original_copy.png"
    cv2.imwrite(str(out_original), img_bgr)  # save original

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # convert grayscale
    out_gray = out_dir / "exp01_gray.png"
    cv2.imwrite(str(out_gray), gray)  # save grayscale

    print("Saved outputs:")
    print(" -", out_original)
    print(" -", out_gray)


if __name__ == "__main__":
    main()