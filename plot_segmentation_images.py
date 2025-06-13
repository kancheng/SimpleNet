# plot_segmentation_images.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
import PIL.Image as Image

# ---------- Plot function ----------
def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)

        f, axes = plt.subplots(1, 2 + int(masks_provided))
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[0].set_title("Image")

        if masks_provided:
            axes[1].imshow(mask.transpose(1, 2, 0))
            axes[1].set_title("GT Mask")

            axes[2].imshow(segmentation)
            axes[2].set_title(f"Segmentation\nScore={anomaly_score}")
        else:
            axes[1].imshow(segmentation)
            axes[1].set_title(f"Segmentation\nScore={anomaly_score}")

        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()

# ---------- Optional main() ----------
def main():
    parser = argparse.ArgumentParser(description="Plot Segmentation Images")
    parser.add_argument('--image_folder', type=str, required=True, help='Folder with original images')
    parser.add_argument('--segmentation_file', type=str, required=True, help='Path to npy file of segmentations')
    parser.add_argument('--savefolder', type=str, default='./vis', help='Folder to save output images')
    parser.add_argument('--mask_folder', type=str, default=None, help='Folder with GT masks (optional)')
    args = parser.parse_args()

    # Prepare image paths
    image_paths = sorted([os.path.join(args.image_folder, fname) for fname in os.listdir(args.image_folder) if fname.endswith(('.png', '.jpg'))])

    # Load segmentations
    segmentations = np.load(args.segmentation_file)  # List[np.ndarray] or np.array

    # Optional masks
    if args.mask_folder:
        mask_paths = sorted([os.path.join(args.mask_folder, fname) for fname in os.listdir(args.mask_folder) if fname.endswith(('.png', '.jpg'))])
    else:
        mask_paths = None

    # Example image transform
    def image_transform(x):
        x = np.array(x).transpose(2, 0, 1) / 255.
        return x

    # Example mask transform
    def mask_transform(x):
        x = np.array(x).transpose(2, 0, 1) / 255.
        return x

    # No anomaly_scores for now
    plot_segmentation_images(
        args.savefolder,
        image_paths,
        segmentations,
        anomaly_scores=None,
        mask_paths=mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
        save_depth=2,
    )

    print(f'Visualization saved to {args.savefolder}')


if __name__ == '__main__':
    main()

