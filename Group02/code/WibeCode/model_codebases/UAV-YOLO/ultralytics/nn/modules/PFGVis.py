import argparse
from pathlib import Path

import cv2
import torch

try:
    from .PhysicalFeatureGuidance import ExtractPhysicalFeature
except ImportError:
    from PhysicalFeatureGuidance import ExtractPhysicalFeature


def load_image_as_tensor(image_path: Path) -> torch.Tensor:
    """Load an image and convert it to a BCHW float tensor in [0, 1]."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def tensor_to_uint8_image(tensor: torch.Tensor):
    """Normalize tensor for visualization and convert to OpenCV BGR image."""
    tensor = tensor.detach().cpu().squeeze(0).clamp(min=0)

    min_val = tensor.min()
    max_val = tensor.max()
    if float(max_val - min_val) < 1e-12:
        tensor = torch.zeros_like(tensor)
    else:
        tensor = (tensor - min_val) / (max_val - min_val)

    image_rgb = (tensor.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def run_high_pass_visualization(image_path: str, output_path: str, cutoff_ratio: float = 0.1) -> None:
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_tensor = load_image_as_tensor(image_path)
    model = ExtractPhysicalFeature(cutoff_ratio=cutoff_ratio)

    with torch.no_grad():
        filtered = model(image_tensor)

    result_image = tensor_to_uint8_image(filtered)
    if not cv2.imwrite(str(output_path), result_image):
        raise RuntimeError(f"Failed to save image: {output_path}")

    print(f"Input image: {image_path}")
    print(f"Output image: {output_path}")
    print(f"cutoff_ratio: {cutoff_ratio}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply FFT high-pass filtering to a single image and save the result."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output image path.",
    )
    parser.add_argument(
        "--cutoff-ratio",
        type=float,
        default=0.1,
        help="High-pass cutoff ratio. Default: 0.1",
    )
    return parser


def main() -> None:
    run_high_pass_visualization(
        image_path="D:/AASchool/Computer Sience/ComputerVision/AAMAIN/datasets/VisDrone2019/images/val/0000369_00500_d_0000242.jpg",
        output_path=r"D:\high_pass_0000014.jpg",
        cutoff_ratio=0.1,
    )


if __name__ == "__main__":
    main()
