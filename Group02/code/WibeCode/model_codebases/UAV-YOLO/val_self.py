import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import yaml
from PIL import Image
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the original Ultralytics val flow and then add COCO-style mAPs/mAPm/mAPl."
    )
    parser.add_argument(
        "--model",
        default="D:/Programs/Noise/train42/weights/best.pt",
        help="Path to the trained .pt file.",
    )
    parser.add_argument(
        "--data",
        default="D:/AASchool/Computer Sience/ComputerVision/AAMAIN/datasets/VisDrone2019/VisDrone2019.yaml",
        help="Path to the dataset yaml file.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size for model.val().")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for model.val().")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold for model.val().")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", default="", help="Device passed to Ultralytics, e.g. 0, cpu, 0,1.")
    parser.add_argument("--project", default="runs/detect", help="Project directory for Ultralytics validation outputs.")
    parser.add_argument("--name", default="val_with_size_metrics", help="Run name for Ultralytics validation outputs.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing the same save directory.")
    return parser.parse_args()


def load_yaml(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml content: {yaml_path}")
    return data


def resolve_split_dir(root: Path, split_value: Union[str, List[str]]) -> Path:
    if isinstance(split_value, list):
        if not split_value:
            raise ValueError("Split path list is empty.")
        split_value = split_value[0]
    split_path = Path(split_value)
    return split_path if split_path.is_absolute() else (root / split_path)


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
    if not images:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return images


def image_id_from_path(image_path: Path) -> Union[int, str]:
    stem = image_path.stem
    return int(stem) if stem.isnumeric() else stem


def yolo_label_to_coco_bbox(parts: List[str], width: int, height: int) -> Tuple[int, List[float], float]:
    cls_id = int(float(parts[0]))
    x_center = float(parts[1]) * width
    y_center = float(parts[2]) * height
    box_width = float(parts[3]) * width
    box_height = float(parts[4]) * height
    x_min = x_center - box_width / 2
    y_min = y_center - box_height / 2
    return cls_id, [x_min, y_min, box_width, box_height], box_width * box_height


def build_coco_gt(dataset_cfg: dict, yaml_path: Path, split: str, save_dir: Path) -> Path:
    root = Path(dataset_cfg["path"])
    image_dir = resolve_split_dir(root, dataset_cfg[split])
    label_dir = root / "labels" / split
    image_paths = list_images(image_dir)

    names = dataset_cfg["names"]
    if isinstance(names, dict):
        categories = [{"id": int(k) + 1, "name": v} for k, v in sorted(names.items(), key=lambda x: int(x[0]))]
    else:
        categories = [{"id": i + 1, "name": name} for i, name in enumerate(names)]

    coco = {
        "info": {"description": f"{yaml_path.stem}-{split}"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    annotation_id = 1
    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size

        image_id = image_id_from_path(image_path)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        with label_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id, bbox, area = yolo_label_to_coco_bbox(parts, width, height)
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": cls_id + 1,
                        "bbox": [round(x, 3) for x in bbox],
                        "area": round(area, 3),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    gt_json_path = save_dir / f"{yaml_path.stem}_{split}_gt.json"
    with gt_json_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f)
    return gt_json_path


def run_original_val(args) -> Tuple[object, Path]:
    model = YOLO(str(Path(args.model).resolve()))
    metrics = model.val(
        data=str(Path(args.data).resolve()),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device if args.device else None,
        save_json=True,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=False,
        verbose=True,
    )

    save_dir = Path(metrics.save_dir)
    pred_json_path = save_dir / "predictions.json"
    if not pred_json_path.exists():
        raise FileNotFoundError(
            f"Ultralytics val finished, but predictions.json was not found at: {pred_json_path}"
        )
    return metrics, pred_json_path


def evaluate_coco(gt_json_path: Path, pred_json_path: Path) -> Dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise ImportError("pycocotools is required. Install it first, then rerun this script.") from e

    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(pred_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP50-95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "mAP75": float(coco_eval.stats[2]),
        "mAPs": float(coco_eval.stats[3]),
        "mAPm": float(coco_eval.stats[4]),
        "mAPl": float(coco_eval.stats[5]),
        "AR1": float(coco_eval.stats[6]),
        "AR10": float(coco_eval.stats[7]),
        "AR100": float(coco_eval.stats[8]),
        "ARs": float(coco_eval.stats[9]),
        "ARm": float(coco_eval.stats[10]),
        "ARl": float(coco_eval.stats[11]),
    }


def get_original_metrics(metrics) -> Dict[str, float]:
    results = metrics.results_dict
    return {
        "precision": float(results.get("metrics/precision(B)", 0.0)),
        "recall": float(results.get("metrics/recall(B)", 0.0)),
        "mAP50": float(results.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(results.get("metrics/mAP50-95(B)", 0.0)),
    }


def main():
    args = parse_args()
    yaml_path = Path(args.data).resolve()
    dataset_cfg = load_yaml(yaml_path)

    metrics, pred_json_path = run_original_val(args)
    save_dir = Path(metrics.save_dir)
    gt_json_path = build_coco_gt(dataset_cfg, yaml_path, args.split, save_dir)
    size_metrics = evaluate_coco(gt_json_path, pred_json_path)
    original_metrics = get_original_metrics(metrics)

    print("\nOriginal Ultralytics val metrics")
    print(f"P          : {original_metrics['precision']:.4f}")
    print(f"R          : {original_metrics['recall']:.4f}")
    print(f"mAP50      : {original_metrics['mAP50']:.4f}")
    print(f"mAP50-95   : {original_metrics['mAP50-95']:.4f}")

    print("\nAdditional size metrics from the same predictions.json")
    print(f"mAPs       : {size_metrics['mAPs']:.4f}")
    print(f"mAPm       : {size_metrics['mAPm']:.4f}")
    print(f"mAPl       : {size_metrics['mAPl']:.4f}")

    summary_path = save_dir / "size_metrics_summary.json"
    summary = {
        "original_val": original_metrics,
        "size_metrics": size_metrics,
        "predictions_json": str(pred_json_path),
        "gt_json": str(gt_json_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
