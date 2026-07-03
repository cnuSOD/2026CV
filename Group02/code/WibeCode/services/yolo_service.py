"""
YOLO 推理服务模块

职责说明：
1. 根据用户选择切换 YOLOv5 / YOLOv8 模型
2. 统一执行推理并整理成前端易用的数据结构
3. 生成两种结果图：
   - 网页展示图：只显示检测框，不显示烧录标签
   - 下载结果图：使用传统检测标签，显示类别名称
4. 为前端的图片-表格联动与局部放大预览提供边界框与原图尺寸信息
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import importlib
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


matplotlib.use("Agg")


LOCAL_CODEBASE_DIR = "model_codebases"
YOLOV5_LOCAL_REPO_DIR = "yolov5_local"


class DetectionServiceError(Exception):
    """目标检测业务异常。"""


MODEL_OPTIONS = [
    "UAV-YOLO.pt",
    "yolov5n.pt",
    "yolov8n.pt",
    "yolo11n.pt",
]


MODEL_PROFILES = {
    "UAV-YOLO.pt": {
        "title": "UAV-YOLO",
        "params": "约 2.9M",
        "description": "专为无人机目标检测设计，具有较好的实时性能和检测精度。",
    },
    "yolov5n.pt": {
        "title": "YOLOv5 Nano",
        "params": "约 2.5M",
        "description": "经典轻量版本，速度快，适合课程演示和基础对比。",
    },
    "yolov8n.pt": {
        "title": "YOLOv8 Nano",
        "params": "约 3.2M",
        "description": "体积小、速度快，适合本地网页快速演示。",
    },
    "yolo11n.pt": {
        "title": "YOLO11 Nano",
        "params": "约 2.6M",
        "description": "新一代轻量模型，性能更优，适合对检测精度有更高要求的场景。"
    },
}


SMALL_TARGET_MAX_SIDE = 32
SMALL_TARGET_MAX_AREA = 32 * 32

BOX_OUTLINE_COLORS = [
    (15, 118, 110, 255),
    (37, 99, 235, 255),
    (14, 165, 233, 255),
    (245, 158, 11, 255),
    (239, 68, 68, 255),
    (99, 102, 241, 255),
]
DOWNLOAD_LABEL_BG = (24, 59, 86, 225)
DOWNLOAD_LABEL_TEXT = (255, 255, 255, 255)


def run_detection_pipeline(
    image_path: Path,
    model_name: str,
    conf_threshold: str,
    iou_threshold: str,
    img_size: str,
    base_dir: Path,
) -> dict:
    """执行完整检测流程，并返回页面渲染所需数据。"""
    validate_model_name(model_name)

    conf_value = parse_float_range(conf_threshold, "置信度阈值", 0.0, 1.0)
    iou_value = parse_float_range(iou_threshold, "IoU 阈值", 0.0, 1.0)
    img_size_value = parse_img_size(img_size)

    model_path = base_dir / "models" / model_name
    if not model_path.exists():
        raise DetectionServiceError(
            f"模型文件不存在：{model_name}。请先把模型权重放到 models/ 目录。"
        )

    if not image_path.exists():
        raise DetectionServiceError("待检测图片不存在，请重新上传图片后再试。")

    image_width, image_height = read_image_size(image_path)

    if is_yolov5_model(model_name):
        detections, inference_ms = run_yolov5_inference(
            image_path=image_path,
            model_path=model_path,
            conf_value=conf_value,
            iou_value=iou_value,
            img_size_value=img_size_value,
            base_dir=base_dir,
        )
    else:
        detections, inference_ms = run_yolov8_inference(
            image_path=image_path,
            model_path=model_path,
            conf_value=conf_value,
            iou_value=iou_value,
            img_size_value=img_size_value,
        )

    class_counter = Counter(item["class_name"] for item in detections)
    small_targets = [item for item in detections if item["is_small_target"]]

    web_image = draw_web_visualization(image_path, detections)
    download_image = draw_download_visualization(image_path, detections)
    web_image_path = save_annotated_image_array(web_image, base_dir, prefix="web")
    download_image_path = save_annotated_image_array(download_image, base_dir, prefix="download")
    chart_path = generate_bar_chart(class_counter, base_dir)

    return {
        "original_image_url": path_to_web_url(image_path, base_dir),
        "result_image_url": path_to_web_url(web_image_path, base_dir),
        "download_result_url": path_to_web_url(download_image_path, base_dir),
        "chart_image_url": path_to_web_url(chart_path, base_dir),
        "image_width": image_width,
        "image_height": image_height,
        "detections": detections,
        "total_count": len(detections),
        "small_target_count": len(small_targets),
        "small_targets": small_targets,
        "class_stats": dict(class_counter),
        "elapsed_ms": round(inference_ms, 2),
        "selected_model": model_name,
        "model_profile": MODEL_PROFILES.get(model_name, {}),
        "thresholds": {
            "conf": conf_value,
            "iou": iou_value,
            "img_size": img_size_value,
        },
    }


def read_image_size(image_path: Path) -> tuple[int, int]:
    """读取原图尺寸，供前端按比例放置交互高亮框。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise DetectionServiceError("无法读取原始图片尺寸，请重新上传图片后再试。")

    height, width = image.shape[:2]
    return width, height


def is_yolov5_model(model_name: str) -> bool:
    """判断当前模型是否属于 YOLOv5 系列。"""
    return model_name.lower().startswith("yolov5")


def validate_model_name(model_name: str) -> None:
    """校验模型名称是否合法。"""
    if model_name not in MODEL_OPTIONS:
        raise DetectionServiceError("模型名称非法，请从页面下拉框中选择模型。")


def parse_float_range(value: str, field_name: str, min_value: float, max_value: float) -> float:
    """解析浮点型阈值参数。"""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise DetectionServiceError(f"{field_name}必须是数字。") from exc

    if not (min_value <= parsed <= max_value):
        raise DetectionServiceError(f"{field_name}必须位于 {min_value} 到 {max_value} 之间。")
    return parsed


def parse_img_size(value: str) -> int:
    """解析推理尺寸。"""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise DetectionServiceError("推理尺寸必须是整数。") from exc

    if parsed < 320 or parsed > 1280:
        raise DetectionServiceError("推理尺寸建议设置在 320 到 1280 之间。")
    return parsed


def run_yolov8_inference(
    image_path: Path,
    model_path: Path,
    conf_value: float,
    iou_value: float,
    img_size_value: int,
) -> tuple[list[dict], float]:
    """YOLOv8 推理，只统计真正的 predict 阶段耗时。"""
    try:
        model = YOLO(str(model_path))
        start_time = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            conf=conf_value,
            iou=iou_value,
            imgsz=img_size_value,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - start_time) * 1000
    except Exception as exc:  # noqa: BLE001
        raise DetectionServiceError(f"YOLOv8 推理失败：{exc}") from exc

    if not results:
        raise DetectionServiceError("YOLOv8 没有返回结果，请检查模型或图片。")

    return extract_ultralytics_rows(results[0]), inference_ms


def run_yolov5_inference(
    image_path: Path,
    model_path: Path,
    conf_value: float,
    iou_value: float,
    img_size_value: int,
    base_dir: Path,
) -> tuple[list[dict], float]:
    """YOLOv5 推理，先官方后本地。"""
    official_error = None
    local_error = None

    try:
        official_model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(model_path),
            force_reload=False,
            trust_repo=True,
        )
        official_model.conf = conf_value
        official_model.iou = iou_value

        start_time = time.perf_counter()
        official_results = official_model(str(image_path), size=img_size_value)
        inference_ms = (time.perf_counter() - start_time) * 1000
        return extract_yolov5_rows(official_results), inference_ms
    except Exception as exc:  # noqa: BLE001
        official_error = exc

    local_repo = find_local_yolov5_repo(base_dir)
    if local_repo is not None:
        try:
            with temporary_yolov5_import_context(local_repo, base_dir):
                local_model = torch.hub.load(
                    str(local_repo),
                    "custom",
                    path=str(model_path),
                    source="local",
                    force_reload=False,
                )
            local_model.conf = conf_value
            local_model.iou = iou_value

            start_time = time.perf_counter()
            local_results = local_model(str(image_path), size=img_size_value)
            inference_ms = (time.perf_counter() - start_time) * 1000
            return extract_yolov5_rows(local_results), inference_ms
        except Exception as exc:  # noqa: BLE001
            local_error = exc

    message_lines = [
        "YOLOv5 模型加载失败。",
        "程序已按以下顺序尝试：",
        "1. 官方 ultralytics/yolov5",
        "2. 本地仓库 model_codebases/yolov5_local",
    ]

    if official_error is not None:
        message_lines.append(f"官方方式错误：{official_error}")

    if local_repo is None:
        message_lines.append("本地仓库未找到：请将官方 yolov5 源码放入 model_codebases/yolov5_local")
    elif local_error is not None:
        message_lines.append(f"本地仓库方式错误：{local_error}")

    raise DetectionServiceError("\n".join(message_lines))


@contextmanager
def temporary_yolov5_import_context(repo_path: Path, base_dir: Path):
    """临时切换到 YOLOv5 自身的导入环境，避免与当前项目的同名模块冲突。"""
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    removed_modules: dict[str, object] = {}

    for module_name in list(sys.modules.keys()):
        if module_name == "utils" or module_name.startswith("utils."):
            removed_modules[module_name] = sys.modules.pop(module_name)
        elif module_name == "models" or module_name.startswith("models."):
            removed_modules[module_name] = sys.modules.pop(module_name)

    try:
        repo_str = str(repo_path.resolve())
        base_str = str(base_dir.resolve())

        cleaned_sys_path: list[str] = []
        for item in sys.path:
            try:
                resolved = str(Path(item).resolve())
            except Exception:  # noqa: BLE001
                resolved = item

            if resolved != base_str:
                cleaned_sys_path.append(item)

        sys.path = [repo_str] + cleaned_sys_path
        os.chdir(repo_str)
        yield
    finally:
        os.chdir(str(original_cwd))
        sys.path = original_sys_path

        for module_name in list(sys.modules.keys()):
            if module_name == "utils" or module_name.startswith("utils."):
                sys.modules.pop(module_name, None)
            elif module_name == "models" or module_name.startswith("models."):
                sys.modules.pop(module_name, None)

        sys.modules.update(removed_modules)


def load_local_yolov5_model(model_path: Path):
    """直接导入本地 YOLOv5 的 hubconf.custom，避免 torch.hub 缓存和工作区同名模块冲突。"""
    sys.modules.pop("hubconf", None)
    hubconf = importlib.import_module("hubconf")
    hubconf = importlib.reload(hubconf)
    return hubconf.custom(path=str(model_path), autoshape=True, _verbose=False, device="cpu")


@contextmanager
def temporary_yolov5_import_context(repo_path: Path, base_dir: Path):
    """只为本地 YOLOv5 仓库构造干净导入环境，避免 WibeCode/utils 与 models 抢占导入。"""
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    removed_modules: dict[str, object] = {}
    workspace_root = base_dir.parent.resolve()

    for module_name in list(sys.modules.keys()):
        if module_name == "utils" or module_name.startswith("utils."):
            removed_modules[module_name] = sys.modules.pop(module_name)
        elif module_name == "models" or module_name.startswith("models."):
            removed_modules[module_name] = sys.modules.pop(module_name)
        elif module_name == "hubconf":
            removed_modules[module_name] = sys.modules.pop(module_name)

    try:
        repo_str = str(repo_path.resolve())
        cleaned_sys_path: list[str] = []

        for item in original_sys_path:
            try:
                resolved_path = Path(item).resolve()
            except Exception:  # noqa: BLE001
                cleaned_sys_path.append(item)
                continue

            # 移除整个当前项目工作区路径，避免 top-level utils/models 被解析到 WibeCode 内部
            if resolved_path == workspace_root or workspace_root in resolved_path.parents:
                continue

            if str(resolved_path) != repo_str:
                cleaned_sys_path.append(item)

        sys.path = [repo_str] + cleaned_sys_path
        os.chdir(repo_str)
        yield
    finally:
        os.chdir(str(original_cwd))
        sys.path = original_sys_path

        for module_name in list(sys.modules.keys()):
            if module_name == "utils" or module_name.startswith("utils."):
                sys.modules.pop(module_name, None)
            elif module_name == "models" or module_name.startswith("models."):
                sys.modules.pop(module_name, None)
            elif module_name == "hubconf":
                sys.modules.pop(module_name, None)

        sys.modules.update(removed_modules)


def run_yolov5_inference(
    image_path: Path,
    model_path: Path,
    conf_value: float,
    iou_value: float,
    img_size_value: int,
    base_dir: Path,
) -> tuple[list[dict], float]:
    """YOLOv5 推理只走本地 yolov5 仓库，避免官方 hub 缓存和导入冲突。"""
    local_repo = find_local_yolov5_repo(base_dir)
    if local_repo is None:
        raise DetectionServiceError(
            "YOLOv5 本地仓库未找到：请确认官方 yolov5 源码位于 model_codebases/yolov5_local。"
        )

    try:
        with temporary_yolov5_import_context(local_repo, base_dir):
            local_model = load_local_yolov5_model(model_path)

        local_model.conf = conf_value
        local_model.iou = iou_value

        start_time = time.perf_counter()
        local_results = local_model(str(image_path), size=img_size_value)
        inference_ms = (time.perf_counter() - start_time) * 1000
        return extract_yolov5_rows(local_results), inference_ms
    except Exception as exc:  # noqa: BLE001
        raise DetectionServiceError(
            "YOLOv5 本地仓库加载失败。\n"
            "当前策略已经改为只使用 model_codebases/yolov5_local，不再走官方 torch.hub 缓存。\n"
            f"本地仓库错误：{exc}"
        ) from exc


def find_local_yolov5_repo(base_dir: Path) -> Path | None:
    """在项目内寻找用户放置的本地 YOLOv5 官方仓库。"""
    candidate = base_dir / LOCAL_CODEBASE_DIR / YOLOV5_LOCAL_REPO_DIR
    if is_standard_yolov5_repo(candidate):
        return candidate
    return None


def is_standard_yolov5_repo(repo_path: Path) -> bool:
    """判断目录是否为标准 YOLOv5 仓库结构。"""
    required_files = [
        repo_path / "hubconf.py",
        repo_path / "utils" / "general.py",
        repo_path / "models" / "common.py",
    ]
    return all(path.exists() for path in required_files)


def extract_ultralytics_rows(result) -> list[dict]:
    """提取 Ultralytics YOLO 结果。"""
    boxes = result.boxes
    names = result.names
    detections: list[dict] = []

    if boxes is None or len(boxes) == 0:
        return detections

    xyxy_array = boxes.xyxy.cpu().numpy()
    conf_array = boxes.conf.cpu().numpy()
    cls_array = boxes.cls.cpu().numpy().astype(int)

    for index, (xyxy, conf, cls_id) in enumerate(zip(xyxy_array, conf_array, cls_array), start=1):
        detections.append(
            build_detection_item(
                index=index,
                class_id=int(cls_id),
                class_name=str(names.get(cls_id, f"class_{cls_id}")),
                confidence=float(conf),
                xyxy=xyxy,
            )
        )

    return detections


def extract_yolov5_rows(results) -> list[dict]:
    """提取 torch.hub YOLOv5 结果。"""
    detections: list[dict] = []
    names = results.names

    if not results.xyxy or len(results.xyxy[0]) == 0:
        return detections

    rows = results.xyxy[0].cpu().numpy()
    for index, row in enumerate(rows, start=1):
        x1, y1, x2, y2, conf, cls_id = row[:6]
        cls_id = int(cls_id)
        detections.append(
            build_detection_item(
                index=index,
                class_id=cls_id,
                class_name=str(names.get(cls_id, f"class_{cls_id}")),
                confidence=float(conf),
                xyxy=[x1, y1, x2, y2],
            )
        )

    return detections


def build_detection_item(
    index: int,
    class_id: int,
    class_name: str,
    confidence: float,
    xyxy,
) -> dict:
    """整理单个检测目标，供表格、统计和前端交互共用。"""
    x1, y1, x2, y2 = [round(float(v), 2) for v in xyxy]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    width = round(x2 - x1, 2)
    height = round(y2 - y1, 2)
    area = round(width * height, 2)
    is_small_target = (
        width < SMALL_TARGET_MAX_SIDE
        or height < SMALL_TARGET_MAX_SIDE
        or area < SMALL_TARGET_MAX_AREA
    )

    return {
        "id": index,
        "index": index,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "bbox": [x1, y1, x2, y2],
        "bbox_text": f"[{x1}, {y1}, {x2}, {y2}]",
        "width": width,
        "height": height,
        "area": area,
        "is_small_target": is_small_target,
    }


def draw_web_visualization(image_path: Path, detections: list[dict]) -> np.ndarray:
    """
    绘制网页展示图。

    设计目的：
    1. 网页里不直接烧录标签，避免遮挡小目标
    2. 只保留检测框，标签交给前端交互层显示
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise DetectionServiceError("无法读取原始图片，无法生成网页展示图。")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    for item in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        color = build_box_color(item["class_id"])
        draw.rounded_rectangle(
            [(x1, y1), (x2, y2)],
            radius=8,
            outline=color,
            width=3,
        )

    result_rgb = np.array(canvas.convert("RGB"))
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def draw_download_visualization(image_path: Path, detections: list[dict]) -> np.ndarray:
    """
    绘制下载版结果图。

    下载图采用传统标签样式：
    1. 显示类别名称
    2. 标签贴在检测框上方
    3. 更适合论文截图或离线保存
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise DetectionServiceError("无法读取原始图片，无法生成下载结果图。")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    image_width, image_height = canvas.size

    for item in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        color = build_box_color(item["class_id"])
        draw.rounded_rectangle(
            [(x1, y1), (x2, y2)],
            radius=8,
            outline=color,
            width=3,
        )
        draw_download_label(
            draw=draw,
            class_name=item["class_name"],
            anchor_x=x1,
            anchor_y=y1,
            image_width=image_width,
            image_height=image_height,
        )

    result_rgb = np.array(canvas.convert("RGB"))
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def draw_download_label(
    draw: ImageDraw.ImageDraw,
    class_name: str,
    anchor_x: int,
    anchor_y: int,
    image_width: int,
    image_height: int,
) -> None:
    """为下载图绘制传统类别标签。"""
    font = load_preferred_font(14, bold=True)
    text = class_name
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    pad_x = 8
    pad_y = 4
    label_width = text_width + pad_x * 2
    label_height = text_height + pad_y * 2
    label_x = anchor_x
    label_y = anchor_y - label_height - 6

    if label_y < 4:
        label_y = min(anchor_y + 4, max(4, image_height - label_height - 4))
    if label_x + label_width > image_width - 4:
        label_x = max(4, image_width - label_width - 4)
    if label_x < 4:
        label_x = 4

    draw.rounded_rectangle(
        (label_x, label_y, label_x + label_width, label_y + label_height),
        radius=8,
        fill=DOWNLOAD_LABEL_BG,
    )
    draw.text(
        (label_x + pad_x, label_y + pad_y - 1),
        text,
        font=font,
        fill=DOWNLOAD_LABEL_TEXT,
    )


def load_preferred_font(
    font_size: int,
    bold: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先加载 Windows 常见中文字体，失败则退回默认字体。"""
    font_candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:  # noqa: BLE001
            continue

    return ImageFont.load_default()


def build_box_color(class_id: int) -> tuple[int, int, int, int]:
    """根据类别编号稳定生成检测框颜色。"""
    return BOX_OUTLINE_COLORS[class_id % len(BOX_OUTLINE_COLORS)]


def save_annotated_image_array(image_array: np.ndarray, base_dir: Path, prefix: str) -> Path:
    """保存检测结果图。"""
    file_name = f"{prefix}_{uuid.uuid4().hex[:12]}.jpg"
    output_path = base_dir / "static" / "generated" / file_name
    cv2.imwrite(str(output_path), image_array)
    return output_path


def generate_bar_chart(class_counter: Counter, base_dir: Path) -> Path:
    """生成类别分布柱状图。"""
    file_name = f"chart_{uuid.uuid4().hex[:12]}.png"
    output_path = base_dir / "static" / "generated" / file_name

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(8, 4.5), dpi=140)
    if class_counter:
        labels = list(class_counter.keys())
        counts = list(class_counter.values())
        colors = ["#0f766e", "#2563eb", "#f59e0b", "#dc2626", "#7c3aed", "#16a34a"]
        plt.bar(labels, counts, color=colors[: len(labels)])
        plt.title("检测类别分布统计")
        plt.xlabel("类别名称")
        plt.ylabel("检测数量")
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        for idx, count in enumerate(counts):
            plt.text(idx, count + 0.05, str(count), ha="center", va="bottom", fontsize=9)
    else:
        plt.text(0.5, 0.5, "未检测到目标", ha="center", va="center", fontsize=16)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("检测类别分布统计")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def path_to_web_url(file_path: Path, base_dir: Path) -> str:
    """把磁盘路径转换为网页可访问的相对路径。"""
    return file_path.relative_to(base_dir).as_posix()


def load_local_yolov5_modules():
    """Load YOLOv5 inference helpers from the isolated local repo."""
    common = importlib.import_module("models.common")
    augmentations = importlib.import_module("utils.augmentations")
    general = importlib.import_module("utils.general")

    return {
        "DetectMultiBackend": common.DetectMultiBackend,
        "letterbox": augmentations.letterbox,
        "check_img_size": general.check_img_size,
        "non_max_suppression": general.non_max_suppression,
        "scale_boxes": general.scale_boxes,
    }


def run_local_yolov5_manual_inference(
    image_path: Path,
    model_path: Path,
    conf_value: float,
    iou_value: float,
    img_size_value: int,
) -> tuple[list[dict], float]:
    """Run local YOLOv5 inference with the same core path as detect.py."""
    modules = load_local_yolov5_modules()
    detect_multi_backend = modules["DetectMultiBackend"]
    letterbox = modules["letterbox"]
    check_img_size = modules["check_img_size"]
    non_max_suppression = modules["non_max_suppression"]
    scale_boxes = modules["scale_boxes"]

    device = torch.device("cpu")
    model = detect_multi_backend(str(model_path), device=device, dnn=False, data=None, fp16=False)
    stride = model.stride
    names = model.names
    pt = model.pt
    imgsz = check_img_size(img_size_value, s=stride)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise DetectionServiceError("Unable to read input image for local YOLOv5 inference.")

    original_image = image_bgr.copy()
    processed = letterbox(image_bgr, new_shape=imgsz, stride=stride, auto=pt)[0]
    processed = processed[:, :, ::-1].transpose(2, 0, 1)
    processed = np.ascontiguousarray(processed)

    tensor = torch.from_numpy(processed).to(device).float() / 255.0
    tensor = tensor.unsqueeze(0)

    start_time = time.perf_counter()
    with torch.no_grad():
        prediction = model(tensor, augment=False, visualize=False)
        prediction = non_max_suppression(
            prediction,
            conf_thres=conf_value,
            iou_thres=iou_value,
            max_det=1000,
        )
    inference_ms = (time.perf_counter() - start_time) * 1000

    detections: list[dict] = []
    if prediction and len(prediction[0]):
        det = prediction[0]
        det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], original_image.shape).round()
        rows = det.cpu().numpy()

        for index, row in enumerate(rows, start=1):
            x1, y1, x2, y2, conf, cls_id = row[:6]
            cls_id = int(cls_id)
            if isinstance(names, dict):
                class_name = str(names.get(cls_id, f"class_{cls_id}"))
            else:
                class_name = str(names[cls_id]) if cls_id < len(names) else f"class_{cls_id}"

            detections.append(
                build_detection_item(
                    index=index,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=float(conf),
                    xyxy=[x1, y1, x2, y2],
                )
            )

    return detections, inference_ms


def run_yolov5_inference(
    image_path: Path,
    model_path: Path,
    conf_value: float,
    iou_value: float,
    img_size_value: int,
    base_dir: Path,
) -> tuple[list[dict], float]:
    """Use only the local YOLOv5 repo and a manual inference path."""
    local_repo = find_local_yolov5_repo(base_dir)
    if local_repo is None:
        raise DetectionServiceError("Local YOLOv5 repo not found: expected model_codebases/yolov5_local.")

    try:
        with temporary_yolov5_import_context(local_repo, base_dir):
            return run_local_yolov5_manual_inference(
                image_path=image_path,
                model_path=model_path,
                conf_value=conf_value,
                iou_value=iou_value,
                img_size_value=img_size_value,
            )
    except Exception as exc:  # noqa: BLE001
        raise DetectionServiceError(
            "Local YOLOv5 load failed.\n"
            "This app now uses only model_codebases/yolov5_local with a manual inference path.\n"
            f"Local repo error: {exc}"
        ) from exc
