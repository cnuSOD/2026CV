"""
文件处理工具模块
职责：
1. 创建目录
2. 校验上传文件类型
3. 保存上传图片
4. 支持重复检测时复用上一张图片
"""

from __future__ import annotations

import uuid
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from services.yolo_service import DetectionServiceError


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_directories(paths: list[Path]) -> None:
    """确保目录存在。"""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_uploaded_image(upload_file: FileStorage | None, upload_dir: Path) -> Path:
    """保存用户上传图片并返回保存路径。"""
    ensure_directories([upload_dir])

    if upload_file is None:
        raise DetectionServiceError("请先选择一张图片后再提交。")

    original_name = upload_file.filename or ""
    if not original_name.strip():
        raise DetectionServiceError("未检测到有效文件名，请重新选择图片。")

    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise DetectionServiceError("仅支持上传 jpg、jpeg、png 格式图片。")

    safe_name = secure_filename(Path(original_name).stem)
    unique_name = f"{safe_name}_{uuid.uuid4().hex[:12]}{suffix}"
    save_path = upload_dir / unique_name

    try:
        upload_file.save(save_path)
    except Exception as exc:  # noqa: BLE001
        raise DetectionServiceError(f"上传图片保存失败：{exc}") from exc

    return save_path


def get_input_image_path(
    upload_file: FileStorage | None,
    existing_image_relative: str,
    base_dir: Path,
    upload_dir: Path,
) -> Path:
    """
    获取本次检测实际使用的图片路径。

    优先级：
    1. 如果用户重新上传了新图片，就使用新图片
    2. 如果没上传，但页面隐藏字段里有上一张图，就继续复用上一张图
    3. 两者都没有则报错
    """
    if upload_file is not None and (upload_file.filename or "").strip():
        return save_uploaded_image(upload_file, upload_dir)

    if existing_image_relative:
        existing_path = (base_dir / existing_image_relative).resolve()
        upload_dir_resolved = upload_dir.resolve()

        if upload_dir_resolved not in existing_path.parents:
            raise DetectionServiceError("当前复用图片路径无效，请重新上传图片。")

        if not existing_path.exists() or not existing_path.is_file():
            raise DetectionServiceError("上一张图片已不存在，请重新上传图片。")

        return existing_path

    raise DetectionServiceError("请先选择一张图片后再提交。")
