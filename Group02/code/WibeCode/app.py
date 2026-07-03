"""
Flask 项目主入口
职责：
1. 渲染首页
2. 处理图片上传与重复检测
3. 调用 YOLO 推理服务
4. 统一处理错误提示
"""

from pathlib import Path

from flask import Flask, abort, render_template, request, send_file

from services.yolo_service import (
    MODEL_OPTIONS,
    MODEL_PROFILES,
    DetectionServiceError,
    run_detection_pipeline,
)
from utils.file_utils import ensure_directories, get_input_image_path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREVIEW_IMAGE = "static/sample_images/demo_uav.png"
REQUIRED_DIRS = [
    BASE_DIR / "models",
    BASE_DIR / "uploads",
    BASE_DIR / "results",
    BASE_DIR / "static" / "generated",
    BASE_DIR / "static" / "sample_images",
    BASE_DIR / "model_codebases",
    BASE_DIR / "model_codebases" / "yolov5_local",
    BASE_DIR / "model_codebases" / "custom_models",
]


app = Flask(__name__)
app.config["SECRET_KEY"] = "cv-course-demo-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def build_default_form_data() -> dict:
    """构造页面默认表单值。"""
    return {
        "model_name": "yolov8n.pt",
        "conf_threshold": "0.25",
        "iou_threshold": "0.45",
        "img_size": "640",
    }


def build_base_context() -> dict:
    """构造首页通用上下文。"""
    return {
        "error_message": "",
        "status_message": "等待上传图片并开始检测。",
        "result_data": None,
        "model_options": MODEL_OPTIONS,
        "model_profiles": MODEL_PROFILES,
        "form_data": build_default_form_data(),
        "current_image_url": DEFAULT_PREVIEW_IMAGE,
        "reusable_image_url": "",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    """首页路由。"""
    ensure_directories(REQUIRED_DIRS)
    context = build_base_context()

    if request.method == "GET":
        return render_template("index.html", **context)

    form_data = {
        "model_name": request.form.get("model_name", "yolov8n.pt").strip(),
        "conf_threshold": request.form.get("conf_threshold", "0.25").strip(),
        "iou_threshold": request.form.get("iou_threshold", "0.45").strip(),
        "img_size": request.form.get("img_size", "640").strip(),
    }
    context["form_data"] = form_data

    try:
        upload_file = request.files.get("image")
        existing_image_path = request.form.get("existing_image_path", "").strip()
        image_path = get_input_image_path(
            upload_file=upload_file,
            existing_image_relative=existing_image_path,
            base_dir=BASE_DIR,
            upload_dir=BASE_DIR / "uploads",
        )

        reusable_relative_path = image_path.relative_to(BASE_DIR).as_posix()
        context["current_image_url"] = reusable_relative_path
        context["reusable_image_url"] = reusable_relative_path

        result_data = run_detection_pipeline(
            image_path=image_path,
            model_name=form_data["model_name"],
            conf_threshold=form_data["conf_threshold"],
            iou_threshold=form_data["iou_threshold"],
            img_size=form_data["img_size"],
            base_dir=BASE_DIR,
        )

        context["result_data"] = result_data
        context["status_message"] = "检测完成，结果已生成。"
        context["current_image_url"] = result_data["original_image_url"]
        context["reusable_image_url"] = result_data["original_image_url"]
        return render_template("index.html", **context)

    except DetectionServiceError as exc:
        context["error_message"] = str(exc)
        context["status_message"] = "检测未完成，请根据提示修正输入后重试。"

        # 如果是重复检测失败，也尽量保留上一张图
        existing_image_path = request.form.get("existing_image_path", "").strip()
        if existing_image_path:
            context["current_image_url"] = existing_image_path
            context["reusable_image_url"] = existing_image_path

        return render_template("index.html", **context)

    except Exception as exc:  # noqa: BLE001
        context["error_message"] = f"系统出现未预期错误：{exc}"
        context["status_message"] = "系统执行失败，请检查模型、依赖或图片文件。"
        return render_template("index.html", **context)


@app.route("/media/<path:relative_path>")
def media_file(relative_path: str):
    """提供项目目录内图片文件的网页访问。"""
    target_path = (BASE_DIR / relative_path).resolve()
    base_resolved = BASE_DIR.resolve()

    if base_resolved not in target_path.parents and target_path != base_resolved:
        abort(404)

    if not target_path.exists() or not target_path.is_file():
        abort(404)

    return send_file(target_path)


@app.errorhandler(413)
def file_too_large(_error):
    """上传文件过大时给出友好提示。"""
    context = build_base_context()
    context["error_message"] = "上传文件过大，请控制在 10MB 以内。"
    context["status_message"] = "检测未完成，请缩小图片后重新上传。"
    return render_template("index.html", **context), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
