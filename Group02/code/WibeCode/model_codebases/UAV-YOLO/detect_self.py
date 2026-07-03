from ultralytics import YOLO
import cv2
import numpy as np
import os

# ============ 配置参数 ============
# 模型权重文件路径
model_path = "D:/AASchool/Computer Sience/ComputerVision/UAV-YOLO/runs/detect/deeper/weights/best.pt"

# 输入图片路径
# image_path = "D:/Programs/yolo11-self-3/ultralytics-main/datasets/SSDD/images/val/000501.jpg"
image_path = "D:/AASchool/Computer Sience/ComputerVision/AAMAIN/datasets/VisDrone2019/images/val/9999938_00000_d_0000019.jpg"
# 标签文件基础路径
# label_base_path = "D:/Programs/yolo11-self-3/ultralytics-main/datasets/SSDD/labels"
label_base_path = "D:/AASchool/Computer Sience/ComputerVision/AAMAIN/datasets/VisDrone2019/labels"
# 输出图片路径
output_path = "detected_image_compared.jpg"
# 置信度阈值
conf_threshold = 0.2
# IoU阈值
iou_threshold = 0.2
# 匹配IoU阈值（用于判断预测框与真实框是否匹配）
match_iou_threshold = 0.2

# ============ 加载模型 ============
print(f"加载模型: {model_path}")
model = YOLO(model_path)


# ============ 辅助函数 ============
def calculate_iou(box1, box2):
    """计算两个框的 IoU.
    box1, box2: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def load_yolo_labels(label_path, img_width, img_height):
    """读取 YOLO 格式标签并转换为像素坐标.
    返回: list of [class_id, x1, y1, x2, y2]
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        print(f"警告: 标签文件不存在: {label_path}")
        return gt_boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2

            gt_boxes.append([class_id, x1, y1, x2, y2])

    return gt_boxes


# ============ 读取图片 ============
print(f"读取图片: {image_path}")
image = cv2.imread(image_path)

if image is None:
    print(f"错误: 无法读取图片 {image_path}")
    raise SystemExit(1)

img_height, img_width = image.shape[:2]
print(f"图片尺寸: {img_width} x {img_height}")


# ============ 读取标签文件 ============
image_filename = os.path.basename(image_path)
image_name_without_ext = os.path.splitext(image_filename)[0]

if "/images/" in image_path.replace("\\", "/"):
    relative_path = image_path.replace("\\", "/").split("/images/")[1]
    label_path = os.path.join(
        label_base_path,
        os.path.dirname(relative_path),
        image_name_without_ext + ".txt",
    )
else:
    label_path = os.path.join(label_base_path, image_name_without_ext + ".txt")

print(f"读取标签文件: {label_path}")
gt_boxes = load_yolo_labels(label_path, img_width, img_height)
print(f"标签文件中有 {len(gt_boxes)} 个真实目标")


# ============ 执行目标检测 ============
print("正在进行目标检测...")
results = model.predict(
    image,
    conf=conf_threshold,
    iou=iou_threshold,
    verbose=False,
)


# ============ 解析检测结果 ============
result = results[0]
boxes = result.boxes

print(f"\n检测到 {len(boxes)} 个预测目标")
print("-" * 80)

pred_boxes = []
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    confidence = float(box.conf[0].cpu().numpy())
    class_id = int(box.cls[0].cpu().numpy())
    class_name = model.names[class_id]

    pred_boxes.append(
        {
            "box": [x1, y1, x2, y2],
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "matched": False,
            "has_gt_overlap": False,
        }
    )

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    print(f"预测目标 {i + 1}:")
    print(f"  类别: {class_name}")
    print(f"  置信度: {confidence:.4f}")
    print(f"  位置坐标: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
    print(f"  中心点: ({center_x:.1f}, {center_y:.1f})")
    print(f"  宽度x高度: {width:.1f} x {height:.1f}")
    print("-" * 80)


# ============ 匹配预测框与真实框 ============
gt_matched = [False] * len(gt_boxes)

print("\n匹配预测框与真实标签:")
print("-" * 80)

for i, pred in enumerate(pred_boxes):
    best_iou = 0.0
    best_gt_idx = -1
    max_iou_any_gt = 0.0

    for j, gt_box in enumerate(gt_boxes):
        iou = calculate_iou(pred["box"], gt_box[1:])
        if iou > max_iou_any_gt:
            max_iou_any_gt = iou

        if gt_matched[j]:
            continue

        # 忽略类别，只按 IoU 匹配一个尚未占用的 GT
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = j

    pred["has_gt_overlap"] = max_iou_any_gt >= match_iou_threshold

    if best_iou >= match_iou_threshold and best_gt_idx != -1:
        pred["matched"] = True
        gt_matched[best_gt_idx] = True
        print(f"预测目标 {i + 1} 匹配到真实目标 {best_gt_idx + 1}, IoU={best_iou:.3f} ✓")
    else:
        print(f"预测目标 {i + 1} 未匹配到真实目标 (最高IoU={best_iou:.3f})")

print("-" * 80)
print("匹配统计:")
print(f"  预测框总数: {len(pred_boxes)}")
print(f"  真实框总数: {len(gt_boxes)}")
print(f"  匹配成功: {sum(p['matched'] for p in pred_boxes)}")
print(f"  漏检(未被预测): {sum(not m for m in gt_matched)}")
print("-" * 80)


# ============ 可视化结果 ============
annotated_image = image.copy()

# 绿色: 成功匹配
# 黄色: 与所有 GT 都不够重叠的真误检
# 不显示: 与某个 GT 重叠但只是重复预测的框
for pred in pred_boxes:
    x1, y1, x2, y2 = [int(v) for v in pred["box"]]

    if pred["matched"]:
        color = (0, 255, 0)
    elif not pred["has_gt_overlap"]:
        color = (0, 255, 255)
    else:
        continue

    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

# 绘制未被检测到的真实框（红色圆圈）
for gt_matched_flag, gt_box in zip(gt_matched, gt_boxes):
    if gt_matched_flag:
        continue

    _, x1, y1, x2, y2 = gt_box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    radius = int(max(x2 - x1, y2 - y1) / 2)

    cv2.circle(annotated_image, (center_x, center_y), radius, (0, 0, 255), 2)
    cv2.line(
        annotated_image,
        (center_x - radius, center_y),
        (center_x + radius, center_y),
        (0, 0, 255),
        1,
    )
    cv2.line(
        annotated_image,
        (center_x, center_y - radius),
        (center_x, center_y + radius),
        (0, 0, 255),
        1,
    )


# ============ 显示并保存结果 ============
cv2.imshow("YOLO Detection Results", annotated_image)
print("\n按任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(output_path, annotated_image)
print(f"检测结果已保存到: {output_path}")