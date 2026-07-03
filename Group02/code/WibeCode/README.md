# 低空目标检测可视化系统

用于基于YOLO的低空无人机图像目标检测系统，支持：

- 上传图片进行目标检测
- 切换不同 YOLO 模型
- 调整置信度阈值、IoU 阈值、推理尺寸
- 显示原图与检测结果图
- 展示检测结果表格
- 生成类别分布柱状图
- 统计小目标数量并列出小目标目标框
- 一键加载示例图片

## 1. 项目目录结构

```text
WibeCode/
├─ app.py
├─ requirements.txt
├─ README.md
├─ templates/
│  └─ index.html
├─ static/
│  ├─ css/
│  │  └─ style.css
│  ├─ js/
│  │  └─ main.js
│  ├─ generated/
│  └─ sample_images/
│     └─ demo_uav.jpg
├─ models/
├─ uploads/
├─ results/
├─ services/
│  ├─ __init__.py
│  └─ yolo_service.py
└─ utils/
   ├─ __init__.py
   └─ file_utils.py
```

## 2. 推荐 Python 版本

- Python 3.10
- 或 Python 3.11

## 3. 安装依赖

```powershell
cd "d:\AASchool\Computer Sience\ComputerVision\WibeCode"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 下载模型文件到 models 目录

- 请把权重文件放到 `models/`
- 推荐从 Ultralytics 官方发布页面或 GitHub Release 下载。

## 5. 运行项目

```powershell
python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

## 6. 如何测试

1. 打开首页。
2. 点击“一键加载示例图”。
3. 再切换不同模型测试。
4. 上传自己的图片继续测试。
5. 下载结果图放到答辩 PPT。

## 7. 小目标统计规则

当前演示版规则：

- 宽度 < 32 像素，或
- 高度 < 32 像素，或
- 面积 < 1024 像素

如需修改，可直接编辑 `services/yolo_service.py` 中的：

- `SMALL_TARGET_MAX_SIDE`
- `SMALL_TARGET_MAX_AREA`
