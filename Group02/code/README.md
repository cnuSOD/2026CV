# Group02 Code 说明

本目录主要包含模型训练代码和网页可视化系统代码，整体结构如下：

```text
code/
├── ultralytics/      # YOLO/Ultralytics 模型相关代码
├── WibeCode/         # 网页可视化系统代码
└── train_self.py     # 自定义训练入口脚本
```

## 1. 模型代码与训练

`ultralytics/` 文件夹下是模型相关文件，包括 YOLO 模型结构、训练、验证、推理等核心代码。

如需训练模型，可以直接使用当前目录下的 `train_self.py`：

```powershell
python train_self.py
```

训练脚本中会加载 `uavyolo.yaml` 模型配置，并调用 Ultralytics 的 `model.train()` 进行训练。使用前请根据自己的环境检查或修改以下参数：

- `data`：数据集配置文件路径
- `epochs`：训练轮数
- `imgsz`：输入图像尺寸
- `batch`：批大小
- `device`：训练设备，如 `cuda` 或 `cpu`
- `resume`：是否继续上一次训练

训练结果通常会保存在 Ultralytics 默认的 `runs/` 目录下。

## 2. 网页可视化系统

`WibeCode/` 路径下是网页代码，用于进行低空目标检测结果的可视化展示。

该部分包含：

- Flask 后端入口 `app.py`
- 前端页面与静态资源
- 模型加载与检测服务代码
- 上传图片、检测结果与示例图片相关目录

网页系统的安装、运行和测试方式请直接阅读：

```text
WibeCode/README.md
```

通常进入 `WibeCode/` 后，安装依赖并运行 `app.py` 即可启动网页服务。

## 3. 使用建议

如果只需要重新训练模型，请优先查看并运行 `train_self.py`。

如果只需要运行演示网页或查看检测效果，请进入 `WibeCode/`，并按照 `WibeCode/README.md` 中的说明操作。
