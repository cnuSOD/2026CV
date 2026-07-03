# 本地模型代码仓库说明

这个目录专门用于存放你自己的模型源码，方便后续接入：

- 官方 `YOLOv5` 源码
- 你自己改进后的模型源码
- 其他以后需要接入网页的检测项目

## 当前程序的查找顺序

当你在网页中选择 `yolov5*.pt` 模型时，程序会按下面顺序尝试：

1. 先尝试官方 `ultralytics/yolov5`
2. 如果官方方式失败，再尝试本地仓库 `model_codebases/yolov5_local/`
3. 如果两者都失败，再弹出错误提示

## 你现在应该怎么放

### 1. 放官方 YOLOv5 源码

请把完整的官方 `YOLOv5` 仓库源码放到：

```text
WibeCode/model_codebases/yolov5_local/
```

放好后，这个目录里至少应当能看到：

```text
hubconf.py
models/common.py
utils/general.py
```

### 2. 放你自己的改进模型源码

建议放到：

```text
WibeCode/model_codebases/custom_models/
```

例如：

```text
WibeCode/model_codebases/custom_models/my_yolov5_improved/
WibeCode/model_codebases/custom_models/my_small_object_model/
```

## 后续怎么扩展

如果你后面要接你自己的改进模型，可以在：

```text
services/yolo_service.py
```

里继续扩展：

- 新增模型名称到 `MODEL_OPTIONS`
- 新增模型说明到 `MODEL_PROFILES`
- 按模型前缀增加新的推理分支

例如以后可以加：

- `myyolo_v1.pt`
- `improved_yolov5s.pt`
- `small_object_best.pt`

然后在后端按前缀判断走不同的代码。
