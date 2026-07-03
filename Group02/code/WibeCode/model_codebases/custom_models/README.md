# custom_models 使用说明

这个目录留给你以后放自己的改进模型源码。

推荐结构：

```text
custom_models/
├─ my_model_a/
├─ my_model_b/
└─ ...
```

当前网页程序还没有自动加载这里的自定义项目，
但接口已经预留好了，后续你只需要在：

```text
services/yolo_service.py
```

里补充对应的模型分支即可。
