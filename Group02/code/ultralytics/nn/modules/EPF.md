# EPF And Physical Guidance Notes

## 1. Current Design Goal

当前这套设计里，EPF 或 PFG 分支的职责不是直接替代 backbone 特征，而是：

1. 从原始输入图像中提取高频指导信息
2. 经过轻量适配，得到和 backbone 对应尺度匹配的 physical feature
3. 再和 backbone 的视觉特征融合
4. 融合后的特征送给 neck

也就是说，系统结构更接近：

```text
input image
-> visual backbone -> visual P2 / P3

input image
-> physical branch -> physical P2 / P3

visual P2 + physical P2 -> fusion -> fused P2
visual P3 + physical P3 -> fusion -> fused P3

fused P2 / P3 -> neck
```

## 2. Why Pure YAML Routing Fails

如果希望保留“原图并联 physical 分支”，不能简单靠 YAML 里的 `from=-2`、`from=-3` 去反复取输入图像。

原因是 Ultralytics 的模型构图和 forward 机制默认是：

1. `-1` 表示上一层输出
2. 其他索引表示已经构建好的历史层输出
3. 中途不能再次从 YAML 中显式回到原始输入图像

因此这类写法容易报错：

```yaml
- [-2, 1, PFG, [3]]
```

常见报错包括：

1. `IndexError: list index out of range`
2. `ch[f]` 越界
3. `from` 索引在当前层尚不存在合法来源

## 3. Recommended Method 1

推荐的方法 1 是：

`保留原图并联 physical 分支，但不在 YAML 中显式画出这条分支，而是在 tasks.py 的 forward 中手动注入。`

这样做的优点：

1. 不会被 YAML `from` 索引机制限制
2. 原始输入图像可以稳定地送入 physical branch
3. physical 分支和 visual 主干职责清晰
4. 更适合你当前的研究思路

## 4. Method 1 Overall Steps

方法 1 需要改 3 个核心位置：

1. `PhysicalFeatureGuidance.py`
2. `yolo11-cut+PFG.yaml`
3. `tasks.py`

思路是：

1. YAML 里只保留正常的 visual backbone 和 head
2. 在 `tasks.py` 中手动建立 physical branch
3. 在 forward 开始时，用原始输入图像生成 `physical_p2` 和 `physical_p3`
4. 在 backbone 跑到对应层时，把它们和 visual 特征融合

## 5. File-Level Modification Guide

### 5.1 Modify `PhysicalFeatureGuidance.py`

建议保留你现有的这些模块：

1. `PhysicalFeatureGuidance`
2. `PhysicalAdapter1`
3. `PhysicalAdapter2`
4. `PhysicalAdapter3`

然后新增一个完整的物理分支类：

```python
class PhysicalBranch(nn.Module):
    def __init__(self, cutoff_ratio=0.1):
        super().__init__()
        self.guidance = PhysicalFeatureGuidance(channels=3, cutoff_ratio=cutoff_ratio)
        self.pa1 = PhysicalAdapter1()   # -> 64x320x320
        self.pa2 = PhysicalAdapter2()   # -> 128x160x160
        self.pa3 = PhysicalAdapter3()   # -> 256x80x80

    def forward(self, x):
        x = self.guidance(x)   # [B, 3, H, W]
        p1 = self.pa1(x)       # [B, 64, H/2, W/2]
        p2 = self.pa2(p1)      # [B, 128, H/4, W/4]
        p3 = self.pa3(p2)      # [B, 256, H/8, W/8]
        return p2, p3
```

这里最重要的是：

1. `physical_p2` 的尺寸和 visual P2 对齐
2. `physical_p3` 的尺寸和 visual P3 对齐

如果输入是 `640x640`，那么：

1. `physical_p2` 应为 `128x160x160`
2. `physical_p3` 应为 `256x80x80`

### 5.2 Modify `yolo11-cut+PFG.yaml`

方法 1 下，YAML 不再显式创建 physical 分支。

因此要把这类层删掉：

```yaml
- [-2, 1, PFG, [3]]
- [-1, 1, PA1, []]
- [-2, 1, PA2, []]
- [-3, 1, PA3, []]
- [[...], 1, POFF, [...]]
```

原因是：

1. 这些层本来就是为了在 YAML 中构造 physical 分支
2. 但方法 1 已经把 physical 分支移动到了 `tasks.py/forward`
3. 所以 YAML 里只保留视觉主干和 head 即可

建议你的 backbone 先恢复为正常视觉 backbone，例如：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0
  - [-1, 1, Conv, [128, 3, 2]]         # 1
  - [-1, 2, C3k2, [128, False, 0.25]]  # 2  <- visual P2
  - [-1, 1, Conv, [256, 3, 2]]         # 3
  - [-1, 2, C3k2, [256, False, 0.25]]  # 4  <- visual P3
  - [-1, 1, Conv, [512, 3, 2]]         # 5
  - [-1, 2, C3k2, [512, True]]         # 6
  - [-1, 1, SPPF, [512, 5]]            # 7
  - [-1, 2, C2PSA, [512]]              # 8
```

注意：

1. 这里的 layer index 非常重要
2. 后面在 `tasks.py` 中你会按这些层号做融合

### 5.3 Modify `tasks.py`

这是方法 1 的核心。

你需要做 3 件事：

1. import `PhysicalBranch`
2. 在 `DetectionModel.__init__()` 中实例化 physical branch 和 fusion module
3. 在 `forward/predict` 的主循环中手动把 physical feature 注入到对应层

## 6. Detailed `tasks.py` Changes

### 6.1 Import Modules

在文件顶部增加：

```python
from ultralytics.nn.modules.PhysicalFeatureGuidance import PhysicalBranch
from ultralytics.nn.modules.PhysicalOriginalFeatureFusion import PhysicalOriginalFeatureFusion
```

### 6.2 Register Modules in `DetectionModel.__init__()`

在：

```python
self.model, self.save = parse_model(...)
```

后面增加：

```python
self.physical_branch = PhysicalBranch(cutoff_ratio=0.1)
self.p2_fusion = PhysicalOriginalFeatureFusion(visual_dim=128, physical_dim=128)
self.p3_fusion = PhysicalOriginalFeatureFusion(visual_dim=256, physical_dim=256)
```

这里的 `128` 和 `256` 要和 YAML 中 visual P2 / P3 的通道严格一致。

### 6.3 Compute Physical Features Before the Main Forward Loop

在真正遍历 `self.model` 之前，先保留原始输入图像：

```python
x0 = x
physical_p2, physical_p3 = self.physical_branch(x0)
```

这一步的含义是：

1. visual 主干继续吃 `x`
2. physical 分支单独吃原始输入 `x0`
3. physical 特征不再依赖 YAML 的 `from`

### 6.4 Inject Fusion at the Target Backbone Layers

在主循环中：

```python
for m in self.model:
    if m.f != -1:
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

    x = m(x)

    if m.i == 2:
        x = self.p2_fusion(x, physical_p2)

    if m.i == 4:
        x = self.p3_fusion(x, physical_p3)

    y.append(x if m.i in self.save else None)
```

这里假设：

1. layer `2` 是 visual P2
2. layer `4` 是 visual P3

如果你 later 改了 YAML，层号也必须一起改。

## 7. Why This Works

方法 1 的核心优点是：

1. YAML 只负责 visual 主图
2. physical branch 直接从原始输入中提取高频先验
3. fusion 明确发生在指定 backbone 层
4. 融合后的结果自然进入 neck

换句话说，YAML 负责“主网络结构”，`tasks.py` 负责“研究型特征注入逻辑”。

## 8. Notes And Risks

### 8.1 Layer Index Must Match YAML

如果你的 backbone 改了，这里：

```python
if m.i == 2:
if m.i == 4:
```

都要同步修改。

### 8.2 Channel Count Must Match

如果 visual P2 / P3 的通道不是 `128 / 256`，那：

```python
self.p2_fusion = PhysicalOriginalFeatureFusion(...)
self.p3_fusion = PhysicalOriginalFeatureFusion(...)
```

里面的参数也要一起改。

### 8.3 Keep Method 1 Hard-Coded First

先不要急着把这套写成完全通用的动态版本。

建议先：

1. 固定一个 YAML
2. 固定 P2 / P3 两个融合点
3. 先跑通
4. 再考虑做可配置化

## 9. Minimal Pseudocode Summary

### In `PhysicalFeatureGuidance.py`

```python
class PhysicalBranch(nn.Module):
    def __init__(self, cutoff_ratio=0.1):
        ...

    def forward(self, x):
        ...
        return p2, p3
```

### In `DetectionModel.__init__()`

```python
self.model, self.save = parse_model(...)

self.physical_branch = PhysicalBranch(cutoff_ratio=0.1)
self.p2_fusion = PhysicalOriginalFeatureFusion(visual_dim=128, physical_dim=128)
self.p3_fusion = PhysicalOriginalFeatureFusion(visual_dim=256, physical_dim=256)
```

### In `forward/predict`

```python
x0 = x
physical_p2, physical_p3 = self.physical_branch(x0)

for m in self.model:
    if m.f != -1:
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

    x = m(x)

    if m.i == 2:
        x = self.p2_fusion(x, physical_p2)

    if m.i == 4:
        x = self.p3_fusion(x, physical_p3)

    y.append(x if m.i in self.save else None)
```

## 10. Final Recommendation

如果你要先验证这条研究路线，方法 1 是当前最稳的选择：

1. 保留原图并联 physical 分支
2. 不依赖 YAML 回取原图
3. 在 `tasks.py/forward` 手动注入 physical guidance
4. 只先对 P2 和 P3 做融合

先跑通、先验证效果，再考虑进一步抽象成更通用的模块化结构。
