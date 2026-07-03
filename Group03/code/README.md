# 基于双轨制架构的康复动作识别与流式交互系统

*(Dual-Track Rehabilitation Action Recognition and Streaming Interaction System)*

## 一、 项目简介

本项目旨在针对深蹲、单腿站立、开合跳等 5 类常见康复/健身动作，提供从底层数据处理、模型训练到前端流式交互的完整端到端解决方案。

**核心架构亮点：**

* **“双轨制”部署策略：** 针对不同硬件算力与使用场景，并行开发了“学术验证版”（主打高精度跨模态融合）与“工程部署版”（主打边缘端纯净稳定）。
* **跨模态特征语义对齐：** 彻底解决 2D 单目摄像头（MediaPipe 33点）与 3D 深度相机（Kinect 25点）的骨骼拓扑错位问题，提取并映射了 12 个肢体核心大关节。
* **HCI 流式防抖拦截系统：** 在 UI 交互层独创基于“运动方差（Motion Variance）”的静态底噪拦截，并结合 60% 动态置信度与状态驻留机制（Hold Counter），实现了极度丝滑的视觉反馈。

---

## 二、 环境配置与安装

建议使用 **Python 3.8 - 3.10** 环境运行本项目。

```bash
# 1. 克隆本项目
# git clone https://github.com/YourUsername/YourRepo.git

# 2. 进入项目根目录
# cd YourRepo

# 3. 安装核心依赖
pip install -r requirements.txt

```

---

## 三、 快速开始 (Quick Start)

为方便快速体验，本项目已将处理完毕的终极结构化数据内置于 `output_csv/` 目录下。但由于文件过大无法上传到这里，您可以直接运行以下两种模式的实时推理系统：

### 模式一：工程部署版 (推荐，实机演示极度稳定)

该版本剔除了开源数据带来的模态干扰，并加入了**空间盆骨中心归一化算法**。它专为单目摄像头直接部署而设计，抗干扰能力极强。

```bash
cd 02_工程部署版_纯净单目
python deploy_live.py

```

> **操作说明：** 运行后站在摄像头前，保证全身入镜即可体验丝滑的实时动作反馈。

### 模式二：学术验证版 (高精度跨模态微调)

该版本展示了复杂的跨模态数据融合成果。采用基础模型预训练+高精度数据微调（Fine-tuning）策略，离线测试准确率达 92.25%，并搭载了特征标准化（StandardScaler）模块。

```bash
cd 01_学术验证版_含NTU融合
python academic_live.py

```

---

## 四、 数据管线与复现指南 (Data Pipeline)

本项目的 `00_数据清洗脚本(Data_Processing)` 目录包含了完整的数据提炼流变史（Data Lineage）。

**脚本功能概览：**

* `step1_isolate_ntu.py`: 从海量源文件中物理隔离 5 类目标动作。
* `step2_parse_ntu_csv.py`: 解析 `.skeleton` 文本，提取 25 点初始骨骼坐标。
* `step3_semantic_align.py`: **本项目核心重构代码**，实现 12 个肢体大关节的严格物理语义对齐。

**如何从零复现数据预处理 (Optional)：**
由于原始 NTU RGB+D 数据集体积庞大（>100GB），本项目未打包上传原始文件。如果您希望从零跑通完整的预处理管线，请按以下步骤操作：

1. 访问 ROSE Lab 官方网站，申请并下载 **NTU RGB+D Dataset**。
2. 将解压后的 `.skeleton` 文件夹路径配置到 `step1_isolate_ntu.py` 的 `SOURCE_DIR` 变量中。
3. 依次运行 `step1` 至 `step3`，系统将自动在 `output_csv/` 目录下生成最终对齐的 CSV 资产。
*(注：仅使用模型预测或二次训练，无需执行此步骤)*

---

## 五、 核心目录结构

```text
├── 00_数据清洗脚本(Data_Processing)    # 全链路数据提取与跨模态对齐脚本
├── 01_学术验证版_含NTU融合(Offline_Training) # 92%高分模型训练、推理代码与权重
├── 02_工程部署版_纯净单目(RealTime_Deployment) # 实机演示专属抗抖动推理系统
├── output_csv                          # 结构化核心数据资产 (开箱即用)
├── requirements.txt                    # 依赖清单
└── README.md                           # 项目说明文档

```

---
