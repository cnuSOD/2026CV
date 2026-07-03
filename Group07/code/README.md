# 基于 L2CS-Net 的视线追踪与注意力分析

这是计算机视觉课程的期末作业。项目主要想实现一个基于普通摄像头的视线追踪程序，并根据注视位置做简单的视觉注意力分析。
一开始我们考虑过使用眼睛和虹膜位置计算视线，但这种方法容易受到光线和头部移动的影响，实际效果不太稳定。后来改为使用预训练的 L2CS-Net 模型。模型可以根据摄像头画面预测视线的 yaw 和 pitch 角度，再通过个人校准把角度转换为屏幕坐标。

## 1. 项目功能

本项目目前支持以下功能：

使用普通摄像头获取实时画面
使用 L2CS-Net 估计视线方向
通过 16 点校准计算屏幕注视位置
显示当前注视点和最近的视线轨迹
将屏幕划分为 3×3 个 AOI 区域
统计各区域的注视次数和注视时间
生成视线轨迹图、热力图和区域转移矩阵
通过持续注视两秒选择屏幕上的数字

流程大概是：摄像头画面 → L2CS-Net → yaw/pitch → 16 点校准 → 屏幕坐标 → AOI 分析
MediaPipe是层级旧的视觉模型已不使用了；
## 2. 项目结构

code/main.py 是程序入口。
code/gaze_tracking/gaze_estimator.py 用来加载 L2CS-Net，并计算 yaw 和 pitch。
code/gaze_tracking/calibration_app.py 是校准界面。
code/gaze_tracking/calibration.py 用来拟合和保存校准结果。
code/gaze_tracking/webcam_app.py 是实时视线追踪程序。
code/gaze_tracking/digit_selection_app.py 是注视数字选择实验。
code/gaze_tracking/analyzer.py 负责计算 fixation、注视时间和区域转移。
code/gaze_tracking/visualization.py 负责生成轨迹图和热力图。

## 3. 环境准备

我们用 Python 环境运行本项目

需要安装依赖：
pip install -r code\requirements.txt

来下载 L2CS-Net 预训练权重文件 `L2CSNet_gaze360.pkl`，并放到下面的位置：
models/L2CSNet_gaze360.pkl



## 4. 快速运行

1.第一次运行需要先进行校准：
python code\main.py --mode calibrate
校准窗口会依次显示 16 个红色目标点。眼睛看着当前目标点，然后按空格开始采集。按 r 可以重新采集当前点，按 q 可以退出。

2.校准完成后，运行实时视线追踪：
python code\main.py --mode webcam
程序会显示摄像头画面和注视点平面。按 q 结束后，分析结果会保存到 outputs 文件夹。

3.运行数字选择实验：
python code\main.py --mode digits --dwell-ms 2000
当视线在同一个数字区域停留两秒时，该数字会被选中。



## 5. 程序的具体运行过程

项目的入口是 code/main.py。程序启动后会先读取命令行参数，根据 mode 选择校准、实时追踪、数字选择或者离线分析。这样几个功能共用同一套视线估计和校准代码，不需要分别运行不同的程序。

1. 视线方向估计
视线估计部分写在 gaze_estimator.py 中。
程序首先创建 GazeEstimator，并通过 L2CS-Net 的 Pipeline 加载 models/L2CSNet_gaze360.pkl。每读取一帧摄像头画面，就调用 estimator.estimate(frame) 进行预测。
L2CS-Net 可能在一帧中检测到多个人脸，因此代码会根据 scores 选择置信度最高的人脸，然后读取该人脸的 yaw 和 pitch。模型原始输出是弧度，代码使用 np.degrees 将其转换为角度。
摄像头画面会有一定抖动，所以 GazeEstimator 使用一个长度为 6 的 deque 保存最近几帧结果，并对 yaw 和 pitch 求平均。estimate 函数最后返回平滑后的视线角度、置信度和人脸关键点。

2. 16 点校准
校准界面写在 calibration_app.py 中。
运行 calibrate 模式后，build_calibration_targets 会在屏幕上生成一个 4×4 的目标点网格。目标点与屏幕边缘之间保留了约 8% 的距离，避免用户注视过于靠近边缘的位置。
用户注视红色目标点并按下空格后，程序默认采集 30 帧视线数据。为了减小偶然误差，每个目标点最终使用 yaw 和 pitch 的中位数作为该点的视线特征，同时记录平均置信度。
16 个点采集完成后，calibration.py 中的 fit_poly2_calibration 会建立下面的输入特征：
1、yaw、pitch、yaw×pitch、yaw²、pitch²。
程序使用 np.linalg.lstsq 分别计算屏幕横坐标和纵坐标的拟合系数，最后把系数保存到 outputs/calibration.json。每个目标点的采样结果还会保存到 calibration_samples.csv，方便检查校准数据是否正常。

3. 实时视线追踪
webcam 模式的主要代码位于 webcam_app.py。
程序先调用 load_calibration 读取 calibration.json。如果没有校准文件，或者读取到的是旧视线方案的校准文件，程序会停止运行并提示重新校准。
随后程序使用 cv2.VideoCapture 打开摄像头。在循环中，每一帧都会经过下面几个步骤：
调用 L2CS-Net 得到 yaw 和 pitch。
调用 calibration.apply，将视线角度转换成屏幕坐标。
使用最近 5 个屏幕坐标再次求平均，减小注视点跳动。
调用 find_aoi，判断注视点位于哪个 AOI。
保存当前时间、屏幕坐标、yaw、pitch、AOI 和置信度。
更新摄像头窗口和 Attention Plane 窗口。
AOI 的定义在 aoi.py 中。build_grid_aois 默认将屏幕分成 3 行、3 列，共 9 个区域，每个区域使用 R1C1、R1C2 等名称表示。find_aoi 会依次检查注视点是否位于某个区域内。如果坐标超出屏幕，则返回 OUT。
Attention Plane 会显示 AOI 网格、当前注视点以及最近 120 个注视点组成的轨迹。用户按下 q 后，摄像头循环结束，程序开始整理和分析已经记录的数据。

4. 注视事件和 AOI 分析
分析代码位于 analyzer.py。
build_fixations 会先按照时间对视线记录排序，然后把连续落在同一个 AOI 的样本合并为一个注视候选。如果持续时间不少于 120 毫秒，并且区域不是 OUT，就将其记录为一次 fixation。
每次 fixation 会保存开始时间、结束时间、持续时间、平均坐标和样本数量。
summarize_aois 会按照 AOI 进行分组，计算：
注视次数
总注视时间
第一次注视持续时间
平均注视持续时间
build_transition_matrix 会读取 fixation 的先后顺序。如果相邻两次 fixation 位于不同 AOI，就给对应的区域转移次数加一。例如视线从 R1C1 移动到 R2C2，矩阵中 R1C1 到 R2C2 的数值就会增加一次。
visualization.py 使用 Matplotlib 和 Seaborn 生成视线轨迹图、注视热力图和 AOI 转移矩阵图。

5. 数字选择实验
digits 模式的代码位于 digit_selection_app.py。
程序先根据数字大小和 target_padding 计算每个数字的可注视区域，然后生成能够放入屏幕的网格位置，并使用 random.shuffle 随机排列数字。
每一帧得到屏幕注视坐标后，程序会判断当前状态：
NO_GAZE：没有检测到可靠视线
OUT_SCREEN：注视坐标超出屏幕范围
BLANK：注视屏幕空白位置
DIGIT：注视点位于某个数字区域
当视线第一次进入数字区域时，程序记录开始时间。如果之后一直注视同一个数字，就累计 dwell_elapsed_ms。达到设定的停留时间后，该数字被认为已经选中。
选择完成后，程序等待约 900 毫秒，再重新排列数字。逐帧视线状态保存在 digit_gaze_log.csv，最终确认的数字保存在 digit_selection_log.csv。
整个程序的实际数据流程是：
摄像头画面 → L2CS-Net 预测 yaw/pitch → 时间窗口平滑 → 16 点校准映射 → 屏幕坐标平滑 → AOI 或数字区域判断 → fixation 分析 → 保存 CSV 和可视化图片。

## 6. 不足

经测试，改模型看向屏幕边缘会有较大误差，可以尝试增加校准点数量、保持头部稳定，或改善光照条件。同时，如果校准后改变人和摄像头的位置较多，精准度也大幅度下降



## 7. 成员分工

曹昂；1231001003 ；负责眼动估计算法和模型接入
蔡卓霖；1231001002；负责系统运行流程、摄像头追踪和校准功能
张万山；1231001032；负责数据分析、可视化结果和项目文档
