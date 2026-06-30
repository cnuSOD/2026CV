import cv2
import mediapipe as mp
import pandas as pd
import os
import glob

# --- 1. 配置路径 ---
VIDEO_DIR = "../raw_videos"
OUTPUT_DIR = "../output_csv"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. 初始化 MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def process_video_with_view(video_path):
    cap = cv2.VideoCapture(video_path)

    # 获取原视频的宽、高、帧率，用于保存带骨架的新视频
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建视频写入器 (保存为 output_动作名_姓名.mp4)
    save_name = "viz_" + os.path.basename(video_path)
    out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, save_name),
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. 检测关键点
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # 2. 在画面上绘制骨架（可视化核心）
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # 关节点绿色
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # 连线红色
            )

        # 3. 显示窗口
        cv2.imshow('Skeleton Visualization', frame)

        # 4. 保存带骨架的帧
        out.write(frame)

        # 按 Esc 退出预览
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# --- 3. 批量执行 ---
all_video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
all_extracted_data = []

# 定义 CSV 列名
columns = ['label', 'user', 'frame_idx']
for i in range(33):
    columns.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])

for v_path in all_video_files:
    data = process_video_with_view(v_path)
    if data:
        # 保存单个视频的 CSV (备份用)
        single_df = pd.DataFrame(data, columns=columns)
        save_name = os.path.basename(v_path).replace(".mp4", ".csv")
        single_df.to_csv(os.path.join(OUTPUT_DIR, save_name), index=False)
        # 加入汇总列表
        all_extracted_data.extend(data)

# --- 4. 汇总总表 ---
if all_extracted_data:
    final_df = pd.DataFrame(all_extracted_data, columns=columns)
    final_df.to_csv(os.path.join(OUTPUT_DIR, "Final_Dataset_GroupXX.csv"), index=False)
    print("\n" + "=" * 30)
    print(f"处理完成！")
    print(f"共处理视频数: {len(all_video_files)}")
    print(f"总帧数(样本量): {len(final_df)}")
    print(f"汇总文件存至: {OUTPUT_DIR}/Final_Dataset_GroupXX.csv")
    print("=" * 30)