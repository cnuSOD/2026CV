import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import pickle

# ==========================================
# 1. 物理关节映射字典 (必须与训练时严格一致)
# ==========================================
# 我们提取的 MediaPipe 12个核心大关节序号
MP_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
labels_list = ['cepingju', 'dantui', 'kaihetiao', 'kuoxiong', 'shendun']


# ==========================================
# 2. 核心网络结构
# ==========================================
class ActionLSTM(nn.Module):
    def __init__(self, input_size=36, hidden_size=64, num_classes=5):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


if __name__ == "__main__":
    # ==========================================
    # 3. 极速加载模型与量纲拉平器
    # ==========================================
    print(" [INFO] 正在加载 LSTM 动作识别大脑...")
    model = ActionLSTM()
    model.load_state_dict(torch.load("action_model.pth"))
    model.eval()

    print(" [INFO] 正在加载 StandardScaler 量纲拉平器...")
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        print(" [SUCCESS] 环境就绪！")
    except Exception as e:
        print(f" [ERROR] 找不到 scaler.pkl，请确认是否已运行 step3_train_aligned.py: {e}")
        exit()

    # ==========================================
    # 4. 初始化 MediaPipe 与 摄像头
    # ==========================================
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    frame_buffer = []

    # UI 防抖控制变量
    display_text = "Waiting..."
    display_conf = 0.0
    hold_counter = 0

    print("\n终极学术混合版实时推理系统已启动！")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            # 提取 12 个核心点
            core_features = []
            for idx in MP_INDICES:
                core_features.extend([lm[idx].x, lm[idx].y, lm[idx].z])

            # 实时计算盆骨中心并平移
            pelvis_x = (core_features[18] + core_features[21]) / 2.0
            pelvis_y = (core_features[19] + core_features[22]) / 2.0
            pelvis_z = (core_features[20] + core_features[23]) / 2.0

            for i in range(12):
                core_features[i * 3] -= pelvis_x
                core_features[i * 3 + 1] -= pelvis_y
                core_features[i * 3 + 2] -= pelvis_z

            # 用持久化的 Scaler 消除传感器差异
            scaled_core_features = scaler.transform([core_features])[0]
            frame_buffer.append(scaled_core_features)

            if len(frame_buffer) > 30:
                frame_buffer.pop(0)

            if len(frame_buffer) == 30:
                motion_variance = np.var(frame_buffer, axis=0).mean()

                # 获取 MediaPipe 原始的左右脚踝 Y 坐标 (27是左脚踝，28是右脚踝)
                # MediaPipe 中 Y 轴是比例，0.0 在最顶端，1.0 在最底端
                left_ankle_y = lm[27].y
                right_ankle_y = lm[28].y
                ankle_diff = abs(left_ankle_y - right_ankle_y)

                # 只有方差极小 【并且】 双脚高度基本一致时，才判定为真·待机
                # 0.05 大约代表屏幕高度 5% 的高度差，你可以根据情况微调
                if motion_variance < 0.005 and ankle_diff < 0.05:
                    if hold_counter == 0:
                        display_text = "Static (Ready)"
                        display_conf = 100.0
                    else:
                        hold_counter -= 1
                else:
                    # 进入预测逻辑...
                    input_tensor = torch.tensor([frame_buffer], dtype=torch.float32)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)[0]
                        action_idx = torch.argmax(probs).item()
                        confidence = probs[action_idx].item() * 100

                        if confidence >= 60.0:
                            display_text = labels_list[action_idx]
                            display_conf = confidence
                            hold_counter = 30
                        else:
                            if hold_counter == 0:
                                display_text = "Recognizing..."
                                display_conf = confidence
                            else:
                                hold_counter -= 1
        else:
            frame_buffer.clear()
            display_text = "No Person"
            display_conf = 0.0
            hold_counter = 0

        # UI 渲染
        cv2.rectangle(frame, (10, 10), (420, 85), (0, 0, 0), -1)
        color = (0, 255, 255) if display_text == "Recognizing..." else (0, 255, 0)

        cv2.putText(frame, f"Action: {display_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Conf: {display_conf:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.imshow('Final Academic Live Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()