import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

# 必须与提取的 12 个关节一致
JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class ActionLSTM(nn.Module):
    def __init__(self, input_size=36, hidden_size=64, num_classes=5):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


if __name__ == "__main__":
    MODEL_PATH = "deploy_model.pth"  # 就在当前02文件夹里，直接读取
    labels_list = ['cepingju', 'dantui', 'kaihetiao', 'kuoxiong', 'shendun']

    model = ActionLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    frame_buffer = []

    # 新增：UI 平滑与视觉防抖控制变量
    display_text = "Waiting..."
    display_conf = 0.0
    hold_counter = 0  # 驻留倒计时

    print("终极防抖推理系统启动！加入了 60% 置信度拦截。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            # 实时几何归一化：把盆骨移到中心
            pelvis_x = (lm[23].x + lm[24].x) / 2.0
            pelvis_y = (lm[23].y + lm[24].y) / 2.0

            frame_feats = []
            for j in JOINTS:
                nx = lm[j].x - pelvis_x
                ny = lm[j].y - pelvis_y
                nz = lm[j].z
                frame_feats.extend([nx, ny, nz])

            frame_buffer.append(frame_feats)
            if len(frame_buffer) > 30: frame_buffer.pop(0)

            if len(frame_buffer) == 30:
                motion_variance = np.var(frame_buffer, axis=0).mean()

                # 状态1：人体处于静止状态
                if motion_variance < 0.001:
                    if hold_counter == 0:
                        display_text = "Static (Ready)"
                        display_conf = 100.0
                    else:
                        hold_counter -= 1  # 动作刚做完，让前一个动作的字在屏幕上多留一会

                # 状态2：人体处于运动状态，交给 AI 判定
                else:
                    input_tensor = torch.tensor([frame_buffer], dtype=torch.float32)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)[0]
                        action_idx = torch.argmax(probs).item()
                        confidence = probs[action_idx].item() * 100

                        # 🌟 核心拦截逻辑：置信度必须大于 60% 才认为是有效动作
                        if confidence >= 60.0:
                            display_text = labels_list[action_idx]
                            display_conf = confidence
                            hold_counter = 30  # 刷新文字，并强行驻留 30 帧（约 1 秒）
                        else:
                            # 置信度低于 60%，说明动作不标准或是过渡动作
                            if hold_counter == 0:
                                display_text = "Recognizing..."  # 显示识别中，避免乱跳
                                display_conf = confidence
                            else:
                                hold_counter -= 1  # 保持上一个确定的动作不乱变

        else:
            frame_buffer.clear()
            display_text = "No Person"
            display_conf = 0.0
            hold_counter = 0

        # 渲染黑底黑框，稍微加宽一点以适应 "Recognizing..." 较长的单词
        cv2.rectangle(frame, (10, 10), (420, 85), (0, 0, 0), -1)

        # 动态改变文字颜色：如果是 Recognizing... 用黄色提醒，否则用绿色
        color = (0, 255, 255) if display_text == "Recognizing..." else (0, 255, 0)

        cv2.putText(frame, f"Action: {display_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Conf: {display_conf:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.imshow('Robust Live Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()