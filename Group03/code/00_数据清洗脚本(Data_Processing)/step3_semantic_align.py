import pandas as pd
import numpy as np

# ==========================================
# 1. 物理关节映射字典
# ==========================================
# 统一标准名称：双肩、双肘、双腕、双髋、双膝、双踝 (12个大关节)
STANDARD_JOINTS = [
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'
]

# MediaPipe 对应的序号 (共12个)
MP_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# NTU Kinect 对应的序号 (共12个)
NTU_INDICES = [4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]


def align_datasets():
    print("开始执行跨模态数据集的语义对齐...")

    # --- 处理 MediaPipe 自采数据 ---
    print("\n[1] 正在重构 MediaPipe 数据 (剔除脸部、手部与可见度v)...")
    mp_df = pd.read_csv("../output_csv/Final_Dataset_GroupXX.csv")
    mp_aligned_data = []

    for i, row in mp_df.iterrows():
        # 提取 label (由于 NTU 那边还是 A0xx，为了省事，我们可以之后在训练代码里统一转拼音)
        row_data = [row['label']]
        for idx in MP_INDICES:
            # 只取 x, y, z，精准丢弃 v
            row_data.extend([row[f'x{idx}'], row[f'y{idx}'], row[f'z{idx}']])
        mp_aligned_data.append(row_data)

    # --- 处理 NTU 开源数据 ---
    print("[2] 正在重构 NTU 数据 (剥离多余躯干与末端关节)...")
    ntu_df = pd.read_csv("../output_csv/NTU_Extracted_Data.csv")
    ntu_aligned_data = []

    for i, row in ntu_df.iterrows():
        row_data = [row['label']]
        for idx in NTU_INDICES:
            # NTU 本来就只有 x, y, z
            row_data.extend([row[f'x{idx}'], row[f'y{idx}'], row[f'z{idx}']])
        ntu_aligned_data.append(row_data)

    # --- 生成统一的列名 ---
    # 列名格式: label, L_Shoulder_x, L_Shoulder_y, L_Shoulder_z, ...
    cols = ['label']
    for joint_name in STANDARD_JOINTS:
        cols.extend([f'{joint_name}_x', f'{joint_name}_y', f'{joint_name}_z'])

    # --- 保存全新的、完美对齐的 CSV ---
    mp_final = pd.DataFrame(mp_aligned_data, columns=cols)
    ntu_final = pd.DataFrame(ntu_aligned_data, columns=cols)

    mp_final.to_csv("../output_csv/Aligned_MediaPipe.csv", index=False)
    ntu_final.to_csv("../output_csv/Aligned_NTU.csv", index=False)

    print("\n" + "=" * 50)
    print("语义对齐完成！")
    print(f"生成的 MediaPipe 形状: {mp_final.shape} (1列标签 + 36列标准坐标)")
    print(f"生成的 NTU 形状: {ntu_final.shape} (1列标签 + 36列标准坐标)")
    print("两个 CSV 的列名和物理含义现在达成一致")
    print("=" * 50)


if __name__ == "__main__":
    align_datasets()