import os
import pandas as pd
import glob

BASE_PATH = "../ntu_selected_actions"  
OUTPUT_FILE = "../output_csv/NTU_Extracted_Data.csv"

# 我们要的动作编号
TARGET_ACTIONS = ["A008", "A022", "A024", "A027", "A052"]


def parse_ntu_skeleton(file_path):
    """
    专业解析 NTU .skeleton 文件
    提取每一帧中第一个人的 25 个关键点坐标
    """
    try:
        with open(file_path, 'r') as f:
            frame_count = int(f.readline().strip())  # 第一行是总帧数
            frames_data = []

            for i in range(frame_count):
                body_count = int(f.readline().strip())  # 这一帧有几个人
                if body_count == 0: continue

                # 读取第一个人的信息
                # 后面几行包含：bodyID, clipedEdgest, leftHandState等，我们跳过
                f.readline()
                joint_count = int(f.readline().strip())  # 关节点数（通常是25）

                joints = []
                for j in range(joint_count):
                    joint_info = f.readline().split()
                    # 前三个是 x, y, z 坐标
                    joints.extend([float(joint_info[0]), float(joint_info[1]), float(joint_info[2])])

                # 如果这一帧还有多余的人，跳过他们的行
                if body_count > 1:
                    for _ in range(body_count - 1):
                        f.readline()  # 跳过 body info
                        jc = int(f.readline().strip())
                        for _ in range(jc): f.readline()

                frames_data.append(joints)
            return frames_data
    except Exception as e:
        print(f"解析失败 {file_path}: {e}")
        return None


# --- 2. 批量搜索并提取 ---
# 使用 recursive=True 自动深入子文件夹找所有 .skeleton 文件
search_pattern = os.path.join(BASE_PATH, "**/*.skeleton")
all_files = glob.glob(search_pattern, recursive=True)

print(f"共发现 {len(all_files)} 个原始文件，开始筛选目标动作...")

final_data = []
for f_path in all_files:
    f_name = os.path.basename(f_path)
    # 提取 Axxx 编号
    action_id = f_name[-13:-9]

    if action_id in TARGET_ACTIONS:
        print(f"正在处理动作 {action_id}: {f_name}")
        frames = parse_ntu_skeleton(f_path)
        if frames:
            for idx, f_coords in enumerate(frames):
                # 构造一行：[动作, 文件名, 帧序号, x1, y1, z1, ...]
                final_data.append([action_id, f_name, idx] + f_coords)

# --- 3. 保存结果 ---
if final_data:
    # 动态生成 25 个点的列名
    cols = ['label', 'file_source', 'frame_idx']
    for i in range(25):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])

    df = pd.DataFrame(final_data, columns=cols)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n成功！开源数据已提取至: {OUTPUT_FILE}")
    print(f"总计提取样本行数: {len(df)}")
else:
    print("\n[错误] 未提取到任何数据！请检查 TARGET_ACTIONS 列表中的编号在文件夹中是否存在。")