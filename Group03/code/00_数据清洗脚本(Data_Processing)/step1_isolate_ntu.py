# ==========================================
# 本脚本为【前期数据流清洗工具】。
# 由于原始 NTU 数据集体积过大，本项目未打包提交原始文件及中间态文件夹。
# 供模型直接使用的最终对齐产物 (Aligned_MediaPipe.csv / Aligned_NTU.csv)
# 已生成并内置于 ../output_csv/ 目录，可直接运行 01/02 文件夹中的核心代码。
# ==========================================

import os
import shutil
import glob

# ==========================================
# 1. 路径配置 (使用相对路径)
# ==========================================
# 源文件夹：你电脑上存放原始 NTU 数据的解压目录
SOURCE_DIR = "D:/1059-3/nturgbd_skeletons_s001_to_s017"

# 目标文件夹：提取出来的文件将单独存放在这里
TARGET_DIR = "../ntu_selected_actions"

# 我们需要的 5 类动作编号 (A008深蹲, A022单腿, A024扩胸, A027开合跳, A052侧平举)
TARGET_ACTIONS = ["A008", "A022", "A024", "A027", "A052"]


def extract_target_skeletons():
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f" [INFO] 创建目标隔离文件夹: {TARGET_DIR}")

    # 使用 glob 递归搜索所有的 .skeleton 文件
    search_pattern = os.path.join(SOURCE_DIR, "**/*.skeleton")
    print(f" [INFO] 正在扫描源目录，请稍候...")
    all_files = glob.glob(search_pattern, recursive=True)

    if not all_files:
        print(" [ERROR] 未在源路径找到任何 .skeleton 文件，请检查 SOURCE_DIR 路径是否正确！")
        return

    print(f" [INFO] 总共找到 {len(all_files)} 个原始文件，开始执行特征动作筛选...")

    copy_count = 0
    for file_path in all_files:
        file_name = os.path.basename(file_path)

        # NTU 文件名标准格式: S001C001P001R001A052.skeleton
        # 动作编号固定在倒数第13到倒数第9个字符之间
        action_id = file_name[-13:-9]

        if action_id in TARGET_ACTIONS:
            target_path = os.path.join(TARGET_DIR, file_name)

            # 使用 shutil.copy2 复制文件，同时保留文件的原始元数据
            shutil.copy2(file_path, target_path)
            copy_count += 1

            # 进度监控：每复制 500 个打印一次
            if copy_count % 500 == 0:
                print(f" 进度: 已安全提取 {copy_count} 个目标文件...")

    print("=" * 50)
    print(f" 第一阶段数据隔离完成！")
    print(f" 共计将 {copy_count} 个干净的目标动作文件提取至: {TARGET_DIR}")
    print(" 接下来可以放心地对这个独立文件夹进行语义映射了。")
    print("=" * 50)


if __name__ == "__main__":
    extract_target_skeletons()