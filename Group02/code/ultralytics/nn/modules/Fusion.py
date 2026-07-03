import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from datetime import datetime
import os


class FuseMap(nn.Module):
    def __init__(self):
        """
        scale_param:权重缩放参数
        """
        super(FuseMap, self).__init__()
        self.scale_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """
        前向传播。
        参数:
            X: [new_rgb_fea, new_ir_fea]，包含两个模态的特征图。
                - new_rgb_fea: RGB 模态特征图，形状为 (batch_size, channels, height, width)。
                - new_ir_fea: IR 模态特征图，形状为 (batch_size, channels, height, width)。
        返回:
            fuse_map: 融合权重图，形状为 (batch_size, channels, height, width)。
        """
        # 分离模态特征图
        rgb_fea, ir_fea = X  #  RGB 和 IR 特征图
        # 激活到0-1
        rgb_fea = self.sigmoid(rgb_fea)
        ir_fea = self.sigmoid(ir_fea)
        # 对 RGB 模态特征图进行通道压缩
        rgb_compressed = torch.max(rgb_fea, dim=1)[0].squeeze(1)  # 形状变为 (bs, h, w)
        # 对 IR 模态特征图进行通道压缩
        ir_compressed = torch.max(ir_fea, dim=1)[0].squeeze(1)  # 形状变为 (bs, h, w)
        # print("rgb_compressed:", rgb_compressed, " ir_compressed:", ir_compressed)
        #计算各模态二值化的阈值
        rgb_threshold = rgb_compressed.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2) # (bs, 1, 1)
        ir_threshold = ir_compressed.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2) # (bs, 1, 1)
        rgb_threshold = rgb_threshold.expand(-1, rgb_compressed.shape[1], rgb_compressed.shape[2]) # (bs, h, w)
        ir_threshold = ir_threshold.expand(-1, ir_compressed.shape[1], ir_compressed.shape[2]) # (bs, h, w)
        # print("rgb:", rgb_threshold, " ir:", ir_threshold)

        # 1. 二值化
        rgb_bin = (rgb_compressed > rgb_threshold).float()  # RGB 模态的二值化特征图
        ir_bin = (ir_compressed > ir_threshold).float()  # IR 模态的二值化特征图
        # print("bin", rgb_bin.device)

        # 2. 计算联通区域
        rgb_regions = self.compute_connected_regions(rgb_bin, "rgb")  # RGB 模态的联通区域
        ir_regions = self.compute_connected_regions(ir_bin, "ir")  # IR 模态的联通区域

        # 3. 找到联通区域的重叠部分计算融合权重
        fuse_map = self.fuse_map(rgb_bin, ir_bin, ir_regions, rgb_regions)
        return fuse_map
    
    def compute_connected_regions(self, binary_fea, modality_name):
        """
        计算联通区域，并返回联通区域的标记矩阵。
        参数:
            binary_fea: 二值化后的特征图，形状为 (batch_size, height, width)
            modality_name: 模态名称(如 'rgb' 或 'ir')
        返回:
            regions: 联通区域标记矩阵，形状为 (batch_size, height, width)
        """
        batch_size, height, width = binary_fea.shape
        regions = []
        device = binary_fea.device

        for b in range(batch_size):
            bin_map = binary_fea[b].cpu().numpy()  # 转换为 NumPy 数组
            labeled, _ = label(bin_map)  # 使用 SciPy 的 label 函数计算联通区域
            regions.append(labeled)  # 将每个标记区域保存为列表

            # 可视化并保存图片
            # self.visualize_and_save_region(labeled, b, modality_name)

        # 返回联通区域标记值，
        return torch.tensor(np.array(regions), dtype=torch.int32, device=device)

    def visualize_and_save_region(self, labeled_region, batch_idx, modality_name):
        """
        可视化并保存联通区域的图片。
        参数:
            labeled_region: 联通区域的标记矩阵
            batch_idx: 当前的 batch 索引
            modality_name: 模态名称（如 'rgb' 或 'ir'）
        """
        h, w = labeled_region.shape  # 获取高度和宽度
        plt.figure(figsize=(8, 6))
        plt.imshow(labeled_region, cmap='jet')  # 使用颜色区分不同联通区域
        plt.colorbar()
        plt.title(f"Connected Regions: {modality_name}, Batch {batch_idx}, ({h}x{w})")
        plt.axis("off")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 保存图片
        save_path = os.path.join(
            "feature_save",
            modality_name,
            f"{modality_name}_batch{batch_idx}_{h}x{w}_{timestamp}.png"
        )
        plt.savefig(save_path)
        plt.close()
    
    def fuse_map(self, rgb_bin, ir_bin, ir_regions, rgb_regions):
        """
        计算 RGB 和 IR 模态的融合权重图。

        参数:
            rgb_bin: 形状 (batch_size, h, w) 的 RGB 二值图。
            ir_bin: 形状 (batch_size, h, w) 的 IR 二值图。
            ir_regions: 形状 (batch_size, h, w) 的 IR 联通区域标记矩阵。
            rgb_regions: 形状 (batch_size, h, w) 的 RGB 联通区域标记矩阵。

        返回:
            fuse_map: 形状 (batch_size, c, h, w) 的融合权重图。
        """
        # 计算两个模态的初始权重比例
        rgb_count = rgb_bin.sum(dim=(1, 2)).float()  # (batch_size,)
        ir_count = ir_bin.sum(dim=(1, 2)).float()    # (batch_size,)

        total_count = rgb_count + ir_count + 1e-6

        rgb_weight = rgb_count / total_count
        ir_weight = ir_count / total_count

        # 计算差值
        gap_rgb_05 = (0.5 - rgb_weight)[:, None, None]
        gap_rgb_1 = (1 - rgb_weight)[:, None, None]
        gap_ir_05 = (0.5 - ir_weight)[:, None, None]
        gap_ir_1 = (1 - ir_weight)[:, None, None]
        # print("rgb_weight:", rgb_weight)
        # print("ir_weight:", ir_weight)

        # 计算各联通区域IOU
        rgb_iou_map = self.compute_iou_map(ir_bin, rgb_regions) # (batch_size, h, w)
        ir_iou_map = self.compute_iou_map(rgb_bin, ir_regions) # (batch_size, h, w)

        # 计算最终融合权重
        adjusted_rgb_weight_1 = rgb_weight[:, None, None] + self.scale_param * rgb_iou_map * gap_rgb_05 + self.scale_param * (1 - rgb_iou_map) * gap_rgb_1  # (batch_size, h, w)
        adjusted_ir_weight_1 = 1 - adjusted_rgb_weight_1 # (batch_size, h, w)

        adjusted_ir_weight_2 = ir_weight[:, None, None] + self.scale_param * ir_iou_map * gap_ir_05 + self.scale_param * (1 - ir_iou_map) * gap_ir_1  # (batch_size, h, w)
        adjusted_rgb_weight_2 = 1 - adjusted_ir_weight_2  # (batch_size, h, w)

        # print("scale", self.scale_param, rgb_bin.shape)
        # 平均权重
        adjusted_rgb_weight = (adjusted_rgb_weight_1 + adjusted_rgb_weight_2) / 2
        adjusted_ir_weight = (adjusted_ir_weight_1 + adjusted_ir_weight_2) / 2
        # print("adjusted_rgb_weight:", adjusted_rgb_weight)
        # print("adjusted_ir_weight:", adjusted_ir_weight)

        # 扩展维度以适应特征图
        adjusted_rgb_weight = adjusted_rgb_weight[:, None, :, :]  # (batch_size, 1, h, w)
        adjusted_ir_weight = adjusted_ir_weight[:, None, :, :]  # (batch_size, 1, h, w)

        return [adjusted_rgb_weight, adjusted_ir_weight]

    def compute_iou_map(self, img_bin, img_regions):
        """
        计算IoU，并返回与原图形状一致的 IoU 映射。

        参数:
            img_bin (tensor): 形状为 (batch_size, h, w) 的二值特征图 (0/1)
            img_regions (tensor): 形状为 (batch_size, h, w) 的联通区域标记矩阵 (0,1,2,...)
        
        返回:
            iou_map (tensor): 形状 (batch_size, h, w)，每个像素值代表该区域的 IoU 值
        """
        device = img_bin.device
        batch_size, h, w = img_bin.shape
        iou_map = torch.zeros((batch_size, h, w), device=device)  # 存储 IoU 结果

        for b in range(batch_size):  # 遍历 batch
            unique_regions = torch.unique(img_regions[b])  # 获取当前 batch 的区域索引
            unique_regions = unique_regions[unique_regions > 0]  # 去除背景 (0)

            for region_id in unique_regions:
                regions_mask = (img_regions[b] == region_id)  # 选出该区域的二值掩码
                bin_mask = img_bin[b]

                intersection = torch.logical_and(regions_mask, bin_mask).sum()  # 计算交集
                union = torch.logical_or(regions_mask, bin_mask).sum()  # 计算并集

                iou = intersection / union if union > 0 else 0  # 避免除零错误
                
                # 在 IoU 映射中填充该区域对应的 IoU 值
                iou_map[b][regions_mask] = iou

        return iou_map  # 返回 IoU 映射