# 通过加入可学习参数，动态调整物理特征和视觉特征的融合权重

import torch
import torch.nn as nn


class PhysicalOriginalFeatureFusion(nn.Module):
    def __init__(self, visual_dim, physical_dim):
        super().__init__()
        
        self.visual_proj = nn.Conv2d(visual_dim, visual_dim, 1)  
        self.physical_proj = nn.Conv2d(physical_dim, visual_dim, 1)  

        self.attention = nn.Sequential(
            nn.Conv2d(visual_dim*2, visual_dim, 1),  
            nn.BatchNorm2d(visual_dim),
            nn.Sigmoid()
        )

    def forward(self, x_visual, x_physical):
        v = self.visual_proj(x_visual)  
        p = self.physical_proj(x_physical)  

        fusion = torch.cat([v, p], dim=1)  
        attn = self.attention(fusion)  

        out = x_visual * (1 - attn) + (v * p) * attn
        # out = x_visual + attn * p

        return  out


class PhysicalOriginalGTFeatureFusion(nn.Module):
    def __init__(self, visual_dim, physical_dim):
        super().__init__()

        self.visual_proj = nn.Conv2d(visual_dim, visual_dim, 1)  
        self.physical_proj = nn.Conv2d(physical_dim, visual_dim, 1)

        # DCNv4/v2
        self.mpred = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim // 4, 1, 1, 0),
            nn.BatchNorm2d(visual_dim // 4),
            nn.SiLU(),
            nn.Conv2d(visual_dim // 4, 1, 1, 1, 0),
            nn.Sigmoid()
        ) 
    
    def forward(self, x_visual, x_physical):
        v = self.visual_proj(x_visual)          # 视觉特征
        p = self.physical_proj(x_physical)      # 物理特征

        # pre = v + p
        # m_pre = self.mpred(pre)
        # p_refined = v * m_pre
        

        m_pre = self.mpred(x_visual)            # 通过视觉特征学习目标权重
        p_refined = p * (1 + m_pre)             # 根据学习到的权重调整物理特征，弱化背景高频影响
        
        out = x_visual + p_refined              # 将调整后的物理特征与原始视觉特征融合，增强目标特征表达

        return out, m_pre

        