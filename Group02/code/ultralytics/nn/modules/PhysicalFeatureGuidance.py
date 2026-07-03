import torch
import torch.nn as nn
import torch.nn.functional as F

# from .CKConv import CKConv

class PhysicalAdapter(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.p3 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x):
        out = self.p3(x)
        return out


class ExtractPhysicalFeature(torch.nn.Module):
    def __init__(self, cutoff_ratio=0.1):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio


    def forward(self, img):
        rows, cols = img.shape[2], img.shape[3]

        f = torch.fft.fft2(img, dim=(-2, -1))
        fshift = torch.fft.fftshift(f, dim=(-2, -1))

        crow, ccol = rows // 2, cols // 2
        y = torch.arange(rows, device=img.device, dtype=img.dtype)
        x = torch.arange(cols, device=img.device, dtype=img.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        D = torch.sqrt((xx - ccol) ** 2 + (yy - crow) ** 2)

        D0 = self.cutoff_ratio * min(rows, cols) / 2
        
        H = 1 - torch.exp(-(D ** 2) / (2 * (D0 ** 2 + 1e-12)))

        H = H.unsqueeze(0).unsqueeze(0)
        fshift_filtered = fshift * H

        f_ishift = torch.fft.ifftshift(fshift_filtered, dim=(-2, -1))
        return torch.fft.ifft2(f_ishift, dim=(-2, -1)).abs()


class PhysicalFeatureGuidance(nn.Module):
    """
        x -> FFT high-pass -> iFFT -> 1x1 
    """

    def __init__(self, channels: int, cutoff_ratio: float = 0.1, demension: int = 2):
        super().__init__()
        self.channels = channels
        self.cutoff_ratio = cutoff_ratio
        # 更契合小目标的卷积或FEM HFP；CBS和小目标的Concat或res
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        # self.proj = CKConv(channels, channels, kk=[3, 5], s=1)  # kk改成1，7

    def _build_high_pass_mask(self, x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape[-2], x.shape[-1]
        crow, ccol = rows // 2, cols // 2

        y = torch.arange(rows, device=x.device, dtype=x.dtype)
        xx = torch.arange(cols, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, xx, indexing="ij")
        dist = torch.sqrt((xx - ccol) ** 2 + (yy - crow) ** 2)

        cutoff = max(float(self.cutoff_ratio), 1e-4) * min(rows, cols) / 2.0
        mask = 1.0 - torch.exp(-(dist ** 2) / (2 * (cutoff ** 2 + 1e-12)))
        return mask.unsqueeze(0).unsqueeze(0)

    def extract_high_frequency(self, x: torch.Tensor) -> torch.Tensor:
        f = torch.fft.fft2(x, dim=(-2, -1))
        fshift = torch.fft.fftshift(f, dim=(-2, -1))
        mask = self._build_high_pass_mask(x)
        filtered = fshift * mask
        ishift = torch.fft.ifftshift(filtered, dim=(-2, -1))
        return torch.fft.ifft2(ishift, dim=(-2, -1)).real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high = self.extract_high_frequency(x)
        high = self.proj(high)
        return high
    

class HaarDWT(nn.Module):
    """
    Pure PyTorch Haar DWT.
    Input:  [B, C, H, W]
    Output:
        LL:   [B, C, H/2, W/2]
        high: [B, 3C, H/2, W/2] = concat(LH, HL, HH)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape

        # 如果 H/W 是奇数，补齐
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode="reflect")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) * 0.5
        LH = (x00 - x01 + x10 - x11) * 0.5
        HL = (x00 + x01 - x10 - x11) * 0.5
        HH = (x00 - x01 - x10 + x11) * 0.5

        high = torch.cat([LH, HL, HH], dim=1)
        return LL, high
    

class WaveletPhysicalBranch(nn.Module):
    """
    用 Wavelet 提取局部高频信息，替代 FFT 高频分支。
    输出尺寸默认是输入的一半。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = HaarDWT()

        self.high_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        _, high = self.dwt(x)
        high_feat = self.high_proj(high)
        return high_feat
    

class PhysicalBranch(nn.Module):
    def __init__(self, cutoff_ratio=0.1):
        super().__init__()
        # self.guidance = PhysicalFeatureGuidance(channels=3, cutoff_ratio=cutoff_ratio)
        self.guidance = WaveletPhysicalBranch(in_channels=3, out_channels=16)
        # self.pa1 = PhysicalAdapter(c1=3, c2=16)
        self.pa2 = PhysicalAdapter(c1=16, c2=32)
        self.pa3 = PhysicalAdapter(c1=32, c2=64)
        self.pa4 = PhysicalAdapter(c1=64, c2=128)

    def forward(self, x):
        x = self.guidance(x)   # [B, 3, H, W]
        # p1 = self.pa1(x)       # [B, 16, H/2, W/2]
        p2 = self.pa2(x)      # [B, 32, H/4, W/4]
        p3 = self.pa3(p2)      # [B, 64, H/8, W/8]
        # for YOLO11
        p4 = self.pa4(p3)      # [B, 128, H/16, W/16]
        return p2, p3, p4
    

def main():
    img = torch.rand(1, 3, 640, 640)
    # epf = ExtractPhysicalFeature(cutoff_ratio=0.1)
    # output = epf(img)
    # print(output.shape)

    block = PhysicalBranch(cutoff_ratio=0.1)
    enhanced = block(img)
    print(enhanced[0].shape)
    print(enhanced[1].shape)


if __name__ == "__main__":
    main()
