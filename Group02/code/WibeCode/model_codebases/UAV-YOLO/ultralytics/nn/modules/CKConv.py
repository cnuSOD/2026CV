from __future__ import annotations

import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class CKConv(nn.Module):
    def __init__(self, c1, c2, kk=[3, 5], s=1):
        super().__init__()

        if not isinstance(kk, list) or not all(ki in [3, 5, 7, 9] for ki in kk):
            raise ValueError("k must be a list containing 3, 5, and/or 7")

        self.kk = kk
        self.c1 = c1
        self.c2 = c2
        self.s = s

        self.branches = nn.ModuleDict()


        for ki in kk:

            self.branches[f'k{ki}_body'] = Conv(c2, c2//2, (3, 3), s=1, g=c2//2)
            self.branches[f'k{ki}_head_h'] = Conv(c2, c2//2, (1, ki), s=s, p=(0, (ki - 1) // 2), g=c2//2)
            self.branches[f'k{ki}_head_v'] = Conv(c2//2, c2//2, (ki, 1), s=s, p=((ki - 1) // 2, 0), g=c2//2)
            self.branches[f'k{ki}_conv2'] = nn.Conv2d(c2//2, c2, 1, groups=c2//2)

        self.conv_fuse = nn.Conv2d(len(kk) * c2, c2, 1)   # note 1

    def forward(self, x):

        outputs = []

        for ki in self.kk:
            y = self.branches[f'k{ki}_head_h'](x)
            # print("1:", y.shape)
            y = self.branches[f'k{ki}_head_v'](y)
            # print("2:", y.shape)
            ys = self.branches[f'k{ki}_body'](x)
            # print("3:", ys.shape)
            out = ys + y
            # print("out1:", out.shape)
            out = self.branches[f'k{ki}_conv2'](out)
            # print("out2:", out.shape)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        # print("concat:", out.shape)
        out = self.conv_fuse(out)

        return out

# __all__ = ("CKConv",)


# class CKConv(nn.Module):
#     def __init__(self, c1: int, c2: int, k=(3, 5, 7), s: int = 1, e: float = 1.0, act=True):
#         super().__init__()
#         if isinstance(k, int):
#             k = (k,)

#         c_ = max(1, int(c2 * e))
#         self.b1 = nn.ModuleList(
#             nn.Sequential(
#                 Conv(c1, c_, k=(1, ki), s=s, p=(0, ki // 2), act=act),
#                 Conv(c_, c_, k=(ki, 1), p=(ki // 2, 0), act=act),
#             )
#             for ki in k
#         )
#         self.b2 = nn.ModuleList(Conv(c1, c_, k=3, s=s, act=act) for _ in k)
#         self.cv1 = nn.ModuleList(Conv(c_, c_, k=1, act=act) for _ in k)
#         self.cv2 = Conv(c_ * len(k), c2, k=1, act=act)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = [cv1(b1(x) + b2(x)) for b1, b2, cv1 in zip(self.b1, self.b2, self.cv1)]
#         return self.cv2(torch.cat(y, dim=1))


def main():
    x = torch.randn(1, 64, 128, 128)
    model = CKConv(c1=64, c2=64)
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()
