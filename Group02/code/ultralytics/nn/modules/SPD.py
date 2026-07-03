"""Space-to-Depth downsampling module."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("SPD",)


class SPD(nn.Module):
    """Space-to-Depth module.

    Rearranges spatial information into channel dimension. For the default
    ``block_size=2``, the output shape changes from ``(B, C, H, W)`` to
    ``(B, 4C, H/2, W/2)``, which matches the SPD diagram.
    """

    def __init__(self, c1: int | None = None, c2: int | None = None, block_size: int = 2, k: int = 3, act: bool = True):
        """Initialize the SPD module.

        Args:
            c1 (int | None): Input channels. Required when channel mixing is enabled.
            c2 (int | None): Output channels after mixing conv. If omitted, SPD behaves as pure rearrangement.
            block_size (int): Spatial block size used for rearrangement.
            k (int): Kernel size for the post-SPD mixing conv.
            act (bool): Whether to use activation in the post-SPD mixing conv.
        """
        super().__init__()
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, but got {block_size}.")
        self.block_size = block_size
        self.spd = nn.PixelUnshuffle(block_size)

        in_channels = None if c1 is None else c1 * block_size**2
        if c2 is None:
            self.mix = nn.Identity()
            self.out_channels = in_channels
        else:
            if c1 is None:
                raise ValueError("c1 must be provided when c2 is set for SPD channel mixing.")
            self.mix = Conv(in_channels, c2, k, 1, act=act)
            self.out_channels = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange spatial patches into channels and optionally mix them."""
        bs = self.block_size
        b, c, h, w = x.shape
        if h % bs != 0 or w % bs != 0:
            raise ValueError(
                f"SPD expects height and width divisible by {bs}, but got input shape {tuple(x.shape)}."
            )
        return self.mix(self.spd(x))
