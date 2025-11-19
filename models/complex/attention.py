import math
import torch as th
import torch.nn as nn
from torchcvnn.nn import MultiheadAttention as ComplexMultiheadAttention
from torchcvnn.nn import LayerNorm as ComplexLayerNorm

from ..nn import checkpoint
from .complex import (
    complex_conv_nd,
)

# ytxie: In this module, time t is not the input.
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by" \
               f" num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = ComplexLayerNorm(channels, dtype=th.complex64)
        
        # torchcvnn MultiheadAttention handles Q,K,V projections internally
        self.attention = ComplexMultiheadAttention(
            embed_dim=channels,
            num_heads=self.num_heads,
            batch_first=False,  # Uses (seq, batch, features) format
            dtype=th.complex64
        )
        
        # Output projection with zero initialization
        self.proj_out = complex_conv_nd(1, channels, channels, 1)
        for p in self.proj_out.parameters():
            p.detach().zero_()

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # [B, C, T]
        residual = x
        
        # LayerNorm over channel dimension: [B, C, T] -> [B, T, C]
        x_ln = x.permute(0, 2, 1)  # [B, T, C]
        x_ln = self.norm(x_ln)     # Normalize over C (last dim)
        
        # Reshape for torchcvnn MultiheadAttention: [B, T, C] -> [T, B, C]
        x_attn = x_ln.permute(1, 0, 2)  # [T, B, C]
        
        # Apply complex multi-head attention (query=key=value for self-attention)
        h, _ = self.attention(x_attn, x_attn, x_attn)
        
        # Reshape back: [T, B, C] -> [B, C, T]
        h = h.permute(1, 2, 0)
        
        h = self.proj_out(h)
        return (residual + h).reshape(b, c, *spatial)


# ytxie: This module is only used in EncoderUNetModel.
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        # Initialize positional embedding as complex-valued directly
        self.positional_embedding = nn.Parameter(
            (th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5).to(th.complex64)
        )
        self.num_heads = embed_dim // num_heads_channels
        
        # Use torchcvnn MultiheadAttention
        self.attention = ComplexMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            batch_first=False,
            dtype=th.complex64
        )
        self.c_proj = complex_conv_nd(1, embed_dim, output_dim or embed_dim, 1)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :]  # NC(HW+1)
        # Reshape for attention: [B, C, T] -> [T, B, C]
        x = x.permute(2, 0, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0)  # [T, B, C] -> [B, C, T]
        x = self.c_proj(x)
        return x[:, :, 0]

