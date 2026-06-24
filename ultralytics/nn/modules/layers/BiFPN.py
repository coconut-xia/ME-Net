import torch
import torch.nn as nn
import torch.nn.functional as F
from ..conv import Conv
from ..block import Bottleneck

__all__ = [
    "AMA", 
    "DWSConv",
]

class AMA(nn.Module):
    """
    AMA handles four scales: P2, P3, P4, P5
    implements full bidirectional feature fusion
    """
    def __init__(self, c1, c2, n=3, e=0.5, dropout=0.0):
        super().__init__()
        self.n = n  # AMA repeat count
        self.channels = c2
        self.dropout = dropout  # dropout rate

        # if c1 is a list, use the first element (assuming all inputs have been unified to the same channel count)
        if isinstance(c1, list):
            # ensure all input channels are equal
            assert all(c == c1[0] for c in c1), f"All input channels should be the same, got {c1}"
            in_channels = c1[0]
        else:
            in_channels = c1

        # if input channels differ from output channels, add channel adaptation layers
        self.channel_adapters = nn.ModuleList()
        if in_channels != c2:
            for i in range(4):  # P2, P3, P4, P5
                self.channel_adapters.append(Conv(in_channels, c2, 1, 1))
        else:
            self.channel_adapters = None

        # create n AMA layers
        self.ama_layers = nn.ModuleList()
        for i in range(n):
            self.ama_layers.append(SingleAMALayer(c2, dropout=dropout))

        # optional: add dropout between AMA layers
        if dropout > 0:
            self.layer_dropout = nn.Dropout2d(dropout)
        else:
            self.layer_dropout = nn.Identity()
    
    def forward(self, inputs):
        """
        Input: when inputs is a tensor, assumed to be concat result that needs splitting
              when inputs is a list, directly [P2, P3, P4, P5]
        Output: [P2_out, P3_out, P4_out, P5_out]
        """
        if isinstance(inputs, torch.Tensor):
            # if input is a single tensor, may need to split
            # but in your config, input should be a list
            raise ValueError("AMA expects a list of tensors as input")
        
        features = inputs
        
        # pass through multiple AMA layers
        for i, ama_layer in enumerate(self.ama_layers):
            features = ama_layer(features)
            # apply dropout between layers (except the last)
            if i < len(self.ama_layers) - 1:
                features = [self.layer_dropout(f) for f in features]
        
        return features

class SingleAMALayer(nn.Module):
    """Single AMA layer — one BiFPN unit handling P2, P3, P4, P5."""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        
        # Top-Down path convolutions
        self.conv_p4_td = DWSConv(channels, channels, 3, 1, dropout=dropout)
        self.conv_p3_td = DWSConv(channels, channels, 3, 1, dropout=dropout)
        self.conv_p2_td = DWSConv(channels, channels, 3, 1, dropout=dropout)
        
        # Bottom-Up path convolutions
        self.conv_p2_out = DWSConv(channels, channels, 3, 1, dropout=dropout)
        self.conv_p3_out = DWSConv(channels, channels, 3, 1, dropout=dropout)
        self.conv_p4_out = DWSConv(channels, channels, 3, 1, dropout=dropout)
        self.conv_p5_out = DWSConv(channels, channels, 3, 1, dropout=dropout)
        
        # Learned fusion weights (Top-Down)
        
        self.p4_td_w = nn.Parameter(torch.ones(2))
        self.p3_td_w = nn.Parameter(torch.ones(2))
        self.p2_td_w = nn.Parameter(torch.ones(2))
        
        # Learned fusion weights (Bottom-Up)
        self.p2_out_w = nn.Parameter(torch.ones(2))    # P2 fuses P2_in + P2_td
        self.p3_out_w = nn.Parameter(torch.ones(3))    # P3 fuses P3_in + P3_td + P2_out↓
        self.p4_out_w = nn.Parameter(torch.ones(3))    # P4 fuses P4_in + P4_td + P3_out↓
        self.p5_out_w = nn.Parameter(torch.ones(2))    # P5 fuses P5_in + P4_out↓
        
        # for resizing feature maps
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(3, 2, 1)
        
        # optional: add dropout after fusion
        if dropout > 0:
            self.fusion_dropout = nn.Dropout2d(dropout)
        else:
            self.fusion_dropout = nn.Identity()
    
    def forward(self, inputs):
        """
        Input: [P2, P3, P4, P5]
        Output: [P2_out, P3_out, P4_out, P5_out]
        """
        p2_in, p3_in, p4_in, p5_in = inputs
        
        # ============ Top-Down Path (P5 → P4 → P3 → P2) ============
        # P4_td = weighted_fusion(P4_in, upsample(P5_in))
        p4_td = self._weighted_fusion(
            [p4_in, self.upsample(p5_in)],
            self.p4_td_w
        )
        p4_td = self.fusion_dropout(p4_td)  # dropout after fusion
        p4_td = self.conv_p4_td(p4_td)
        
        # P3_td = weighted_fusion(P3_in, upsample(P4_td))
        p3_td = self._weighted_fusion(
            [p3_in, self.upsample(p4_td)],
            self.p3_td_w
        )
        p3_td = self.fusion_dropout(p3_td)
        p3_td = self.conv_p3_td(p3_td)
        
        # P2_td = weighted_fusion(P2_in, upsample(P3_td))
        p2_td = self._weighted_fusion(
            [p2_in, self.upsample(p3_td)],
            self.p2_td_w
        )
        p2_td = self.fusion_dropout(p2_td)
        p2_td = self.conv_p2_td(p2_td)
        
        # ============ Bottom-Up Path (P2 → P3 → P4 → P5) ============
        # P2_out = weighted_fusion(P2_in, P2_td)
        p2_out = self._weighted_fusion(
            [p2_in, p2_td],
            self.p2_out_w
        )
        p2_out = self.fusion_dropout(p2_out)
        p2_out = self.conv_p2_out(p2_out)
        
        # P3_out = weighted_fusion(P3_in, P3_td, downsample(P2_out))
        p3_out = self._weighted_fusion(
            [p3_in, p3_td, self.downsample(p2_out)],
            self.p3_out_w
        )
        p3_out = self.fusion_dropout(p3_out)
        p3_out = self.conv_p3_out(p3_out)
        
        # P4_out = weighted_fusion(P4_in, P4_td, downsample(P3_out))
        p4_out = self._weighted_fusion(
            [p4_in, p4_td, self.downsample(p3_out)],
            self.p4_out_w
        )
        p4_out = self.fusion_dropout(p4_out)
        p4_out = self.conv_p4_out(p4_out)
        
        # P5_out = weighted_fusion(P5_in, downsample(P4_out))
        p5_out = self._weighted_fusion(
            [p5_in, self.downsample(p4_out)],
            self.p5_out_w
        )
        p5_out = self.fusion_dropout(p5_out)
        p5_out = self.conv_p5_out(p5_out)
        
        return [p2_out, p3_out, p4_out, p5_out]
    
    def _weighted_fusion(self, inputs, weights):
        """weighted fusion using Softmax for stability"""
        # reshape to (n, 1, 1, 1) so TensorRT can parse the Softmax node (requires >=2D)
        w = weights.reshape(-1, 1, 1, 1)
        normalized_weights = F.softmax(w, dim=0)
        out = sum(x * normalized_weights[i] for i, x in enumerate(inputs))
        return out


class DWSConv(nn.Module):
    """Depthwise Separable Convolution """
    def __init__(self, c1, c2, k=3, s=1, act=True, dropout=0.0):
        super().__init__()
        self.dconv = nn.Conv2d(c1, c1, k, s, k//2, groups=c1, bias=False)
        self.pconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        
        # add dropout after activation
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = nn.Identity()
    
    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)  # apply dropout after activation
        return x