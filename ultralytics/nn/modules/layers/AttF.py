# AttF.py - Advanced Cross-Modal Feature Fusion Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..conv import Conv
from ..block import Bottleneck
from ultralytics.nn.modules import C2f

__all__ = [
    "ACF", 
    "LightAttentionFusion",
    "C2f_FCA",
    "MER"
]


class ACF(nn.Module):
    """
    Full attention fusion module for fusing RGB and Events modality features
    Uses cross-modal attention to dynamically adjust weights of different modality features
    """
    def __init__(self, c1, c2, num_heads=8, dropout=0.1):
        """
        Args:
            c1: input channels (total after concat of two modalities)
            c2: output channels
            num_heads: number of multi-head attention heads
            dropout: dropout rate
        """
        super().__init__()
        
        # channels per modality
        self.modal_channels = c1 // 2
        self.c2 = c2
        self.num_heads = num_heads
        
        # feature projection
        self.rgb_proj = Conv(self.modal_channels, c2, 1)
        self.event_proj = Conv(self.modal_channels, c2, 1)
        
        # cross-modal attention
        self.cross_attention = CrossModalAttention(
            c2, 
            num_heads=num_heads,
            dropout=dropout
        )
        
        # self-attention enhancement (lightweight implementation)
        self.spatial_attention = SpatialAttention(c2)
        self.channel_attention = ChannelAttention(c2)
        
        # feature fusion
        self.fusion_conv = nn.Sequential(
            Conv(c2 * 2, c2, 1),
            Conv(c2, c2, 3, p=1)
        )
        
        # learnable gate scaling factor (initialized to 0.5)
        self.gate_scale = nn.Parameter(torch.tensor(0.5))

        # gating mechanism
        self.gate = nn.Sequential(
            Conv(c2 * 2, c2 // 4, 1),
            nn.SiLU(),
            Conv(c2 // 4, c2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        x: can be list [rgb_feat, event_feat] or tensor (concatenated features)
        """
        # handle input format
        if isinstance(x, list):
            rgb_feat, event_feat = x[0], x[1]
        else:
            # if concatenated tensor, need to separate
            rgb_feat = x[:, :self.modal_channels, :, :]
            event_feat = x[:, self.modal_channels:, :, :]
        
        # feature projection
        rgb_proj = self.rgb_proj(rgb_feat)      # [B, c2, H, W]
        event_proj = self.event_proj(event_feat) # [B, c2, H, W]
        
        # cross-modal attention
        rgb_enhanced, event_enhanced = self.cross_attention(rgb_proj, event_proj)
        
        # self-attention enhancement
        rgb_enhanced = self.spatial_attention(rgb_enhanced)
        rgb_enhanced = self.channel_attention(rgb_enhanced)
        event_enhanced = self.spatial_attention(event_enhanced)
        event_enhanced = self.channel_attention(event_enhanced)
        
        # feature concatenation
        concat_feat = torch.cat([rgb_enhanced, event_enhanced], dim=1)
        
        # fusion feature
        fused_feat = self.fusion_conv(concat_feat)
        
        # gating mechanism
        gate_weights = self.gate(concat_feat)
        
        # weighted output (Events are more important, give event_proj higher initial weight)
        output = fused_feat * gate_weights + event_proj * (1 - gate_weights * self.gate_scale)
        
        return output


class LightAttentionFusion(nn.Module):
    """Lightweight attention fusion module - more efficient, suited for shallow features"""
    def __init__(self, c1, c2):
        super().__init__()
        self.modal_channels = c1 // 2
        
        # feature projection
        self.rgb_conv = Conv(self.modal_channels, c2, 1)
        self.event_conv = Conv(self.modal_channels, c2, 1)
        
        # simplified channel attention - uses SE structure
        reduction = 8
        self.se_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2 * 2, c2 * 2 // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(c2 * 2 // reduction, c2 * 2, 1, bias=False),
            nn.Sigmoid()
        )
        
        # lightweight spatial attention
        self.spatial_attention = nn.Sequential(
            Conv(c2 * 2, 1, 7, p=3),
            nn.Sigmoid()
        )
        
        # fusion layer
        self.fusion = Conv(c2 * 2, c2, 3, p=1)
        
        # event feature enhancement (event features are more important)
        self.event_enhance = nn.Parameter(torch.ones(1, c2, 1, 1) * 1.2)

    def forward(self, x):
        # handle input format
        if isinstance(x, list):
            rgb, event = x[0], x[1]
        else:
            rgb = x[:, :self.modal_channels, :, :]
            event = x[:, self.modal_channels:, :, :]
        
        # feature projection
        rgb_feat = self.rgb_conv(rgb)
        event_feat = self.event_conv(event) * self.event_enhance  # enhance event features
        
        # concatenate features
        concat_feat = torch.cat([rgb_feat, event_feat], dim=1)
        
        # channel attention
        channel_weights = self.se_module(concat_feat)
        feat_ca = concat_feat * channel_weights
        
        # spatial attention
        spatial_weights = self.spatial_attention(concat_feat)
        feat_sa = concat_feat * spatial_weights
        
        # combine both attentions
        enhanced_feat = feat_ca + feat_sa
        
        # final fusion
        output = self.fusion(enhanced_feat)
        
        return output


class CrossModalAttention(nn.Module):
    """Cross-modal attention module - used by ACF"""
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projection
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, rgb_feat, event_feat):
        B, C, H, W = rgb_feat.shape
        
        # RGB as Query, Event as Key and Value (Event is more important)
        q_rgb = self.q_proj(rgb_feat).view(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        k_event = self.k_proj(event_feat).view(B, self.num_heads, self.head_dim, H*W)
        v_event = self.v_proj(event_feat).view(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        
        # Event as Query, RGB as Key and Value
        q_event = self.q_proj(event_feat).view(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        k_rgb = self.k_proj(rgb_feat).view(B, self.num_heads, self.head_dim, H*W)
        v_rgb = self.v_proj(rgb_feat).view(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        
        # compute attention
        attn_rgb = torch.matmul(q_rgb, k_event) * self.scale
        attn_rgb = F.softmax(attn_rgb, dim=-1)
        attn_rgb = self.dropout(attn_rgb)
        rgb_enhanced = torch.matmul(attn_rgb, v_event)
        
        attn_event = torch.matmul(q_event, k_rgb) * self.scale
        attn_event = F.softmax(attn_event, dim=-1)
        attn_event = self.dropout(attn_event)
        event_enhanced = torch.matmul(attn_event, v_rgb)
        
        # reshape back to original shape
        rgb_enhanced = rgb_enhanced.transpose(-2, -1).contiguous().view(B, C, H, W)
        event_enhanced = event_enhanced.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        # output projection and residual connection
        rgb_enhanced = self.out_proj(rgb_enhanced) + rgb_feat
        event_enhanced = self.out_proj(event_enhanced) + event_feat
        
        return rgb_enhanced, event_enhanced


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(channels, 1, 7, p=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn = self.conv(x)
        return x * attn


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # use standard nn.Conv2d without BatchNorm
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * (avg_out + max_out)


class FCA(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = PConv3(
            dim,
            n_div,
            pconv_fw_type
        )
        self.attention = CAA(dim)  # Use CAA instead of EMA

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        # x = shortcut + self.attention(self.drop_path(self.mlp(x)))

        x = self.drop_path(self.mlp(x))
        attn_factor = self.attention(x)
        x = shortcut + x * attn_factor  
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class C2f_FCA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FCA(self.c, self.c) for _ in range(n))


class MER(nn.Module):
    """
    Enhanced event filter - implements pixel-level neural processing
    Learns independent event patterns and filtering parameters for each pixel
    """
    def __init__(self, in_channels=3, reduction=2, pixel_wise=True):
        super().__init__()
        self.in_channels = in_channels
        self.pixel_wise = pixel_wise
        hidden_channels = max(in_channels // reduction, 4)
        
        # 1. pixel-level feature extractor - extract local features per pixel
        self.pixel_feature_extractor = nn.Sequential(
            # use 1x1 conv to extract pixel-level features
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            # another 1x1 conv for feature transformation
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. local context extraction - 3x3 conv for local neighborhood info
        self.local_context = nn.Sequential(
            # Depthwise conv - each channel processed independently
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, 
                     groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. pixel-level gating mechanism - generate independent gate weights per pixel
        self.pixel_gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=True),  # output channels same as input
            nn.Sigmoid()
        )
        
        # 4. pixel-level threshold learning - learn event thresholds per location
        if pixel_wise:
            # use 1x1 conv to learn spatially-varying thresholds
            self.threshold_predictor = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, 1, bias=True),
                nn.Sigmoid()  # output threshold in [0,1]
            )
        else:
            # global threshold
            self.threshold = nn.Parameter(torch.tensor(0.5))
        
        # 5. temporal modeling (optional) - capture temporal event patterns
        self.temporal_filter = nn.Sequential(
            # use multi-scale kernels to capture patterns at different temporal scales
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2, 
                     groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )
        
        # 6. adaptive blending weights
        self.alpha = nn.Parameter(torch.tensor(0.3))  # gate strength
        self.beta = nn.Parameter(torch.tensor(0.2))   # temporal feature strength
        
    def forward(self, x):
        """
        Input: [B, in_channels, H, W] - event features
        Output: [B, in_channels, H, W] - filtered event features
        """
        B, C, H, W = x.shape
        
        # 1. extract pixel-level features
        pixel_features = self.pixel_feature_extractor(x)  # [B, hidden, H, W]
        
        # 2. get local context
        local_features = self.local_context(pixel_features)  # [B, hidden, H, W]
        
        # 3. combine pixel features and local context
        combined_features = torch.cat([pixel_features, local_features], dim=1)  # [B, hidden*2, H, W]
        
        # 4. generate pixel-level gate weights
        pixel_weights = self.pixel_gate(combined_features)  # [B, C, H, W]
        
        # 5. apply pixel-level threshold (if enabled)
        if self.pixel_wise:
            # learn threshold for each pixel location
            pixel_threshold = self.threshold_predictor(x)  # [B, 1, H, W]
            # use threshold for soft filtering
            threshold_mask = torch.sigmoid((torch.abs(x).mean(dim=1, keepdim=True) - pixel_threshold) * 10)
        else:
            # use global threshold
            threshold_mask = torch.sigmoid((torch.abs(x).mean(dim=1, keepdim=True) - self.threshold) * 10)
        
        # 6. temporal filtering
        temporal_features = self.temporal_filter(x)  # [B, C, H, W]
        
        # 7. combine all components
        # pixel-level gating + threshold mask + temporal features
        filtered = x * pixel_weights * self.alpha  # pixel-level weighting
        filtered = filtered * threshold_mask  # apply threshold mask
        filtered = filtered + temporal_features * self.beta  # add temporal features
        
        # 8. residual connection
        output = x + filtered
        
        return output

