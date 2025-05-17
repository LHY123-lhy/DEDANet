import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import warnings

from . import resnet 
from models.vmamba import VSSM
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, constant_init
from typing import Optional, Union, Sequence
from .LVC_blocks import LVCBlock
from models.FIE_block import *
from models.wav import create_wavelet_filter,wavelet_transform,inverse_wavelet_transform


class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )
#尺度变换
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


######### Module ###########

class CA(nn.Module):
    def __init__(self, channel, reduction):
        super(CA, self).__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.conv(x)
        return x * y

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DCAB(nn.Module):
    def __init__(self, dim, reduction, wt_levels: int = 0,conv_bias: bool = True,):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0),
            CA(dim,reduction),
        )

    def forward(self, x):
        x = self.block(x) + x
        return x

class MSCA_CAA(BaseModule):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            reduction = 32,
            h_kernel_size: int = 7,
            v_kernel_size: int = 7,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.channels = channels
        self.avg_pool = nn.AvgPool2d(9, 1, 4)
        self.max_pool = nn.MaxPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0,0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (0,0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.conv3 = ConvModule(channels, channels, 3, 1, 1, 
                                 norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.act = nn.Sigmoid()

        self.q = nn.Sequential(
            ConvModule(channels, channels, 1, 1, 0,norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Sigmoid(),
        )
        self.k = nn.Sequential(
            ConvModule(channels, channels, 1, 1, 0,norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Sigmoid(),
        )    
        self.v = nn.Sequential(
            nn.Conv2d(self.channels*2, self.channels, kernel_size=3, padding=1, groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
        )
        self.v1 = LVCBlock(in_channels=channels, out_channels=channels, num_codes=64) 
        self.ff = nn.Sequential(
            DCAB(self.channels,reduction),
        )
        self.ln2 = LayerNorm(self.channels, eps=1e-6, data_format="channels_first")

        self.conv0 =nn.Sequential(
            nn.Conv2d(self.channels*2, self.channels, kernel_size=1),
        )
    def forward(self, x,y):
        b,c,h,w = x.shape
        x1 = self.conv1(self.avg_pool(x))
        xh = self.h_conv(x1)#b,c,64,58
        xw = self.v_conv(x1)#b,c,58,64

        y1 = self.conv0(y)
        y1 = self.conv1(self.avg_pool(y1))
        yh = self.h_conv(y1)#b,c,64,54
        yw = self.v_conv(y1)#b,c,54,64
  

        f1 = torch.matmul(xh, yw)
        f1 = self.q(f1)  # b, c, h, w

        f2 = torch.matmul(yh, xw)
        f2 = self.k(f2)  # b, c, h, w

        xt = self.v1(y)
        out = x * f1 * f2 + xt
        out = self.ln2(out)
        out = self.ff(out)
        
        return out
    

class DifferenceAggregator(nn.Module):
    def __init__(self,channel_dim):
        super(DifferenceAggregator, self).__init__()

        self.channel_dim=channel_dim
   
        self.coAtt_1 = MSCA_CAA(channels=channel_dim)
        
    def forward(self,x1,x2):
        B,C,H,W = x1.shape
        f_d = torch.abs(x1-x2) #B,C,H,W
        f_c = torch.cat((x1, x2), dim=1)  # B,2C,H,W
        out = self.coAtt_1(f_d,f_c)

        return out
    


###############################################################################
# OUR-Net
###############################################################################


class Backbone(torch.nn.Module):
    def __init__(self, args, input_nc, output_nc,
                 resnet_stages_num=5,
                 output_sigmoid=False, if_upsample_2x=True):
        """
        
        """
        super(Backbone, self).__init__()


        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        #
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        #
        # self.resnet_stages_num = resnet_stages_num
        #
        self.if_upsample_2x = if_upsample_2x

        #
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()


class DEDANet(Backbone):
    def __init__(self, args, input_nc, output_nc,):
        super(DEDANet, self).__init__(args, input_nc, output_nc)
        self.stage_dims = [64, 128, 256, 512]
        self.n = [0,2,4,6]
        self.output_nc=output_nc
        self.backbone = resnet.resnet18(pretrained=True)

        self.BDA1  =  DifferenceAggregator(self.stage_dims[0])
        self.BDA2  =  DifferenceAggregator(self.stage_dims[1])
        self.BDA3  =  DifferenceAggregator(self.stage_dims[2])
        self.BDA4  =  DifferenceAggregator(self.stage_dims[3])
        
        self.mamba = VSSM(patch_size=4, in_chans=3, num_classes=2, depths=[2, 2, 9, 2], depths_decoder=[2, 2, 2, 1],
                 dims=[64,128,256,512], dims_decoder=[512,256,128,64], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False)
        
        self.FIE = ForegroundInformationEnhancer(self.stage_dims, self.stage_dims)
      
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8,mode='bilinear')
  
        self.conv4 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

        self.cls4 = nn.Conv2d(self.stage_dims[3], 1, kernel_size=1)
        self.cls3 = nn.Conv2d(self.stage_dims[2], 1, kernel_size=1)
        self.cls2 = nn.Conv2d(self.stage_dims[1], 1, kernel_size=1)
        self.cls1 = nn.Conv2d(self.stage_dims[0], 1, kernel_size=1)

        self.conv_final1 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x1, x2):
        #res18
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        x1_0,x1_1,x1_2,x1_3 = f1
        x2_0,x2_1,x2_2,x2_3 = f2
   
        #FIE
        x1_0,x1_1,x1_2,x1_3 = self.FIE(x1_0,x1_1,x1_2,x1_3)
        x2_0,x2_1,x2_2,x2_3 = self.FIE(x2_0,x2_1,x2_2,x2_3)
        
        #BDA
        d1 = self.BDA1(x1_0, x2_0)
        d2 = self.BDA2(x1_1, x2_1)
        d3 = self.BDA3(x1_2, x2_2)
        d4 = self.BDA4(x1_3, x2_3)
       
        p4,p3,p2,p1 = self.mamba.forward_features_up1(d4,d3,d2,d1)
     
        mask4 = self.cls4(p4)
        mask4 = F.interpolate(mask4, scale_factor=(32, 32), mode='bilinear')
        mask4 = torch.sigmoid(mask4)

        mask3 = self.cls3(p3)
        mask3 = F.interpolate(mask3, scale_factor=(16, 16), mode='bilinear')
        mask3 = torch.sigmoid(mask3)

        mask2 = self.cls2(p2)
        mask2 = F.interpolate(mask2, scale_factor=(8, 8), mode='bilinear')
        mask2 = torch.sigmoid(mask2)

        mask1 = self.cls1(p1)
        mask1 = F.interpolate(mask1, scale_factor=(4, 4), mode='bilinear')
        mask1 = torch.sigmoid(mask1)
 
        p4_up = self.upsample8(p4)
        p4_up =self.conv4(p4_up)

        p3_up = self.upsample4(p3)
        p3_up = self.conv3(p3_up)

        p2_up = self.upsample2(p2)
        p2_up = self.conv2(p2_up)

        p = p1+p2_up+p3_up+p4_up
        p_up =self.upsample4(p)

        output = self.conv_final1(p_up)
        output = torch.sigmoid(output)

        return output, mask4,mask3,mask2,mask1


