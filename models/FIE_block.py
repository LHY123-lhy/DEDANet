import torch
import math
import torch.nn as nn
import torch.nn.functional as F



class ForegroundInformationEnhancer(nn.Module):
    def __init__(self, in_d=None, out_d=None,conv_bias=True,):
        super(ForegroundInformationEnhancer, self).__init__()
        if in_d is None:
            in_d = [64,128,256,512]
        self.in_d = in_d
        self.mid_d = 128

        if out_d is None:
            out_d = [64,128,256,512]
        self.out_d = out_d

        # scale 1
        self.conv_scale1_c1 = nn.Sequential(
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        

        # scale 2
        self.conv_scale1_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
      

        # scale 3
        self.conv_scale1_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
       

        # scale 4
        self.conv_scale1_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, groups=self.mid_d),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

        # fusion
        self.conv_aggregation_s1 = FeatureFusionModule(self.mid_d * 4, self.in_d[0], self.out_d[0])
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 4, self.in_d[1], self.out_d[1])
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 4, self.in_d[2], self.out_d[2])
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 4, self.in_d[3], self.out_d[3])
       

    def forward(self, c1, c2, c3, c4, ):
        # scale 1
        c1_s1 = self.conv_scale1_c1(c1)
        c1_s2 = self.conv_scale2_c1(c1)
        c1_s3 = self.conv_scale3_c1(c1)
        c1_s4 = self.conv_scale4_c1(c1)
        

        # scale 2
        c2_s1 = F.interpolate(self.conv_scale1_c2(c2), scale_factor=(2, 2), mode='bilinear')
        c2_s2 = self.conv_scale2_c2(c2)
        c2_s3 = self.conv_scale3_c2(c2)
        c2_s4 = self.conv_scale4_c2(c2)
       

        # scale 3
        c3_s1 = F.interpolate(self.conv_scale1_c3(c3), scale_factor=(4, 4), mode='bilinear')
        c3_s2 = F.interpolate(self.conv_scale2_c3(c3), scale_factor=(2, 2), mode='bilinear')
        c3_s3 = self.conv_scale3_c3(c3)
        c3_s4 = self.conv_scale4_c3(c3)
       
        # scale 4
        c4_s1 = F.interpolate(self.conv_scale1_c4(c4), scale_factor=(8, 8), mode='bilinear')
        c4_s2 = F.interpolate(self.conv_scale2_c4(c4), scale_factor=(4, 4), mode='bilinear')
        c4_s3 = F.interpolate(self.conv_scale3_c4(c4), scale_factor=(2, 2), mode='bilinear')
        c4_s4 = self.conv_scale4_c4(c4)

        s1 = self.conv_aggregation_s1(torch.cat([c1_s1, c2_s1, c3_s1, c4_s1], dim=1), c1)
        s2 = self.conv_aggregation_s2(torch.cat([c1_s2, c2_s2, c3_s2, c4_s2], dim=1), c2)
        s3 = self.conv_aggregation_s3(torch.cat([c1_s3, c2_s3, c3_s3, c4_s3], dim=1), c3)
        s4 = self.conv_aggregation_s4(torch.cat([c1_s4, c2_s4, c3_s4, c4_s4], dim=1), c4)
       
        return s1, s2, s3, s4


class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.fuse_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.fuse_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fuse_d, self.fuse_d, kernel_size=3, stride=1, padding=1, groups=self.fuse_d),
            nn.BatchNorm2d(self.fuse_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_d),
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))
        return c_out

