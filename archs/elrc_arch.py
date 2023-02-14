
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

from archs.utils import Conv2d1x1, Conv2d3x3, ShiftConv2d1x1, MeanShift, Upsampler




# CA Layer comes from RCAN
class CALayer(nn.Module):
    def __init__(self, channel=32, reduction=16):
        super(CALayer, self).__init__()
        # 平均全局池化 ：feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 特征通道下采样和上采样 --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        output = self.conv_du(y)
        return output * x

# shift-Conv comes from ELAN
class ShiftConv2d(nn.Module):
    def __init__(self,in_channel=32,out_channel=32):
        super(ShiftConv2d, self).__init__()
        self.in_channel = in_channel
        self.weight = nn.Parameter(torch.zeros(in_channel,1,3,3),requires_grad=False)
        self.n_div = 5
        g =in_channel // self.n_div
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0 # left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0 # right
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0 # up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0 # down
        self.weight[4 * g: ,0, 1, 1] = 1.0 # identity

        self.conv = nn.Conv2d(in_channel,out_channel,1)

    def forward(self,x):
        y = F.conv2d(input=x,weight=self.weight,bias=None,stride=1,padding=1,groups=self.in_channel)
        output = self.conv(y)
        return output


class ELRCAB(nn.Module):
    def __init__(self,channel=64,reduction=16):
        super(ELRCAB, self).__init__()
        # self.conv_a = nn.Conv2d(channel,channel // 2, 1,  bias=True)
        # self.conv_b = nn.Conv2d(channel,channel // 2, 1,  bias=True)

        self.calayer = CALayer(channel ,reduction)
        self.sf_conv = ShiftConv2d(channel,channel)
        # self.combine = nn.Conv2d(channel,channel, kernel_size=3, padding=1)
    def forward(self,x):
        # ka = self.conv_a(x)
        # kb = self.conv_b(x)
        # ka = kb = self.conv_a(x)
        # ya = self.calayer(ka)
        # yb = self.sf_conv(kb)
        # yb = self.sf_conv(kb)
        # output = self.combine(torch.cat([ya,yb],1))
        y = self.sf_conv(x)
        output = self.calayer(y)
        return output

# gmsa comes form ELAN
class GMSA(nn.Module):
    def __init__(self,channel=64,shifts=4,window_sizes=[4,8,16]):
        super(GMSA, self).__init__()
        # self.channel = channel
        self.shifts = shifts
        self.window_sizes = window_sizes

        self.split_chns = [channel * 2 //3,channel * 2 //3,channel * 2 //3]
        self.project_inp = nn.Sequential(
            nn.Conv2d(channel,channel*2,kernel_size=1),
            nn.BatchNorm2d(channel*2)
        )
        self.project_out = nn.Conv2d(channel,channel,kernel_size=1)

    def forward(self,x):
        b,c,h,w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x,self.split_chns,dim=1)
        ys = []
        atnss = []

        xss = []
        qss = []
        vss = []

        for id,el in enumerate(xs):
            wsize = self.window_sizes[id]
            if self.shifts > 0:
                el = torch.roll(el, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                xss.append(el)
            qq, vv = rearrange(
                el, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
                qv=2, dh=wsize, dw=wsize
            )
            qss.append(qq)
            vss.append(vv)

        idx = [0,1,2,1,2,0,2,0,1]

        i = 0
        while i<9:
            q = qss[idx[i]]
            k = qss[idx[i+1]]
            v = vss[idx[i+2]]

            a, _, _ = q.size()
            b, _, _ = k.size()
            c, _, _ = v.size()

            if(a > b):
                ws = int(a/b)
                k = rearrange( k, 'b (c wsize) h  -> (b wsize) c h',wsize=ws)
            else:
                ws = int(b / a)
                k = rearrange( k, '(b wsize) c h  -> b (c wsize) h',wsize=ws)

            if (a > c):
                ws = int(a / c)
                v = rearrange(v, 'b (c wsize) h  -> (b wsize) c h', wsize=ws)
            else:
                ws = int(c / a)
                v = rearrange(v, '(b wsize) c h  -> b (c wsize) h', wsize=ws)

            atn = (q @ k.transpose(-2, -1))
            atn = atn.softmax(dim=-1)
            y = (atn @ v)
            i = i + 3
            atnss.append(y)

        for id, el in enumerate(atnss):
            wsize = self.window_sizes[id]
            # print("前",el.shape)
            y = rearrange(
                el, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                h=h // wsize, w=w // wsize, dh=wsize,dw=wsize
            )
            # print("后",y.shape)
            if self.shifts > 0:
                y = torch.roll(y, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            ys.append(y)

        y = torch.cat(ys,dim=1)
        output = self.project_out(y)
        return output


class Body(nn.Module):
    def __init__(self,channel=64,shifts=4,window_sizes=[4,8,16],reduction=16):
        super(Body, self).__init__()
        self.elrcab = ELRCAB(channel,reduction)
        self.gmsa = GMSA(channel,shifts,window_sizes)

    def forward(self,x):
        y = self.elrcab(x)
        output = self.gmsa(x)
        return output
        # return y

@ARCH_REGISTRY.register()
class ELRC(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int = 60,num_blocks: int = 24,
                window_sizes= [4, 8, 16],reduction: int = 16) -> None:
        super(ELRC, self).__init__()
        # shifts=4,reduction=16
        self.scale = upscale
        self.window_sizes = window_sizes
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)


        m_head = [nn.Conv2d(num_in_ch, planes, kernel_size=3, stride=1, padding=1)]
        m_body = [Body(planes, 4, self.window_sizes, reduction) for _ in range(num_blocks)]
        # m_body = Body(planes, 4, self.window_sizes, reduction)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)
    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x = self.sub_mean(x)
        x = x1 = self.head(x)
        x = self.body(x)
        x = self.tail(x + x1)
        output = self.add_mean(x)
        return output[:, :, 0:H * self.scale, 0:W * self.scale]

    def check_image_size(self, x):
        b, c, h, w = x.size()
        # print("ELAN: ==>", b, c, h, w)
        wsize = self.window_sizes[0]
        # print("waimian",wsize)
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
            # print("limian", wsize)
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


