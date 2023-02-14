import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

from archs.utils import Conv2d1x1, Conv2d3x3, ShiftConv2d1x1, MeanShift, Upsampler


# LayerNormChannel comes from RCAN
class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)


# ResBlock comes from PPON
class ResBlock(nn.Module):
    def __init__(self, nc):
        super(ResBlock, self).__init__()
        self.c1 = conv_layer(nc, nc, 1, 1, 1)
        self.d1 = conv_layer(nc, nc//2, 1, 1, 1)  # rate=1
        self.d2 = conv_layer(nc, nc//2, 1, 1, 2)  # rate=2

        self.act = nn.LeakyReLU(0.2,True)
        self.c2 = conv_layer(nc, nc, 1, 1, 1)  # 128-->64

    def forward(self, input):
        output1 = self.act(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)

        add1 = d1 + d2

        # combine = torch.cat([d1, add1, add2, add3,add4,add5], 1)
        combine = torch.cat([d1, add1], 1)
        output2 = self.c2(self.act(combine))
        output = input + output2.mul(0.2)

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
                ws = int(a / b)
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

class ELCS(nn.Module):

    def __init__(self,planes: int = 60,shifts: int=4,window_sizes: tuple =[4,8,16]):
        super(ELCS, self).__init__()
        self.left = nn.Sequential(
            LayerNormChannel(planes),
            # Conv2d3x3(planes, planes),
            ShiftConv2d1x1(planes,planes),
            nn.ReLU(inplace=True),
            ResBlock(planes)
        )

        self.right = nn.Sequential(
            LayerNormChannel(planes),
            GMSA(planes, shifts, window_sizes)
        )

        # self.ln1 = LayerNormChannel(planes)
        # self.ln2 = LayerNormChannel(planes)
        #
        # self.conv = Conv2d3x3(planes,planes)
        # self.relu = nn.ReLU(inplace=True)
        # self.resConv = ResBlock(planes)
        #
        # self.gmsa = GMSA(planes,shifts,window_sizes)

    def forward(self,x):
        a = self.left(x) + x
        output = a + self.right(a)

        return output


@ARCH_REGISTRY.register()
class ELP(nn.Module):
    # upscale: int, num_in_ch: int, num_out_ch: int, task: str,
    def __init__(self,upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int = 180, num_blocks: int = 24,
                 shifts: int = 4, window_sizes=[4, 8, 16]) -> None:
        super(ELP, self).__init__()

        self.scale = upscale
        self.window_sizes = window_sizes
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)
        # m_head = [nn.Conv2d(num_in_ch, planes, kernel_size=3, stride=1, padding=1)]
        # self.head = nn.Sequential(*m_head)
        self.head = Conv2d3x3(num_in_ch, planes)

        m_body = [ELCS(planes, shifts, self.window_sizes) for _ in range(num_blocks)]
        self.body = nn.Sequential(*m_body)

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)

    def forward(self,x):
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
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x