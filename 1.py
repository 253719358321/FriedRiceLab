# q = [1,5,8]
# v = [2,4,6]
# idx = [0,1,2,1,2,0,2,0,1]
# i = 0
# while i<9:
#     print(q[idx[i]] * q[idx[i+1]] * v[idx[i+2]])
#     i = i+3
#     print("+++++++")
# i=+1
# print(i)
import torch
import torch.nn as nn
from einops import rearrange
k = nn.Parameter(torch.zeros([2048, 16 , 20]))
print(k.size())
q = rearrange(k,  '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                h=4, w=4, dw=16
            )
# q = rearrange( k, '(b wsize dsize) c h  -> b (c wsize) h dsize',wsize=4,dsize=4)
print(q.size())
# a,_,_ = k.size()
# b,_,_ = q.size()
# print(int(a/b))