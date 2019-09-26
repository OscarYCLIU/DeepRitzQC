import numpy as np
import torch
import torch.nn as nn

# class cubicActivation(nn.Module):
#     def __init__(self):
#         super(cubicActivation, self).__init__()
#     def forward(self, x):
#         return
        # return torch.clamp(x, min=0.0)


class Residual_Block(nn.Module):
    def __init__(self, m):
        super(Residual_Block, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(m, m),
            nn.Tanh(),
            # cubicActivation(),
            nn.Linear(m, m),
            nn.Tanh(),
            # cubicActivation(),
        )    
    def forward(self, x):
        return self.net(x) + x
        
class model(nn.Module):
    def __init__(self, m, block_num, out):
        super(model, self).__init__()
        self.resblocks = self._make_layer(Residual_Block, m, block_num)
        self.outlayer = nn.Linear(m, out)
    def _make_layer(self, block, m, block_num):
        layers = []
        for i in range(block_num):
            layers.append(block(m))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.outlayer(self.resblocks(x))
    
def weights_init(w, std):
    classname = w.__class__.__name__
    if classname.find('Linear') != -1:
        stdv = std / np.sqrt(w.weight.size(1))
        w.weight.data.uniform_(-stdv, stdv)
        if w.bias is not None:
            w.bias.data.uniform_(-stdv, stdv)