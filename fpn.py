"""Feature Pyramid Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from typing import List, Dict
from collections import OrderedDict

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels:List[int], out_channel:int=256):
        super().__init__()
        self.out_channel = out_channel
        self.out_layers = nn.ModuleList()
        self.lateral_layers = nn.ModuleList()

        for in_channel in in_channels:
            self.lateral_layers.append(nn.Conv2d(in_channel, out_channel, 1))
            self.out_layers.append(nn.Conv2d(out_channel, out_channel, 3, padding=1))

        """Init conv layers weight & biases"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x:Dict[str, Tensor]) -> Dict[str, Tensor]:
        fpn_output = []
        layer_names = list(x.keys())
        layer_outputs = list(x.values())

        last_lateral_output = self.lateral_layers[-1](layer_outputs[-1])
        fpn_output.append(self.out_layers[-1](last_lateral_output))

        for i in range(len(layer_outputs) - 2, -1, -1):
            lateral_output = self.lateral_layers[i](layer_outputs[i])
            prev_interp = F.interpolate(last_lateral_output, size=lateral_output.shape[-2:], mode='nearest')
            last_lateral_output = lateral_output + prev_interp

            fpn_output.insert(0, self.out_layers[i](last_lateral_output))

        return OrderedDict([(k, v) for k, v in zip(layer_names, fpn_output)])

if __name__ == "__main__":
    dict = OrderedDict()
    dict['f1'] = torch.randn(1, 64, 112, 112)
    dict['f2'] = torch.randn(1, 128, 64, 64)
    dict['f3'] = torch.randn(1, 256, 32, 32)
    fpn = FeaturePyramidNetwork([64,128,256], 256)
    output = fpn(dict)
    print([(k, v.shape) for k, v in output.items()])

