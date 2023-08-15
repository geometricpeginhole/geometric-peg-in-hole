import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpDecoder(nn.Module):
    def __init__(self, layers, device) -> None:
        super().__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)
        self.to(device)
    
    def forward(self, x):
        B, O, D = x.shape
        x = torch.reshape(x, (B, -1))
        x = self.layers(x)
        x = torch.reshape(x, (B, 1, -1))
        return x
