import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
        vgg = nn.Sequential(*list(vgg.features.children())[:35])
        self.encoder = vgg

    def forward(self, x):
        return self.encoder(x)
