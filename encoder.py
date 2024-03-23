import torch
import torch.nn as nn


class Encoder(nn.Module):
    """ 
    VGG encoder until layer Relu 4_1.
    Shape of output : (B, C, H, W) = (B, 512, 2, 2)
    """
    def __init__(self):
        super(Encoder, self).__init__()
        vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
        vgg = nn.Sequential(*list(vgg.features.children())[:31]) # 31 because we take layers up to Relu 4_1
        self.encoder = vgg

    def forward(self, x):
        return self.encoder(x)
