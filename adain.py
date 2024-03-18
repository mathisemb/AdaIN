import torch
import torch.nn as nn

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNorm, self).__init__()

    def forward(self, content_feat, style_feat):
        # Compute mean and std deviation of the content feature
        content_mean = torch.mean(content_feat, dim=(2, 3), keepdim=True)
        content_std = torch.std(content_feat, dim=(2, 3), keepdim=True)

        # Compute mean and std deviation of the style feature
        style_mean = torch.mean(style_feat, dim=(2, 3), keepdim=True)
        style_std = torch.std(style_feat, dim=(2, 3), keepdim=True)

        # Normalize the content feature with the style statistics
        normalized_feat = (content_feat - content_mean) / content_std

        # Scale and shift the normalized feature using style statistics
        stylized_feat = normalized_feat * style_std + style_mean

        return stylized_feat
