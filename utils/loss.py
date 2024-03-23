import torch
import torch.nn.functional as F

# Define content loss (MSE loss)
def content_loss(input, target):
    return F.mse_loss(input, target, reduction='sum')

# Define style loss
def style_loss(style_encoding_activation, output_encoding_activations):
    """
    output_encoding_activations: the output of the full model being reencoded
    style_encoding_activation: the initial style image being encoded
    """
    loss = 0.0
    # Iterate over each layer
    #for output_features, style_features in zip(output_encoding_activations, style_encoding_activation):
    for (key_o, output_features), (key_s, style_features) in zip(output_encoding_activations.items(), style_encoding_activation.items()):
        # Compute mean and standard deviation along the batch dimension
        gen_mean = torch.mean(output_features, dim=[2, 3]) # dim=[2,3] because we have to match IN statistics.
        style_mean = torch.mean(style_features, dim=[2, 3])

        gen_std = torch.std(output_features, dim=[2, 3])
        style_std = torch.std(style_features, dim=[2, 3])
        
        # Compute L2 distance for mean and standard deviation
        mean_loss = F.mse_loss(gen_mean, style_mean, reduction='sum')
        std_loss = F.mse_loss(gen_std, style_std, reduction='sum')

        # Accumulate the loss
        loss += mean_loss + std_loss
    
    return loss
