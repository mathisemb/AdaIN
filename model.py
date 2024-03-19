import torch
import torch.nn as nn
from encoder import Encoder
from adain import AdaptiveInstanceNorm
from decoder import Decoder
from tqdm import tqdm
from copy import deepcopy
from loss import content_loss, style_loss

class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        self.encoder = Encoder()
        self.adain = AdaptiveInstanceNorm()
        self.decoder = Decoder()

    def forward(self, content, style):
        # Extract content and style features
        content_features = self.encoder(content)
        style_features = self.encoder(style)

        # Perform AdaIN
        stylized_features = self.adain(content_features, style_features)

        # Decode the stylized features
        stylized_img = self.decoder(stylized_features)

        return stylized_img

# training function
def train_decoder(model, content_loader, style_loader, nb_epochs, learning_rate, lam):
    """
    model: style transfer model containing an encoder, an adain and a decoder
    content_loader: dataloader on the content images dataset
    style_loader: dataloader on the sytle images dataset
    nb_epochs: number of epochs of the training
    learning_rate: learning rate of the gradient descent
    lam: style loss weight
    """

    # we just train the decoder
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate)

    # hook encoder activations for style loss
    encoder_activations = {}
    def get_activation(layer_num: int):
        def hook(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
            encoder_activations[str(layer_num)] = output.clone().detach()
            # print('activation_hook -> layer_num: {}'.format(layer_num))
        return hook
    enc_style_layers = [3, 10, 17, 30] # vgg19 relu1_1, relu2_1, relu3_1, relu4_1
    for layer_pos in enc_style_layers:
        model.encoder.encoder[layer_pos].register_forward_hook(get_activation(layer_pos))
    
    # training loop
    for epoch in tqdm(range(nb_epochs), desc="Epochs"):
        for content_label_batch, style_label_batch in zip(content_loader, style_loader):
            # we are just interested in the images, not their labels
            content_batch = content_label_batch[0]
            style_batch = style_label_batch[0]

            # --- RUN THE MODEL ---
            # We first encode the content image and the style image
            content_features = model.encoder(content_batch)
            style_features = model.encoder(style_batch)

            # We need to save of the activations for the style loss (it was the last being encoded)
            style_activations = deepcopy(encoder_activations)
            encoder_activations = {} # reinitialize activations

            stylized_features = model.adain(content_features, style_features)

            stylized_img = model.decoder(stylized_features)

            # We save output for the content loss and activations for the style loss
            output_features = model.encoder(stylized_img)
            output_activations = deepcopy(encoder_activations)
            encoder_activations = {}

            # Content loss
            Lc = content_loss(output_features, stylized_features)

            # Style loss
            Ls = style_loss(style_activations, output_activations)

            # Loss
            L = Lc + lam*Ls

            #print("loss:", L)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
        
            print("loss =", L.item())
        
    return 0
