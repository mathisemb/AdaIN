import torch
import torch.nn as nn
from encoder import Encoder
from utils.adain import AdaptiveInstanceNorm
from decoder import Decoder
from tqdm import tqdm
from copy import deepcopy
from utils.loss import content_loss, style_loss

import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StyleTransfer(nn.Module):
    def __init__(self, lr = 3e-4, lam = 2., saving_path: str = 'model_checkpoints/Adain/'):
        super(StyleTransfer, self).__init__()
        self.encoder = Encoder().to(device)
        self.adain = AdaptiveInstanceNorm().to(device)
        self.decoder = Decoder().to(device)

        self.lr = lr
        self.lam = lam

        # we just train the decoder
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        
        self.current_epoch = 0
        
        self.LOSS = [[], []]

        self.saving_path = saving_path

    def update_paths(self):
        self.LOSS_path = f"{self.saving_path}loss_history_epoch_{self.current_epoch}.pkl"
        self.optimizer_path = f"{self.saving_path}optimizer_epoch_{self.current_epoch}.pt"
        self.decoder_path = f"{self.saving_path}decoder_epoch_{self.current_epoch}.pt"

    def forward(self, content, style):
        # Extract content and style features
        content_features = self.encoder(content)
        style_features = self.encoder(style)

        # Perform AdaIN
        stylized_features = self.adain(content_features, style_features)

        # Decode the stylized features
        stylized_img = self.decoder(stylized_features)

        return stylized_img
    
    # Training function
    def train_decoder(self, content_loader, style_loader, nb_epochs):
        """
        model: style transfer model containing an encoder, an adain and a decoder
        content_loader: dataloader on the content images dataset
        style_loader: dataloader on the sytle images dataset
        nb_epochs: number of epochs of the training
        learning_rate: learning rate of the gradient descent
        lam: style loss weight
        """
        self.decoder.train()
        # hook encoder activations for style loss
        encoder_activations = {}
        def get_activation(layer_num: int):
            def hook(self: nn.Module, input: torch.Tensor, output: torch.Tensor):
                encoder_activations[str(layer_num)] = output.clone().detach()
                # print('activation_hook -> layer_num: {}'.format(layer_num))
            return hook
        enc_style_layers = [1, 6, 11, 20] # vgg19 relu1_1, relu2_1, relu3_1, relu4_1
        for layer_pos in enc_style_layers:
            self.encoder.encoder[layer_pos].register_forward_hook(get_activation(layer_pos))
        
        # training loop
        tqdm_bar = tqdm(range(self.current_epoch, self.current_epoch + nb_epochs), desc="Epochs")
        for epoch in tqdm_bar :
            content_loss_count = 0.
            style_loss_count = 0.
            for content_label_batch, style_label_batch in zip(content_loader, style_loader):
                # We are just interested in the images, not their labels
                content_batch = content_label_batch[0].to(device)
                style_batch = style_label_batch[0].to(device)
                # --- RUN THE MODEL ---
                # We first encode the content image and the style image
                # During encoding, hooks are activated and fill encoder_activations.
                content_features = self.encoder(content_batch)
                style_features = self.encoder(style_batch)

                # We need to save of the activations for the style loss (it was the last being encoded)
                style_activations = deepcopy(encoder_activations)
                encoder_activations = {} # reinitialize activations

                stylized_features = self.adain(content_features, style_features)

                stylized_img = self.decoder(stylized_features)

                # We save output for the content loss and activations for the style loss
                output_features = self.encoder(stylized_img)
                output_activations = deepcopy(encoder_activations)
                encoder_activations = {}

                # Content loss
                Lc = content_loss(output_features, stylized_features)

                # Style loss
                Ls = style_loss(style_activations, output_activations)
                
                # Loss
                L = Lc + self.lam*Ls

                #print("loss:", L)

                self.optimizer.zero_grad()
                L.backward()
                self.optimizer.step()

                content_loss_count += Lc.item()
                style_loss_count += Ls.item()

            
            tqdm_bar.set_description(f'Epoch : {epoch + 1} Content Loss : {content_loss_count:.3f} Style Loss : {style_loss_count:.3f}')
            self.LOSS[0].append(content_loss_count)
            self.LOSS[1].append(style_loss_count)
        
        self.current_epoch = self.current_epoch + nb_epochs

    def update_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def save(self): 

        self.update_paths()
        # Save model parameters
        torch.save(self.decoder.state_dict(), self.decoder_path)
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
        # Save loss history
        with open(self.LOSS_path, "wb") as f:
            pickle.dump(self.LOSS, f)

    def load(self, epoch):
        """ 
        epoch : corresponds to the checkpoint you want to load.
        """
        # Set initial paths
        self.current_epoch = epoch
        self.update_paths()

        # Load model parameters
        self.decoder.load_state_dict(torch.load(self.decoder_path, map_location=device))
        # Load optimizer state
        self.optimizer.load_state_dict(torch.load(self.optimizer_path))
        # Load loss history
        with open(self.LOSS_path, "rb") as f:
            self.LOSS = pickle.load(f)

        


