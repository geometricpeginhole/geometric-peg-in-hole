import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmDecoder(nn.Module):
    def __init__(self, layers, device) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.layers = nn.LSTM(input_size=layers[0], hidden_size=layers[1], num_layers=len(layers) - 2,
                              batch_first=True, device=device)
        self.mlp = nn.Linear(layers[1], layers[-1], device=device)
    
    def forward(self, x):
        B, O, _ = x.shape
        x, _ = self.layers(x)
        # B, O, hidden_size
        x = x[:, [-1]]
        # B, 1, hidden_size
        x = self.mlp(x)
        # B, 1, action_size
        return x
