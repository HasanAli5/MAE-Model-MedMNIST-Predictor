
from torch import nn
import torch
from mae import AutoEncoder
from timm.layers.pos_embed_sincos  import build_sincos2d_pos_embed, build_rotary_pos_embed, apply_rot_embed


class PredictorFeedForward(nn.Module):

    def __init__(self,
                 embed_dim:int=512,hid_dim:int=1024,n_labels=9):
        super().__init__()
        # dimension
        self.embed_dim = embed_dim
        # back to initial feature size
        self.fc = nn.Linear(embed_dim, n_labels)
        # layers
        self.fc1 = nn.Linear(embed_dim,hid_dim)
        self.fc2 = nn.Linear(hid_dim,embed_dim)
        # activation
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x):
        identity = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = x + identity
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    def __init__(self,
                 n_labels,
                 input_shape:tuple=(3,64,64),
                 autoencoder = None,
                 predictor = None):
        
        super().__init__()
        self.autoencoder = autoencoder if autoencoder else AutoEncoder(input_shape)
        self.predictor = predictor if predictor else PredictorFeedForward(n_labels=n_labels)
    
    def forward(self,x):
        x = self.autoencoder(x)
        x = self.predictor(x)
        return x

