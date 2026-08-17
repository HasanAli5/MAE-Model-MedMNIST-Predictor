import torch
from torch import nn
from .mae import AutoEncoder
from .basic_modules import CrossAttensionTransformerBlock

"""
Hybrid Cross Attension Reducer Design

we are using the cross attended summary parameter tokens
and the original Global Average Pooled query tokens as residual connection
to produce predictions

"""

class DecoderPredictiorCrossReducer(nn.Module):

    def __init__(self,
                 embed_dim:int=512,n_labels=9,num_head:int=8):
        super().__init__()
        # dimension
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(0.1)
        # back to initial feature size
        self.fc = nn.Linear(embed_dim, n_labels)
        self.cross_reduce = CrossAttensionTransformerBlock(embed_dim=embed_dim,num_heads=num_head)
        self.summarise_token = nn.Parameter(torch.empty(1,1,self.embed_dim))
        # the intial distibution of the q tokens on startup of fresh model
        nn.init.normal_(self.summarise_token, std=.02)
    
    def forward(self,x:torch.Tensor):
        batches = x.shape[0]
        summary_tokens = self.summarise_token.expand(batches,-1,-1)
        fx = self.cross_reduce(summary_tokens,x).flatten(1)

        x = x.mean(dim=1)
        x = fx + x 
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
        self.predictor = predictor if predictor else DecoderPredictiorCrossReducer(n_labels=n_labels)
    
    def forward(self,x):
        x = self.autoencoder(x)
        x = self.predictor(x)
        return x

