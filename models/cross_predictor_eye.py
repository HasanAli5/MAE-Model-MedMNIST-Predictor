import torch
from torch import nn
from .mae import AutoEncoder
from .basic_modules import CrossAttensionTransformerBlock

"""
Alternative Cross Attension Reducer Design

Removed the parameter for summary token to prevent any sort of bias from the specific query tokens themselves

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
    
    def forward(self,x:torch.Tensor):
        batches = x.shape[0]
        summary_tokens = torch.eye(n=1,m=self.embed_dim).unsqueeze(0).expand(batches,-1,-1).to(x.device)
        x:torch.Tensor = self.cross_reduce(summary_tokens,x)
        x = x.flatten(1)
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

