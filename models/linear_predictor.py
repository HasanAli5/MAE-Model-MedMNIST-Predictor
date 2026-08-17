from torch import nn
from .mae import AutoEncoder

class DecoderPredictiorLinear(nn.Module):

    def __init__(self,
                 embed_dim:int=512,n_labels=9):
        super().__init__()
        # dimension
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(0.1)
        # back to initial feature size
        self.fc = nn.Linear(embed_dim, n_labels)
    
    def forward(self,x):
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
        self.predictor = predictor if predictor else DecoderPredictiorLinear(n_labels=n_labels)
    
    def forward(self,x):
        x = self.autoencoder(x)
        x = self.predictor(x)
        return x

