from torch import nn
import torch
from .mae import AutoEncoder
from .basic_modules import *


class PredictorTransformerBlock(nn.Module):

    # specific transformer block implementation
    # that passes on the previous q as well

    def __init__(self,embed_dim,num_head):
        super().__init__()
        self.embed_dim = embed_dim
        self.block = CrossAttensionTransformerBlock(embed_dim=embed_dim,num_heads=num_head)
        self.self_attending_block = CrossAttensionTransformerBlock(embed_dim=embed_dim,num_heads=num_head)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,q:torch.Tensor,prev_q:torch.Tensor,kv:torch.Tensor):
        # extraction of micro details to new_q
        new_q = self.block(q,kv)
        # we then attend macro details to the micro details to create
        # hierachial relationship mapping between them if there is one.

        attended_q = self.self_attending_block(prev_q,new_q)
        # we then relay the micro detail as the next q
        # and the macro-micro relationship as the previous q
        # effictively honing q tokens
        # and undestanding how the q tokens fit with the prev_q
        return new_q,attended_q,kv

class DecoderPredictor(nn.Module):

    def Tblocks(self,length:int,embed_dim:int, num_head:int):
        # multi layer maker
        assert length > 0,"Invalid Block Length"
        blocks = []
        for i in range(length):
            blocks.append(PredictorTransformerBlock(embed_dim,num_head))
        return nn.ModuleList(blocks)

    def __init__(self,
                 n_labels:int=9,
                 t_length:int=4,
                 pred_tokens:int=128,
                 embed_dim:int=512,num_head:int=8):
        super().__init__()
        # dimension
        self.embed_dim = embed_dim
        self.pred_tokens = pred_tokens
        # layers
        self.transformer_blocks = self.Tblocks(t_length,embed_dim,num_head)
        self.cross_attend = CrossAttensionTransformerBlock(embed_dim=embed_dim,num_heads=num_head)
        self.cross_reduce = CrossAttensionTransformerBlock(embed_dim=embed_dim,num_heads=num_head)
        self.fc = nn.Linear(embed_dim,n_labels)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)
        self.predictor_tokens = nn.Parameter(torch.zeros(1,1,self.embed_dim))
        nn.init.normal_(self.predictor_tokens, std=.02)
        self.summarise_token = nn.Parameter(torch.empty(1,1,self.embed_dim))
        nn.init.normal_(self.summarise_token, std=.02)

    def forward(self,kv):
        batches, tokens, embed_dim = kv.shape
        # expand to the kv and encode with positional embedding
        q = self.predictor_tokens.expand(batches,self.pred_tokens,-1)
        summarise_tokens = self.summarise_token.expand(batches,-1,-1)
        # pass through the transformer blocks
        loop_q =  q
        loop_prev_q = kv
        for block in self.transformer_blocks:
            loop_q,loop_prev_q,_ = block(loop_q,loop_prev_q,kv)
        x = torch.concat([loop_prev_q,loop_q],dim=1)
        # average pool and pass through linear layer for predictions
        x = self.norm(self.cross_reduce(summarise_tokens,x))
        x = x.squeeze(1)
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
        self.predictor = predictor if predictor else DecoderPredictor(n_labels=n_labels)
    
    def forward(self,x):
        x = self.autoencoder(x)
        x = self.predictor(x)
        return x

