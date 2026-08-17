
from torch import nn
import math
import torch
from .basic_modules import *
from timm.layers.pos_embed_sincos  import build_sincos2d_pos_embed, build_rotary_pos_embed, apply_rot_embed



class DecoderPredictorStandard(nn.Module):

    def Tblocks(self,length:int,embed_dim:int, num_head:int):
        # multi layer maker
        assert length > 0,"Invalid Block Length"
        blocks = []
        for i in range(length):
            blocks.append(CrossAttensionTransformerBlock(embed_dim,num_head))
        return nn.ModuleList(blocks)

    def __init__(self,
                 n_labels:int=9,
                 t_length:int=8,
                 pred_tokens:int=128,
                 embed_dim:int=512,num_head:int=8):
        super().__init__()
        # dimension
        self.embed_dim = embed_dim
        self.pred_tokens = pred_tokens
        # layers
        self.transformer_blocks = self.Tblocks(t_length,embed_dim,num_head)
        self.cross_reduce = CrossAttensionTransformerBlock(embed_dim=embed_dim,num_heads=num_head)
        self.fc = nn.Linear(embed_dim,n_labels)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(embed_dim)
        self.predictor_tokens = nn.Parameter(torch.zeros(1,1,self.embed_dim))
        nn.init.normal_(self.predictor_tokens, std=.02)
        self.prev_embed = nn.Parameter(torch.empty(1,1,self.embed_dim))
        nn.init.normal_(self.prev_embed, std=.02)
        self.summarise_token = nn.Parameter(torch.empty(1,1,self.embed_dim))
        nn.init.normal_(self.summarise_token, std=.02)

    def forward(self,kv):
        batches, tokens, embed_dim = kv.shape
        # expand to the kv and encode with positional embedding
        q = self.predictor_tokens.expand(batches,self.pred_tokens,-1)
        summarise_tokens = self.summarise_token.expand(batches,-1,-1)
        # pass through the transformer blocks
        loop_q =  q
        loop_prev_q = q + self.prev_embed
        for block in self.transformer_blocks:
            loop_q,loop_prev_q,_ = block(loop_q,loop_prev_q,kv)
        x = self.norm(loop_prev_q)
        # average pool and pass through linear layer for predictions
        x = self.cross_reduce(summarise_tokens,x)
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

