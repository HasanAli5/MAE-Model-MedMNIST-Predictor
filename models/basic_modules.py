from torch import nn
import torch
from timm.layers.pos_embed_sincos  import build_sincos2d_pos_embed, build_rotary_pos_embed, apply_rot_embed


class MultiHeadAttention(nn.Module):

    def __init__(self,embed_dim:int,num_heads:int):
        super().__init__()
        assert embed_dim % num_heads == 0,f"Embedding dimension must be a multiple of {num_heads}"
        # parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # layers
        self.linear_qkv = nn.Linear(embed_dim, 3* embed_dim)
        self.linear_out = nn.Linear(embed_dim,embed_dim)
        
        # use xavier uniform for MHA weights
        nn.init.xavier_uniform_(self.linear_qkv.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        # zero all biases
        nn.init.zeros_(self.linear_qkv.bias)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self,x:torch.Tensor):
        batch, seq, embed_dim = x.shape
        qkv:torch.Tensor = self.linear_qkv(x)
        # splitting embed dim between heads
        qkv = qkv.reshape(batch,seq,self.num_heads,3*self.head_dim)
        #switch num_heads with seq for batch matmul
        qkv = qkv.transpose(1,2)
        #cuts the head dim into three parts
        q,k,v = qkv.chunk(3,-1)
        values = nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=False)
        values = values.transpose(1,2)
        values = values.reshape(batch,seq,embed_dim)
        output = self.linear_out(values)
        return output

class CrossAttention(nn.Module):

    def __init__(self,embed_dim:int,num_heads:int):
        super().__init__()
        assert embed_dim % num_heads == 0,f"Embedding dimension must be a multiple of {num_heads}"
        # parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # layers
        self.linear_kv = nn.Linear(embed_dim, 2* embed_dim)
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_out = nn.Linear(embed_dim,embed_dim)
        
        # use xavier uniform for MHA weights
        nn.init.xavier_uniform_(self.linear_kv.weight)
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        # zero all biases
        nn.init.zeros_(self.linear_kv.bias)
        nn.init.zeros_(self.linear_q.bias)
        nn.init.zeros_(self.linear_out.bias)

        self.sin_embed, self.cos_embed = build_rotary_pos_embed(feat_shape=(512,),dim=self.head_dim*2)
        self.register_buffer('sin_pos',self.sin_embed)
        self.register_buffer('cos_pos',self.cos_embed)

    def forward(self,q:torch.Tensor,kv:torch.Tensor):
        kv_batch, kv_seq, kv_embed_dim = kv.shape
        q_batch, q_seq, q_embed_dim = q.shape
        kv = self.linear_kv(kv)
        q = self.linear_q(q)
        # splitting embed dim between heads
        kv = kv.reshape(kv_batch,kv_seq,self.num_heads,2*self.head_dim)
        #switch num_heads with seq for batch matmul
        kv = kv.transpose(1,2)
        q = q.reshape(q_batch, q_seq, self.num_heads, self.head_dim)
        q = q.transpose(1,2)
        #cuts the head dim into two parts
        k,v = kv.chunk(2,-1)

        q_sin = self.sin_pos[:q_seq, :]
        q_cos = self.cos_pos[:q_seq, :]
        
        k_sin = self.sin_pos[:kv_seq, :]
        k_cos = self.cos_pos[:kv_seq, :]

        q = apply_rot_embed(q, q_sin, q_cos)
        k = apply_rot_embed(k, k_sin, k_cos)

        values = nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=False)
        values = values.transpose(1,2)
        values = values.reshape(q_batch,q_seq,q_embed_dim)
        output = self.linear_out(values)
        return output

class FeedForward(nn.Module):

    def __init__(self,embed_dim:int,hid_dim):
        super().__init__()
        # layers
        self.fc1 = nn.Linear(embed_dim,hid_dim)
        self.fc2 = nn.Linear(hid_dim,embed_dim)
        # activation
        self.gelu = nn.GELU()
        # distro
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc1.bias)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self,
                 embed_dim:int,
                 num_heads:int):
        super().__init__()
        # layers
        self.mha = MultiHeadAttention(embed_dim,num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim,4*embed_dim)
        
    def forward(self,x):
        x = self.mha(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x

class CrossAttensionTransformerBlock(nn.Module):

    def __init__(self,
                 embed_dim:int,
                 num_heads:int):
        super().__init__()
        # layers
        self.ca = CrossAttention(embed_dim,num_heads)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim,4*embed_dim)
        
    def forward(self,q,kv):
        x = self.ca(self.q_norm(q),self.kv_norm(kv)) + q
        x = self.ff(self.norm2(x)) + x
        return x