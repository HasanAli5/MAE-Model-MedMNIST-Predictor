from torch import nn
import torch
from .basic_modules import *
from timm.layers.pos_embed_sincos  import build_sincos2d_pos_embed

class CNN(nn.Module):
    # the cnn encoder that breaks the image down into patches
    def __init__(self,
                 input_shape:tuple=(3,64,64),
                 h_dim:int=1024,embed_dim:int=512,
                 window_size:tuple=(4,4)):
        super().__init__()
        # dimensions
        self.in_dim = input_shape[0]
        self.h_dim = h_dim
        self.embed_dim = embed_dim
        # layers
        self.conv = nn.Conv2d(self.in_dim,h_dim,kernel_size=window_size,stride=window_size)
        self.fc = nn.Linear(h_dim,embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self,x:torch.Tensor):
        # downsample using convolution
        #print(f"before conv : {x.shape}")
        x = self.conv(x)
        #print(f"after conv : {x.shape}")
        # merge x and y dim into singular dim
        x = x.flatten(2)
        #print(f"after flatten : {x.shape}")
        # swap feature and tokens dimensions.
        x = x.transpose(-1,-2)
        #print(f"after transpose : {x.shape}")
        # simplifies features using linear layer but keeps tokens amount due to transpose swap
        x = self.fc(x)
        #print(f"after fc : {x.shape}")
        x = self.norm(x)
        return x


class Encoder(nn.Module):

    def Tblocks(self,length:int, embed_dim:int, num_head:int):
        # multi layer maker
        assert length > 0,"Invalid Block Length"
        blocks = []
        for _ in range(length):
            blocks.append(TransformerBlock(embed_dim,num_head))
        blocks.append(nn.LayerNorm(embed_dim))
        return nn.Sequential(*blocks)

    def __init__(self,
                 input_shape:tuple=(3,64,64),
                 t_length:int=8,
                 embed_dim:int=512,
                 num_head:int=8):
        super().__init__()
        # dimension
        self.embed_dim = embed_dim
        # layers
        self.transformer_blocks = self.Tblocks(t_length,embed_dim,num_head)

    def forward(self,x):
        x = self.transformer_blocks(x)
        return x


class Decoder(nn.Module):

    def Tblocks(self,length:int,embed_dim:int, num_head:int):
        # multi layer maker
        assert length > 0,"Invalid Block Length"
        blocks = []
        for _ in range(length):
            blocks.append(TransformerBlock(embed_dim,num_head))
        blocks.append(nn.LayerNorm(embed_dim))
        return nn.Sequential(*blocks)

    def __init__(self,
                 input_shape:tuple=(3,64,64),
                 t_length:int=4,
                 embed_dim:int=128,num_head:int=4,
                 window_size:tuple=(4,4),):
        super().__init__()
        # dimension
        self.input_shape = input_shape
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.out_dim = input_shape[0]
        self.patch_pixels = self.out_dim * window_size[0] * window_size[1]
        # layers
        self.transformer_blocks = self.Tblocks(t_length,embed_dim,num_head)
        self.fc = nn.Linear(embed_dim,self.patch_pixels)
        self.tanh = nn.Tanh()

    def forward(self,x):
        # pass through decoder block
        x = self.transformer_blocks(x)
        # convert token into patches correct size
        x = self.fc(x)
        x = self.tanh(x)
        return x

class MaskedAutoEncoder(nn.Module):

    def __init__(self,
                 input_shape:tuple=(3,64,64),
                 patcher = None,
                 encoder = None,
                 decoder = None,
                 mask_rate = 0.75,
                 window_size:tuple=(4,4)):
        super().__init__()

        self.input_shape = input_shape
        self.window_size = window_size
        grid_size_h = input_shape[1] // window_size[0]
        grid_size_w = input_shape[2] // window_size[1]

        # layers
        self.patcher = patcher if patcher else CNN(input_shape,window_size=window_size)
        self.encoder = encoder if encoder else Encoder(input_shape)
        self.decoder = decoder if decoder else Decoder(input_shape,window_size=window_size)

        self.enc_to_dec = nn.Linear(self.encoder.embed_dim,self.decoder.embed_dim)

        # postional encoding
        self.register_buffer(
            'pos_embed',
            build_sincos2d_pos_embed(feat_shape=(grid_size_h, grid_size_w),dim=self.encoder.embed_dim).float().unsqueeze(0))
        self.register_buffer(
            'decoder_pos_embed',
            build_sincos2d_pos_embed(feat_shape=(grid_size_h, grid_size_w),dim=self.decoder.embed_dim).float().unsqueeze(0))

        # parameter
        self.mask_rate = mask_rate
        # masking
        self.mask_tokens = nn.Parameter(torch.zeros(1,1,self.decoder.embed_dim))
        nn.init.normal_(self.mask_tokens, std=.02)
        # distro
        nn.init.xavier_uniform_(self.enc_to_dec.weight)
        nn.init.zeros_(self.enc_to_dec.bias)

    def patchify(self, images):
        b = images.shape[0]
        p_h = self.window_size[0]
        p_w = self.window_size[1]
        c = self.input_shape[0]
        h = self.input_shape[1] // p_h
        w = self.input_shape[2] // p_w
        x = images.reshape(shape=(b, c, h, p_h, w, p_w))
        x = torch.permute(x,(0,2,4,1,3,5))
        raw_pixel_patches = x.reshape(shape=(b, h * w, c * p_h * p_w))
        return raw_pixel_patches

    def unpatchify(self, patches):
        b = patches.shape[0]
        p_h = self.window_size[0]
        p_w = self.window_size[1]
        c = self.input_shape[0]
        h = self.input_shape[1] // p_h
        w = self.input_shape[2] // p_w
        x = patches.reshape(b, h, w, c, p_h, p_w)
        x = torch.permute(x,(0,3,1,4,2,5))
        # collapse the dimensions
        images = x.reshape(b, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        return images
    
    def masking(self,x):
        # masking process where we want to seperate eeg tokens into two piles
        batch, seq_len, data_len  = x.shape
        len_keep = int(seq_len * (1 - self.mask_rate))
        
        noise = torch.rand(batch, seq_len, device=x.device)
        # sort by noise
        shuffled_ids = torch.argsort(noise, dim=1)
        # sort restore id by shuffle for undo
        restoration_ids = torch.argsort(shuffled_ids, dim=1)

        kept_ids = shuffled_ids[:, :len_keep]
        masked_x = torch.gather(x, dim=1, index=kept_ids.unsqueeze(-1).expand(-1, -1, data_len))
        
        loss_mask = (restoration_ids>=len_keep).float()
        return masked_x, restoration_ids, loss_mask
    
    def unmasking(self,x,restore_ids):
        batch, seq_len, data_len = x.shape
        # create the mask tokens for the decoder to decipher

        mask_tokens = self.mask_tokens.repeat(batch, restore_ids.shape[1] - seq_len, 1)
        # add back to the encoded tokens

        x = torch.cat([x, mask_tokens], dim=1)
        # resort the tokens

        x = torch.gather(x, dim=1, index=restore_ids.unsqueeze(-1).expand(-1, -1, data_len))
        return x

    def image_mask(self,token_loss_mask):
        # converts a token loss mask to real image mask
        batch  = token_loss_mask.shape[0]
        p_h = self.input_shape[1]//self.window_size[0]
        p_w = self.input_shape[2]//self.window_size[1]
        loss_mask = token_loss_mask.unsqueeze(-1).expand(-1,-1,self.window_size[0]*self.window_size[1])
        loss_mask = loss_mask.reshape(batch,p_h,p_w,1,self.window_size[0],self.window_size[1])
        loss_mask = torch.permute(loss_mask,(0,3,1,4,2,5))
        loss_mask = loss_mask.reshape(batch, 1, self.input_shape[1], self.input_shape[2])
        return loss_mask

    def forward(self,x:torch.Tensor):

        #real_images = x

        patches = self.patchify(x)
        #print(f"patches shape : {patches.shape}")
        x = self.patcher(x)
        #print(f"patcher shape : {x.shape}")
        x = x + self.pos_embed
        x, restore_id,loss_mask = self.masking(x)
        x = self.encoder(x)
        x = self.enc_to_dec(x)
        x = self.unmasking(x,restore_id)
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        #print(f"decoder shape : {x.shape}")

        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-6)

        normalised_patches = (patches - mean) / (std)

        #pred_image = self.unpatchify(x)
        #loss_mask_2d = self.image_mask(loss_mask)
        #smoothed_mask = nn.functional.avg_pool2d(loss_mask_2d, kernel_size=3, stride=1, padding=1)
        bool_mask = loss_mask.bool()
        loss = nn.functional.smooth_l1_loss(x[bool_mask],normalised_patches[bool_mask],reduction='mean',beta=0.05)
        #expanded_loss_mask = loss_mask.unsqueeze(-1).expand(-1,-1,element_wise_loss.size(-1)).to(x.dtype)
        #loss = (element_wise_loss * expanded_loss_mask).sum() / (expanded_loss_mask.sum() + 1e-8)

        with torch.no_grad():
            pred_image = self.unpatchify((x * std) + mean)

        return pred_image, self.image_mask(loss_mask), loss


class AutoEncoder(nn.Module):
    # used in predictor model
    def __init__(self,
                 input_shape:tuple=(3,64,64),
                 patcher = None,
                 encoder = None,
                 window_size:tuple=(4,4)):
        
        super().__init__()

        grid_size_h = input_shape[1] // window_size[0]
        grid_size_w = input_shape[2] // window_size[1]

        
        self.patcher = patcher if patcher else CNN(input_shape,window_size=window_size)
        self.encoder = encoder if encoder else Encoder(input_shape)
        self.register_buffer(
                    'pos_embed',
                    build_sincos2d_pos_embed(feat_shape=(grid_size_h, grid_size_w),dim=self.encoder.embed_dim).float().unsqueeze(0))
    
    def forward(self,x):
        x = self.patcher(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        return x