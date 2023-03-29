import torch
import torch.nn as nn

from timm.models.layers import DropPath,trunc_normal_
from utils.logger import *

from .EdgeConv import DGCNN_Grouper

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GeoCrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=1, qkv_bias=False, qk_scale=1, attn_drop=0., proj_drop=0., aggregate_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Identity() # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Identity() # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.x_map = nn.Identity() # nn.Linear(aggregate_dim, 1)
        
    def forward(self, q, k, v):
        B, N, _ = q.shape
        C = self.out_dim
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, NK, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, 3)

        x = self.x_map(x)

        return x

class SubFold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()
        self.in_channel = in_channel
        self.step = step
        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x, c):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = c.to(x.device) # b 3 n2
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_pred=16, num_point=128):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.norm_q = nn.Identity() # norm_layer(dim)
        self.norm_k = nn.Identity () # norm_layer(dim)
        self.attn = GeoCrossAttention(dim, dim, num_heads=1, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, aggregate_dim=16)
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.fold_step = int(pow(num_pred, 0.5) + 0.5)
        self.generate_anchor = SubFold(dim, step = self.fold_step, hidden_dim = dim // 2)
        
        self.num_pred = num_pred
        self.num_point = num_point
        self.generate_feature = nn.Sequential(
            nn.Conv1d(self.num_point, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 16, 1)
        )
        
    def forward(self, x, coor):
        norm_x = self.norm1(x)
        x_1 = self.self_attn(norm_x)
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x))) # B N dim
        
        global_x = torch.max(x, dim=1, keepdim=False)[0] # B dim
        diff_x = global_x.unsqueeze(1).repeat(1,self.num_point,1) - x
        x_2 = self.generate_feature(diff_x)
        
        norm_k = self.norm_k(x) # B N dim
        norm_q = self.norm_q(x_2) # B L dim
        coor_2 = self.attn(q=norm_q, k=norm_k, v=coor) 
        coor_2 = self.generate_anchor(global_x, coor_2.transpose(1,2)).transpose(1,2)
        
        x = torch.cat([x, x_2], dim=1)        
        coor = torch.cat([coor,coor_2], dim=1) # coor: B N 3 -> B N+L 3
        return x, coor

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, q, v):
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        q = q + self.drop_path(q_2)
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q

# --- Transfomer Architecture --- #
class AnchorTransformer(nn.Module):
    """ Edge Convolution +  Transformer Encoder +  Point Scattering + Transformer Decoder
    """
    def __init__(self, in_chans=3, embed_dim=768, depth=[6, 6], num_query = 224, num_heads=6, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_query = num_query
        self.depth = depth

         # feature extraction
        self.grouper = DGCNN_Grouper()  # B 3 N to B 3 N and B C N
        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        # encoder
        self.encoder = nn.ModuleList([
            EncoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, num_pred=16, num_point=128+i*16)
            for i in range(depth[0])])

        # increase dim and make coarse query
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * (num_query-128))
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1)
        )

        # decoder 
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[1])])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, inpc):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        # build point feature
        bs = inpc.size(0)
        coor, f = self.grouper(inpc.transpose(1,2).contiguous())
        pos =  self.pos_embed(coor).transpose(1,2)
        x = self.input_proj(f).transpose(1,2)
        coor = coor.transpose(1,2)
        
        # encoder
        x = x + pos
        for i, blk in enumerate(self.encoder): 
            # x, coor = blk(x+pos, coor)
            x, coor = blk(x, coor)

        # point scattering
        global_feature = self.increase_dim(x.transpose(1,2)) # B 1024 N 
        global_feature = torch.max(global_feature, dim=-1)[0] # B 1024
        coarse_relative = torch.cat([torch.zeros([bs, 128, 3]).to(x.device), self.coarse_pred(global_feature).reshape(bs, -1, 3)], dim=1)  # B M C(3)
        coarse_point = coarse_relative + coor
        
        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, coarse_point.size(1), -1), 
            coarse_point], dim=-1) # B M C+3 
        q = self.mlp_query(query_feature.transpose(1,2)).transpose(1,2) # B M C 
        
        # decoder
        for i, blk in enumerate(self.decoder): 
            q = blk(q, x)
            
        return q, coarse_point
