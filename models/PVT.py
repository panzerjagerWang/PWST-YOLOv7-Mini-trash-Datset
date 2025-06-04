import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from models import wavemlp


class PVTBlock(nn.Module):
    def __init__(self, c1, c2, patch_size, mlp_ratio, sr_ratios, depths, n=1):
        super().__init__()
        self.conv = None
        num_heads = c2//64
        self.blocks = nn.Sequential(*[PVTLayer(in_chans=c1, embed_dims=c2, patch_size=patch_size, num_heads=num_heads, 
                                   mlp_ratio=mlp_ratio, sr_ratios=sr_ratios, depths=depths) for i in range(n)])

    def forward(self, x):
        # print("输入:"+str(x.shape))
        x = self.blocks(x)
        # print("输出:"+str(x.shape))
        return x

class PVTLayer(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dims=64,
                 num_heads=1, mlp_ratio=8, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=3, sr_ratios=8):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # dpr = [x.item() for x in range(torch.linspace(0, drop_path_rate, sum(depths)))]  # stochastic depth decay rule

        self.encoder = nn.ModuleList([EncoderBlock(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])
        
    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        # print("patch embed后:"+str(x.shape))
        num_patches = H * W
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims)).to(x.device)
        trunc_normal_(pos_embed, std=.02)
        if pos_embed.dtype != x.dtype:
            pos_embed = pos_embed.half()
            # print("x:")
            # print(x.dtype)
            # print("pos_embed:")
            # print(pos_embed.dtype)
        x = self.pos_drop(x + pos_embed)
        for enc in self.encoder:
            x = enc(x, H, W)
        # print("block后:"+str(x.shape))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, H, W

class EncoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
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

#-----------------------------------------------------------------
class PVTBlockv2(nn.Module):
    def __init__(self, c1, c2, patch_size, stride, mlp_ratio, sr_ratios, depths, n=1):
        super().__init__()
        num_heads = c2//64
        self.blocks = nn.Sequential(*[PVTLayerv2(in_chans=c1, embed_dims=c2, patch_size=patch_size, stride=stride, num_heads=num_heads, 
                                   mlp_ratio=mlp_ratio, sr_ratios=sr_ratios, depths=depths, linear=True) for i in range(n)])

    def forward(self, x):
        # print("输入:"+str(x.shape))
        x = self.blocks(x)
        # print("输出:"+str(x.shape))
        return x

class PVTLayerv2(nn.Module):
    def __init__(self, patch_size=4, stride=4, in_chans=3, embed_dims=64,
                 num_heads=1, mlp_ratio=8, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=3, sr_ratios=8, linear=False):
        super().__init__()
        self.depths = depths
        self.patch_embed = OverlapPatchEmbed(patch_size=patch_size, stride=stride, in_chans=in_chans, embed_dim=embed_dims)

        self.encoder = nn.ModuleList([EncoderBlockv2(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios, linear=linear)
            for j in range(depths)])
        self.norm = norm_layer(embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        for enc in self.encoder:
            x = enc(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class EncoderBlockv2(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attentionv2(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlpv2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class Attentionv2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Mlpv2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

#-----------------------------------------------------------------
class CPVTBlock(nn.Module):
    def __init__(self, c1, c2, patch_size, stride, mlp_ratio, sr_ratios, depths, n=1):
        super().__init__()
        num_heads = c2//64
        self.blocks = nn.Sequential(*[CPVTLayer(in_chans=c1, embed_dims=c2, patch_size=patch_size, stride=stride, num_heads=num_heads, 
                                   mlp_ratio=mlp_ratio, sr_ratios=sr_ratios, depths=depths, linear=True) for i in range(n)])

    def forward(self, x):
        x = self.blocks(x)
        return x

class CPVTLayer(nn.Module):
    def __init__(self, patch_size=4, stride=4, in_chans=3, embed_dims=64,
                 num_heads=1, mlp_ratio=8, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=3, sr_ratios=8, linear=False):
        super().__init__()
        self.depths = depths
        self.patch_embed = OverlapPatchEmbed(patch_size=patch_size, stride=stride, in_chans=in_chans, embed_dim=embed_dims)

        self.encoder = nn.ModuleList([EncoderBlockv2(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios, linear=linear)
            for j in range(depths)])
        self.pos_block = PosCNN(in_chans=embed_dims, embed_dim=embed_dims)
        self.norm = norm_layer(embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        for i, enc in enumerate(self.encoder):
            x = enc(x, H, W)
            if i == 0:
                x = self.pos_block(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),)
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

#-----------------------------------------------------------------
class WavePVT(nn.Module):
    def __init__(self, c1, c2, patch_size, stride, mlp_ratio, sr_ratios, depths, n=1):
        super().__init__()
        num_heads = c2//64
        self.blocks = nn.Sequential(*[WavePVTLayer(in_chans=c1, embed_dims=c2, patch_size=patch_size, stride=stride, num_heads=num_heads, 
                                   mlp_ratio=mlp_ratio, sr_ratios=sr_ratios, depths=depths, linear=True) for i in range(n)])

    def forward(self, x):
        x = self.blocks(x)
        return x

class WavePVTLayer(nn.Module):
    def __init__(self, patch_size=4, stride=4, in_chans=3, embed_dims=64,
                 num_heads=1, mlp_ratio=8, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=3, sr_ratios=8, linear=False):
        super().__init__()
        self.depths = depths
        self.patch_embed = OverlapPatchEmbed(patch_size=patch_size, stride=stride, in_chans=in_chans, embed_dim=embed_dims)

        self.encoder = nn.ModuleList([WaveEncoderBlock(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios, linear=linear)
            for j in range(depths)])
        self.norm = norm_layer(embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        for enc in self.encoder:
            x = enc(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

class WaveEncoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attentionv2(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = WaveMLP(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, drop_path=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class WaveMLP(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = wavemlp.PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop,mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = wavemlp.Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        # self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        
    def forward(self, x, H, W):
        B = x.shape[0]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = self.dwconv(x) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        x = x.flatten(2).transpose(1, 2)
        return x

#-----------------------------------------------------------------
class WavePVTv2(nn.Module):
    def __init__(self, c1, c2, patch_size, stride, mlp_ratio, sr_ratios, depths, n=1):
        super().__init__()
        num_heads = c2//64
        self.blocks = nn.Sequential(*[WavePVTLayerv2(in_chans=c1, embed_dims=c2, patch_size=patch_size, stride=stride, num_heads=num_heads, 
                                   mlp_ratio=mlp_ratio, sr_ratios=sr_ratios, depths=depths, linear=True) for i in range(n)])

    def forward(self, x):
        # print("输入:"+str(x.shape))
        x = self.blocks(x)
        # print("输出:"+str(x.shape))
        return x

class WavePVTLayerv2(nn.Module):
    def __init__(self, patch_size=4, stride=4, in_chans=3, embed_dims=64,
                 num_heads=1, mlp_ratio=8, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=3, sr_ratios=8, linear=False):
        super().__init__()
        self.depths = depths
        self.patm = wavemlp.PATM(in_chans)
        self.patch_embed = OverlapPatchEmbed(patch_size=patch_size, stride=stride, in_chans=in_chans, embed_dim=embed_dims)

        self.encoder = nn.ModuleList([EncoderBlockv2(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios, linear=linear)
            for j in range(depths)])
        self.norm = norm_layer(embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(self.patm(x))
        for enc in self.encoder:
            x = enc(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

#-----------------------------------------------------------------
class WavePVTv3(nn.Module):
    def __init__(self, c1, c2, patch_size, stride, mlp_ratio, sr_ratios, depths, n=1):
        super().__init__()
        num_heads = c2//64
        self.blocks = nn.Sequential(*[WavePVTLayerv3(in_chans=c1, embed_dims=c2, patch_size=patch_size, stride=stride, num_heads=num_heads, 
                                   mlp_ratio=mlp_ratio, sr_ratios=sr_ratios, depths=depths, linear=True) for i in range(n)])

    def forward(self, x):
        # print("输入:"+str(x.shape))
        x = self.blocks(x)
        # print("输出:"+str(x.shape))
        return x

class WavePVTLayerv3(nn.Module):
    def __init__(self, patch_size=4, stride=4, in_chans=3, embed_dims=64,
                 num_heads=1, mlp_ratio=8, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=3, sr_ratios=8, linear=False):
        super().__init__()
        self.depths = depths
        self.patch_embed = wavemlp.PatchEmbedOverlapping(patch_size=patch_size, stride=stride, padding=patch_size // 2, in_chans=in_chans, embed_dim=embed_dims)
        self.patm = wavemlp.PATM(embed_dims)

        self.encoder = nn.ModuleList([EncoderBlockv2(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios, linear=linear)
            for j in range(depths)])
        self.norm = norm_layer(embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self.patm(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for enc in self.encoder:
            x = enc(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

#-----------------------------------------------------------------
class WavePVTv4(nn.Module):
    def __init__(self, c1, c2, patch_size, stride, mlp_ratio, sr_ratios, depths, n=1):
        super().__init__()
        num_heads = c2//64
        self.blocks = nn.Sequential(*[WavePVTLayerv4(in_chans=c1, embed_dims=c2, patch_size=patch_size, stride=stride, num_heads=num_heads, 
                                   mlp_ratio=mlp_ratio, sr_ratios=sr_ratios, depths=depths, linear=True) for i in range(n)])

    def forward(self, x):
        # print("输入:"+str(x.shape))
        x = self.blocks(x)
        # print("输出:"+str(x.shape))
        return x

class WavePVTLayerv4(nn.Module):
    def __init__(self, patch_size=4, stride=4, in_chans=3, embed_dims=64,
                 num_heads=1, mlp_ratio=8, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=3, sr_ratios=8, linear=False):
        super().__init__()
        self.depths = depths
        self.patch_embed = wavemlp.PatchEmbedOverlapping(patch_size=patch_size, stride=stride, padding=patch_size // 2, in_chans=in_chans, embed_dim=embed_dims)
        self.patm = wavemlp.PATM(embed_dims)

        self.encoder = nn.ModuleList([WaveEncoderBlock(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            norm_layer=norm_layer, sr_ratio=sr_ratios, linear=linear)
            for j in range(depths)])
        self.norm = norm_layer(embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self.patm(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for enc in self.encoder:
            x = enc(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

