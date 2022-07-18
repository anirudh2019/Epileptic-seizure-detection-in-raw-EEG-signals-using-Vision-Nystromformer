import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange
from math import ceil

# helper functions
def exists(val):
    return val is not None
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

###### NYSTROM ATTENTION
def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

class NystromAttention(nn.Module):
    def __init__(self, *, dim, dim_head, heads, dropout, num_landmarks, residual, residual_conv_kernel):
        super().__init__()
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = 6

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.attn_gradients = None
        self.attention_map = None

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, register_hook= False):
        b, n, _, h, m, iters = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        divisor = l

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        attn_approx = attn1 @ attn2_inv @ attn3

        out = attn_approx @ v

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if register_hook:
            self.save_attention_map(attn_approx)
            attn_approx.register_hook(self.save_attn_gradients)

        return out

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map
############################################################
###### STANDARD ATTENTION
class Attention(nn.Module):
    def __init__(self, *, dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.attn_gradients = None
        self.attention_map = None

    def forward(self, x, register_hook=False):
        b, n, _, h = *x.shape, self.num_heads

        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        self.save_attention_map(attn)
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)           

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.proj(out)
        out = self.proj_drop(out)
        return out

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map
############################################################
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, register_hook = False, **kwargs):
        x = self.norm(x)
        return self.fn(x, register_hook, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 2, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mult), dim)
        )

    def forward(self, x, register_hook= False):
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, dim_head, heads, attn_dropout, ff_dropout, ff_mult, attention, num_landmarks, attn_values_residual, attn_values_residual_conv_kernel):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if attention == "nystrom":
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                    PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
                ]))
            elif attention == "standard":
                print("standard attention")
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim = dim, num_heads = heads, dropout = attn_dropout)),
                    PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
                ]))

    def forward(self, x, register_hook= False):
        for attn, ff in self.layers:
            x = attn(x, register_hook) + x
            x = ff(x, register_hook ) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, num_heads, attn_dropout, ff_dropout, ff_mult, attention, num_landmarks, attn_values_residual, attn_values_residual_conv_kernel):
        super().__init__()

        self.dim = dim
        dim_head = dim//num_heads

        image_size_h, image_size_w = pair(image_size)
        patch_size_h, patch_size_w = pair(patch_size)
        assert image_size_h % patch_size_h == 0 and image_size_w % patch_size_w == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size_h // patch_size_h) * (image_size_w // patch_size_w)
        patch_dim = patch_size_h * patch_size_w

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = patch_size_h , p2 = patch_size_w),
            nn.Linear(patch_dim, dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.transformer = TransformerEncoder(dim, depth, dim_head, num_heads, attn_dropout, ff_dropout, ff_mult, attention, num_landmarks, attn_values_residual, attn_values_residual_conv_kernel)
        self.mlp_head = nn.Linear(dim, 2)

    def forward(self, img, register_hook = False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x, register_hook)
        x = x[:, 0] # CLS Pooling

        return self.mlp_head(x)