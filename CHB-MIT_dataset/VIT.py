import torch, math
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(self, pos_emb_scaling, *, image_size, patch_size, num_classes, dim, transformer, channels = 1):
        super().__init__()

        self.dim = dim
        self.pos_emb_scaling = pos_emb_scaling

        image_size_h, image_size_w = pair(image_size)
        patch_size_h, patch_size_w = pair(patch_size)
        assert image_size_h % patch_size_h == 0 and image_size_w % patch_size_w == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size_h // patch_size_h) * (image_size_w // patch_size_w)
        patch_dim = channels * patch_size_h * patch_size_w

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_h , p2 = patch_size_w),
            nn.Linear(patch_dim, dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.transformer = transformer
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, register_hook = False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, :(n + 1)]
        x = x + self.pos_embedding[:, :(n + 1)] * ( 1 if not self.pos_emb_scaling else (1/math.sqrt(self.dim)) )

        x = self.transformer(x, register_hook)
        x = x[:, 0] # CLS Pooling

        return self.mlp_head(x)