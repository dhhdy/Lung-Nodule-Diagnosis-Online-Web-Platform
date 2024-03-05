import torch
import torch.nn as nn
from functools import partial
from lib.eva_vit_model import EVAVisionTransformer, _cfg_B16
from timm.models.registry import register_model
from apex.normalization import FusedLayerNorm
from torchsummary import summary
class EVA(EVAVisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        # self.pos_embed = nn.Parameter(torch.zeros(1, 196 + 1, self.embed_dim))
        if not self.is_pe:
            # B = x.shape[0]
            x = self.patch_embed(x)
        # print(x.shape)
        pe = self.pos_embed

        x = x + pe
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)

        x = self.norm(x)
        return x


# modify img,pt_hw 就 ok
@register_model
def eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE(dim=768, size=224, pretrained=False, is_pe=False, **kwargs):
    ## patch_size=16, pt_hw_seq_len=14

    model = EVA(
        img_size=size,
        patch_size=16,
        embed_dim=dim,
        depth=12,
        num_heads=12,
        mlp_ratio=4*2/3,
        qkv_bias=True,
        norm_layer=partial(FusedLayerNorm, eps=1e-6),
        subln=True,
        xattn=True,
        naiveswiglu=True,
        rope=True,
        pt_hw_seq_len=size//16,   # img_size/patch_size
        intp_freq=True,
        is_pe=is_pe,
        **kwargs)
    model.default_cfg = _cfg_B16()
    pe = model.pos_embed[:, 1:, :].detach()  # [196, 768]
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    if pretrained:
        checkpoint = torch.load('/root/autodl-tmp/pretrained/EVA02_CLIP_B_psz16_s8B.pt')
        # print(type(checkpoint))
        dic = {}
        for k, v in checkpoint.items():
            tmp = k.split('.')
            if tmp[0] == 'visual' and tmp[1] != 'head' and k != 'pos_embed' and k!='patch_embed.proj.weight':
                # str = ''1
                # for p in range(1, len(tmp)):
                #     str += tmp[p]
                #     if (p + 1) != len(tmp):
                #         str += '.'
                # print(str)
                dic[k] = v
        model.load_state_dict(dic, strict=False)
    return model




if __name__ == '__main__':
    model = eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE()
    summary(model.cuda(),input_size=(1,224,224))
    # # print(model)
    # a = torch.ones(1, 1, 224, 224).cuda()
    # a = model(a).cuda()
    # print('qaq: ',a.shape)


    # if pretrained:
    #     checkpoint = torch.load('pretrained/EVA02_CLIP_B_psz16_s8B.pt')
    #     # print(type(checkpoint))
    #     dic = {}
    #     for k, v in checkpoint.items():
    #         tmp = k.split('.')
    #         if tmp[0] == 'visual' and tmp[1] != 'head':
    #             str = ''
    #             for p in range(1, len(tmp)):
    #                 str += tmp[p]
    #                 if (p + 1) != len(tmp):
    #                     str += '.'
    #             # print(str)
    #             dic[str] = v
    #     model.load_state_dict(dic, strict=False)
































