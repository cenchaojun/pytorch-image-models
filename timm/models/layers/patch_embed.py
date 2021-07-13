""" Image to Patch Embedding using Conv2d
使用2d卷积，对图像进行小块embedding
A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""

from torch import nn as nn

from .helpers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    默认输入图像大小为224.每个小块的大小为16，拉伸后的维度embed_dim为16*16*3=768，
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)  # 重回两个相同的元组,也就是图像的尺寸。
        patch_size = to_2tuple(patch_size)  # 每个图像小块的尺寸
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # 这个就是图像一行有多少个，网格的大小224/16=14
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 一张图像总共有多少个的小块，14*14=196个
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # 这个就是用了2维的卷积进行
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() # 如果传入norm_layer，就对其进行归一化，如果不传入，就不进行任何处理

    def forward(self, x):
        # 将图片数据传入进来
        B, C, H, W = x.shape  # 获取输入的tensor形状大小
        # 如果输入的图像的大小和设定的不一样，那么就会主动报错，输入图像的大小与模型不匹配。
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # 之后进行卷积操作
        # 如果有拉平操作，使用，输入的时候设置为True,那么就要拉伸，
        # 原先的tensor的形状是 B, C, H, W，通过对第2个维度的拉伸，也就是前两个不变，就变成了 B C HW
        # 之后将第1个和第二个维度进行交换一下，变为B HW C
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # 如果没有norm层，就不进行变化
        x = self.norm(x)
        # 最终我们得到B HW C这个tensor
        return x
