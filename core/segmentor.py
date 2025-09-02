# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 11:57:29 2025

@author: zhouy
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d, to_2tuple, trunc_normal_
from .module import ConvLayer, MFFBlock, MLLAFormer
from mmcv.cnn import ConvModule

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """
    
    def __init__(self, 
                 dim, 
                 ratio=4.0):
        super(PatchMerging, self).__init__()
        
        self.dim = dim
        
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), 
                      kernel_size=3, stride=2, padding=1, 
                      groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )
        
        return
    
    def forward(self, x):
        x = self.conv(x)
        return x

class BasicLayer(nn.Module):
    """ A basic MFFB layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """
    
    def __init__(self, 
                 dim, 
                 input_resolution, 
                 depth, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop=0.,
                 drop_path=0., 
                 norm_layer=LayerNorm2d, 
                 downsample=None,
                 **kwargs):

        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            MFFBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                     norm_layer=norm_layer, **kwargs)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            down = self.downsample(x)
        else:
            down = x
        return x, down
    
class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, 
                 img_size=224, 
                 patch_size=4, 
                 in_chans=3, 
                 embed_dim=96):
        super().__init__()
        
        img_size = to_2tuple(img_size) if isinstance(img_size, int) else img_size
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, 
                               stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, 
                      stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, 
                      stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, 
                      stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, 
                      bias=False, act_func=None)
        )
        return
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        
        return x


class MFFEncoder(nn.Module):
    r""" MLLA
        A PyTorch impl of : `Demystify Mamba in Vision: A Linear Attention Perspective`
        
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each MLLA layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
    """
    
    def __init__(self, 
                 img_size=512,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=LayerNorm2d,
                 ape=False,
                 **kwargs):
        super(MFFEncoder, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = Stem(img_size=img_size, 
                                patch_size=patch_size, 
                                in_chans=in_chans, 
                                embed_dim=embed_dim)
        
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 
                embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout2d(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               **kwargs
                               )
            self.layers.append(layer)
            
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        outputs = list()
        for layer in self.layers:
            y, x = layer(x)
            outputs.append(y)
            
        return outputs

###############################################################################
#**************** DECODER ***************************************************##
class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768, identity=False):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(x)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
                    
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class MFFDecoder(nn.Module):
    def __init__(self, 
                 image_size=512,
                 num_classes=3,
                 input_dim=96,
                 feat_proj_dim=320,
                 embed_dim=180,
                 qkv_bias=True, 
                 dropout_ratio=0.1,
                 norm_layer=LayerNorm2d, 
                 align_corners=False,
                 **kwargs):
        super(MFFDecoder, self).__init__()
        
        self.num_classes = num_classes
        image_size = to_2tuple(image_size) if isinstance(image_size, int) else tuple(image_size)
        self.image_size = image_size
        image_size = [(image_size[0]//2**(i+2), image_size[1]//2**(i+2)) for i in range(4)]
        
        self.in_channels = [int(input_dim * 2 ** i_layer) for i_layer in range(4)]
        self.embed_dim = embed_dim
        self.feat_proj_dim = feat_proj_dim
        self.align_corners = align_corners
        
        # F-Fusion
        # try using all features at once
        self.linear_c4 = MLP(self.in_channels[-1], self.feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], self.feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], self.feat_proj_dim)
        
        self.linear_fuse = ConvModule(
                        in_channels=self.feat_proj_dim*3,
                        out_channels=self.embed_dim,
                        kernel_size=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        # F-Fusion ending
        
        # AVGPooling
        self.short_path = ConvModule(
                            in_channels=self.embed_dim,
                            out_channels=self.embed_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True)
                            )
        
        self.image_pool = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1), 
                                ConvModule(self.embed_dim, 
                                           self.embed_dim, 
                                           1, 
                                           conv_cfg=None, 
                                           norm_cfg=None, 
                                           act_cfg=dict(type='ReLU')))
        # AVGPooling ending
        
        # MMSCopE
        self.conv_downsample_2 = ConvModule(
                        self.embed_dim, self.embed_dim*2, kernel_size=3, stride=2, padding=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        self.conv_downsample_4 = ConvModule(
                        self.embed_dim, self.embed_dim*4, kernel_size=5, stride=4, padding=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        self.reduce_channels = nn.ModuleList([ConvModule(in_channels=self.embed_dim*4*(2**i),
                                out_channels=self.embed_dim,kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True)) for i in range(3)])
        
        vssm_dim = self.embed_dim*3
        self.vssm =MLLAFormer(vssm_dim, image_size[-1], 2, qkv_bias=qkv_bias, 
                              mlp_ratio=1, drop=0.1, norm_layer=norm_layer)
        
        self.proj_out = ConvModule(in_channels=vssm_dim,
                                out_channels=self.feat_proj_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True))
        # MMSCopE ending
        
        # MLP of Decoder
        feat_concat_dim = self.embed_dim*(2 + 3) + self.feat_proj_dim*3
        self.cat = ConvModule(in_channels=feat_concat_dim,
                                out_channels=self.embed_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True)) 
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=1)
        
        return

    def forward_mlp_decoder(self, inputs):
        c1, c2, c3, c4 = inputs

        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)
   
        _c4 = resize(_c4, size=inputs[1].size()[2:],mode='bilinear',align_corners=False).contiguous()
        _c3 = resize(_c3, size=inputs[1].size()[2:],mode='bilinear',align_corners=False).contiguous()
       
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))
        
        return _c, _c2, _c3, _c4

    def forward_winssm(self, x: torch.Tensor, c2, c3, c4, c1=None):
        out = [self.short_path(x), 
               resize(self.image_pool(x),
                      size=x.size()[2:],
                      mode='bilinear',
                      align_corners=self.align_corners).contiguous()]
        
        ## MMSCopE #################
        B, C, H, W = x.size()

        # obtain multi scale features
        x_2 = self.conv_downsample_2(x) # 1/2 resolution
        x_4 = self.conv_downsample_4(x) # 1/4 resolution

        # unshuffle all features to size 1/4 resolution (16x16 for 512 input res)
        x_2_unshuffle = F.pixel_unshuffle(x_2, downscale_factor=2)
        x_unshuffle = F.pixel_unshuffle(x, downscale_factor=4)

        # reduce channels
        x_unshuffle = self.reduce_channels[2](x_unshuffle)
        x_2_unshuffle = self.reduce_channels[1](x_2_unshuffle)
        x_4 = self.reduce_channels[0](x_4)

        multi_x = torch.cat([x_unshuffle, x_2_unshuffle, x_4], dim=1)

        _out = self.vssm(multi_x)

        _out = resize(
            _out, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
        
        _out_ = self.proj_out(_out)
        ############################
        
        c2 = c2 + _out_
        c3 = c3 + _out_
        c4 = c4 + _out_
 
        out += [_out, c2, c3, c4]

        out = self.cat(torch.cat(out, dim=1))
        
        return out
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        
        return output
    
    def forward(self, inputs):
        x = inputs
        x, c2, c3, c4 = self.forward_mlp_decoder(x)
        
        x = self.forward_winssm(x, c2, c3, c4)
        output = self.cls_seg(x)
        output = resize(output, size=self.image_size, mode='bilinear', 
                        align_corners=self.align_corners)
        
        return output

class MFFSegmentor(nn.Module):
    def __init__(self,
                 img_size=512,
                 in_chans=3,
                 patch_size=4,
                 enc_embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=LayerNorm2d,
                 ape=False,
                 feat_proj_dim=320,
                 dec_embed_dim=180,
                 num_classes=3,
                 **kwargs
                 ):
        super(MFFSegmentor, self).__init__()
        
        self.encoder = MFFEncoder(img_size, patch_size, in_chans, enc_embed_dim,
                                  depths, num_heads, mlp_ratio, qkv_bias, drop_rate,
                                  drop_path_rate, norm_layer, ape, **kwargs)
        self.decoder = MFFDecoder(img_size, num_classes, enc_embed_dim,
                                  feat_proj_dim, dec_embed_dim, qkv_bias,
                                  drop_path_rate, norm_layer, **kwargs)
        return
    
    def forward(self, image):
        features = self.encoder(image)
        mask = self.decoder(features)
        return mask

def build_segmentor(args, **kwargs):
    segmentor = MFFSegmentor(img_size = args.image_size, 
                             in_chans = args.input_channels, 
                             patch_size = args.encoder.patch_size,
                             enc_embed_dim = args.encoder.embed_dim,
                             depths = args.encoder.depths,
                             num_heads = args.encoder.num_heads,
                             mlp_ratio = args.encoder.mlp_ratio,
                             qkv_bias = args.encoder.qkv_bias,
                             drop_rate = args.encoder.drop_rate,
                             drop_path_rate = args.encoder.drop_path_rate,
                             ape = args.encoder.ape,
                             feat_proj_dim = args.decoder.feat_proj_dim,
                             dec_embed_dim = args.decoder.embed_dim,
                             num_classes = args.decoder.num_classes,
                             **kwargs
                             )
    return segmentor

if __name__ == '__main__':
    img = torch.rand((2,3,512,512), dtype=torch.float32)
    
    model = MFFSegmentor()
    result = model(img)
    
    print(result.shape)