# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
import numpy as np
import cv2


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type=''):
    if norm_type == '':
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'RegionNorm':
        return RNGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        assert 0


class RNGroupNorm(nn.GroupNorm):  # Region Normalization
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super(RNGroupNorm, self).__init__(num_groups, num_channels, eps, affine)
    
    def forward(self, input, mask_in=None):
        assert mask_in is not None
        B, C, H, W = input.size()
        assert C % self.num_groups == 0
        if mask_in.shape[2] == input.shape[2]:
            resized_mask = mask_in
        else:  # ideally we can add a soft mask / use another interpolation method
            resized_mask = F.interpolate(mask_in, size=(input.shape[2], input.shape[3]))
        x = input.view(B, self.num_groups, C // self.num_groups, H, W)
        resized_mask2 = resized_mask.unsqueeze(1)
        sum_feat = torch.sum((1.0 - resized_mask2) * x, dim=[2, 3, 4])  # B, NG
        num_feat = torch.sum(1.0 - resized_mask2, dim=[2, 3, 4])  # B, 1
        mu = (sum_feat / num_feat)  # B, NG
        mu = mu.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # B, NG, 1, 1, 1
        new_feat = (1.0 - resized_mask2) * x + resized_mask2 * mu
        # note: normalized features need to be rescaled according to mask area
        normalized_feat = F.group_norm(new_feat.view(B, C, H, W), self.num_groups, eps=self.eps) * torch.sqrt(num_feat.unsqueeze(-1).unsqueeze(-1) / (C // self.num_groups * H * W))
        return (normalized_feat * self.weight[None, :, None, None] + self.bias[None, :, None, None]) * (1.0 - resized_mask) + super(RNGroupNorm, self).forward(input) * resized_mask


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, conv_type='', downsample_legacy=True):
        super().__init__()
        self.with_conv = with_conv
        self.downsample_legacy = downsample_legacy
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = SpecialConv2d(in_channels,
                                      in_channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=0, conv_type=conv_type, downsample_legacy=self.downsample_legacy)

    def forward(self, x, mask_in=None):
        if self.with_conv:  # there was a bug here
            if self.downsample_legacy:  # the original implementation
                pad = (0,1,0,1)  
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)      
            x = self.conv(x, mask_in=mask_in)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, conv_type='', norm_type=''):
        super().__init__()
        self.in_channels = in_channels
        self.conv_type = conv_type
        self.norm_type = norm_type
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type=self.norm_type)
        self.conv1 = SpecialConv2d(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1, conv_type=self.conv_type)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels, norm_type=self.norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = SpecialConv2d(out_channels,
                                   out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1, conv_type=self.conv_type)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                assert 0  # might be changed
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)  # linear layer

    def forward(self, x, temb, mask_in=None):
        h = x
        if self.norm_type == '':
            h = self.norm1(h)
        elif self.norm_type == 'RegionNorm':
            h = self.norm1(h, mask_in=mask_in)
        else:
            assert 0
        
        h = nonlinearity(h)
        h = self.conv1(h, mask_in=mask_in)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        if self.norm_type == '':
            h = self.norm2(h)
        elif self.norm_type == 'RegionNorm':
            h = self.norm2(h, mask_in=mask_in)
        else:
            assert 0
            
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h, mask_in=mask_in)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, conv_type='', norm_type=''):
        super().__init__()
        self.in_channels = in_channels
        self.conv_type = conv_type
        self.norm_type = norm_type
        self.norm = Normalize(in_channels, norm_type=self.norm_type)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x, mask_in=None):  # all linear layers
        h_ = x
        if self.norm_type == '':
            h_ = self.norm(h_)  # different from x
        elif self.norm_type == 'RegionNorm':
            h_ = self.norm(h_, mask_in=mask_in)
        else:
            assert 0
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape  # bs, 512, 16, 16
        q = q.reshape(b,c,h*w)  # bs, 512, 16x16
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        if self.conv_type == 'PartialConv' and mask_in is not None:
            assert mask_in is not None
            if mask_in.shape[2] == x.shape[2]:
                resized_mask = mask_in
            else:
                resized_mask = F.interpolate(mask_in, size=(x.shape[2], x.shape[3]))
            if resized_mask.shape[0] == 1:  # one batch has the same mask
                resized_mask_flatten = resized_mask.squeeze().reshape(-1)
                masked_indices = torch.nonzero(resized_mask_flatten).squeeze()
                attention_mask = torch.ones(w_.shape[1], w_.shape[2], device=x.device)
                attention_mask[:, masked_indices] = 0.0
                attention_mask[masked_indices, :] = 1.0
                attention_mask = attention_mask.view(1, w_.shape[1], w_.shape[2])
            else:
                attention_mask = torch.ones(resized_mask.shape[0], w_.shape[1], w_.shape[2], device=x.device)
                for bid in range(resized_mask.shape[0]):
                    resized_mask_flatten = resized_mask[bid].squeeze().reshape(-1)
                    masked_indices = torch.nonzero(resized_mask_flatten).squeeze()
                    attention_mask[bid, :, masked_indices] = 0.0
                    attention_mask[bid, masked_indices, :] = 1.0
                    
            w_ = w_.masked_fill(attention_mask == 0, float('-inf'))

        w_ = torch.nn.functional.softmax(w_, dim=2)  # important for attention computation; can be combined with mask input

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, t=None):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SpecialConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        conv_type: str = '',  # '' | 'PartialConv' |       used to define the type of mask-conditioned conv
        downsample_legacy: bool = True,  # it's better to set it to be False
    ):
        self.conv_type = conv_type
        self.downsample_legacy = downsample_legacy
        super(SpecialConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        if self.conv_type == 'PartialConv':
            self.weight_maskUpdater = torch.ones(1, 1, 3, 3)  # similar to weight, on cpu

    def forward(self, input, mask_in=None):  # mask_in might need to be resized
        if self.conv_type == '' or mask_in is None:  # is True during pre-training stage of PartialConv
            assert mask_in is None
            output = super(SpecialConv2d, self).forward(input)
            return output
        elif self.conv_type == 'PartialConv':
            assert mask_in is not None
            assert self.padding_mode == 'zeros'

            # resize the mask_in to the size of input
            if mask_in.shape[2] == input.shape[2]:
                resized_mask = mask_in
            else:  # ideally we can add a soft mask / use another interpolation method
                resized_mask = F.interpolate(mask_in, size=(input.shape[2], input.shape[3]))
            if self.stride[0] == 1:
                update_mask = F.conv2d(resized_mask, self.weight_maskUpdater.to(resized_mask.device), stride=self.stride, padding=self.padding, dilation=self.dilation)
                update_mask = torch.clamp(update_mask, 0, 1)  # dilated input mask
                leaked_mask = torch.clamp(update_mask - resized_mask, 0, 1)  # M_l
                output = super(SpecialConv2d, self).forward(input) * (1.0 - leaked_mask) + super(SpecialConv2d, self).forward(input * (1.0 - resized_mask)) * leaked_mask
            elif self.stride[0] == 2:  # debug more TODO
                pad = (0, 1, 0, 1)
                resized_mask2 = F.pad(resized_mask, pad, mode="constant", value=0)
                # stride-2 padding-0 convolution
                update_mask = F.conv2d(resized_mask2, self.weight_maskUpdater.to(resized_mask.device), stride=self.stride, padding=self.padding, dilation=self.dilation)
                update_mask = torch.clamp(update_mask, 0, 1)  # dilated input mask  
                #haha = update_mask - F.interpolate(resized_mask, scale_factor=0.5)
                #haha = haha.cpu().squeeze().numpy()
                #haha = (update_mask - F.interpolate(resized_mask, scale_factor=0.5)).cpu().squeeze().numpy()
                #assert np.amin(haha) >= 0.0
                leaked_mask = torch.clamp(update_mask - F.interpolate(resized_mask, scale_factor=0.5), 0, 1)  # M_l
                if self.downsample_legacy:
                    output = super(SpecialConv2d, self).forward(input) * (1.0 - leaked_mask) + super(SpecialConv2d, self).forward(input * (1.0 - resized_mask)) * leaked_mask
                else:
                    output = super(SpecialConv2d, self).forward(F.pad(input, pad, mode="constant", value=0)) * (1.0 - leaked_mask) + super(SpecialConv2d, self).forward(F.pad(input * (1.0 - resized_mask), pad, mode="constant", value=0)) * leaked_mask
            else:
                assert 0
            # TS convolution
            return output
        else:
            assert 0
        
    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, conv_type='', norm_type='', downsample_legacy=True, **ignore_kwargs):  # set downsample_legacy=False to disable the downsample bug
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_type = conv_type
        self.norm_type = norm_type
        # downsampling
        self.conv_in = SpecialConv2d(in_channels,
                                     self.ch,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1, conv_type=self.conv_type)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()  # a list of 5 modules
        for i_level in range(self.num_resolutions):  # 5
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout, conv_type=self.conv_type, norm_type=self.norm_type))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, conv_type=self.conv_type, norm_type=self.norm_type))
            down = nn.Module()  # block --> attn --> downsample
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:  # i_level != 4 ----- the last one doesn't have downsampling layer
                down.downsample = Downsample(block_in, resamp_with_conv, conv_type=self.conv_type, downsample_legacy=downsample_legacy)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout, conv_type=self.conv_type, norm_type=self.norm_type)
        self.mid.attn_1 = AttnBlock(block_in, conv_type=self.conv_type, norm_type=self.norm_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout, conv_type=self.conv_type, norm_type=self.norm_type)

        # end
        self.norm_out = Normalize(block_in, norm_type=self.norm_type)
        self.conv_out = SpecialConv2d(block_in, 
                                      2*z_channels if double_z else z_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1, conv_type=self.conv_type)


    def forward(self, x, mask_in=None, output_hidden=False):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
        if self.conv_type == '':  # original convolution
            # timestep embedding
            temb = None

            # downsampling
            hs = [self.conv_in(x)]
            for i_level in range(self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    hs.append(h)
                if i_level != self.num_resolutions-1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
            # middle
            h = hs[-1]
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

            # end
            h = self.norm_out(h)
            h = nonlinearity(h)
            h = self.conv_out(h)
        elif self.conv_type == 'PartialConv':
            # timestep embedding
            temb = None

            # downsampling
            hs = [self.conv_in(x, mask_in=mask_in)]
            for i_level in range(self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    h = self.down[i_level].block[i_block](hs[-1], temb, mask_in=mask_in)
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h, mask_in=mask_in)
                    hs.append(h)
                if i_level != self.num_resolutions-1:
                    hs.append(self.down[i_level].downsample(hs[-1], mask_in=mask_in))
            # middle
            h = hs[-1]
            h = self.mid.block_1(h, temb, mask_in=mask_in)
            h = self.mid.attn_1(h, mask_in=mask_in)
            h = self.mid.block_2(h, temb, mask_in=mask_in)

            # end
            if self.norm_type == '':
                h = self.norm_out(h)
            elif self.norm_type == 'RegionNorm':
                h = self.norm_out(h, mask_in=mask_in)
            else:
                assert 0
            h = nonlinearity(h)
            h = self.conv_out(h, mask_in=mask_in)  
        else:
            assert 0          
        if output_hidden:
            return hs
        else:
            return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VUNet(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, c_channels,
                 resolution, z_channels, use_timestep=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(c_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.z_in = torch.nn.Conv2d(z_channels,
                                    block_in,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=2*block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, z):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        z = self.z_in(z)
        h = torch.cat((h,z),dim=1)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

