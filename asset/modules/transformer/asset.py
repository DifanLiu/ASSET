import copy
import math
import random
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BartConfig
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel, 
    _make_causal_mask,
    _expand_mask,
)
from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)


def to_block_order(input_features, block_h=8):  # to "blockify" order
    B, L, C = input_features.shape
    H = int(math.sqrt(L))  # 16
    token_h = H // block_h  # each block has token_h x token_h tokens
    ans = input_features.view(B, H, H, C).view(B, block_h, token_h, block_h, token_h, C).permute(0, 1, 3, 2, 4, 5).contiguous()\
        .view(B, block_h, block_h, -1, C).view(B, block_h * block_h, -1, C).view(B, -1, C)
    return ans


def to_normal_order(input_features, block_h=8):  # to "normal" order
    B, L, C = input_features.shape
    H = int(math.sqrt(L))  # 16
    token_h = H // block_h  # each block has token_h x token_h tokens
    ans = input_features.view(B, block_h * block_h, -1, C).view(B, block_h, block_h, -1, C).view(B, block_h, block_h, token_h, token_h, C)\
        .permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, H, C).view(B, -1, C)
    return ans


def normal_to_strip_order(input_features, block_h=16, block_w=4):  # 'normal' to 'strip' order
    B, L, C = input_features.shape
    H = int(math.sqrt(L))  # 16
    token_h = H // block_h  # 1
    token_w = H // block_w  # 4
    ans = input_features.view(B, H, H, C).view(B, block_h, token_h, block_w, token_w, C).permute(0, 1, 3, 2, 4, 5).contiguous()\
        .view(B, block_h, block_w, -1, C).view(B, block_h * block_w, -1, C).view(B, -1, C)
    
    return ans


def strip_to_normal_order(input_features, block_h=16, block_w=4):  # 'normal' to 'strip' order
    B, L, C = input_features.shape
    H = int(math.sqrt(L))  # 16
    token_h = H // block_h  # 1
    token_w = H // block_w  # 4
    ans = input_features.view(B, block_h * block_w, -1, C).view(B, block_h, block_w, -1, C).view(B, block_h, block_w, token_h, token_w, C)\
        .permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, H, C).view(B, -1, C)
    
    return ans


def generate_block_ids(input_features, block_h=16, block_w=4):  # output is normal order
    B, L, _ = input_features.shape
    H = int(math.sqrt(L))  # 16, 32, 64
    token_h = H // block_h  # 1
    token_w = H // block_w  # 4

    ans = torch.zeros(B, block_h * block_w, token_h * token_w, dtype=torch.int64) + torch.arange(block_h * block_w, dtype=torch.int64).unsqueeze(0).unsqueeze(-1)
    ans = ans.to(input_features.device).view(B, block_h, block_w, token_h, token_w).permute(0, 1, 3, 2, 4).contiguous().view(B, H, H).view(B, -1)
    return ans


class ASSETConfig(BartConfig):
    model_type = "asset"
    def __init__(self, block_h=8, 
                 block_w=8,
                 PEG_list=[], PEG_ks=3, **kwargs):
        super().__init__(**kwargs)
        self.block_h = block_h  # default 8x8 blocks across different resolutions
        self.block_w = block_w  # default grid size: 16x4
        self.PEG_list = PEG_list  # ids of layers to insert PEG     -1   layer-1  0   layer-2  1       e0, e1,    d0, d1

        self.PEG_ks = PEG_ks


class ASSETAttention(nn.Module):  # based on BartAttention     Done for 256-resolution
    """SGA attention + the original attention"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        double_ic: bool = False,
        block_h: int = 8,
        block_w: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.block_h = block_h
        self.block_w = block_w
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        if double_ic:
            factor_dim = 2
        else:
            factor_dim = 1
        self.k_proj = nn.Linear(factor_dim * embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(factor_dim * embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(factor_dim * embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def torch_gather_b2(params, indices):
        # this operation is equivalent to tf.gather when batch_dims=2

        if params.shape[:2] != indices.shape[:2]:  # bs, nh
            raise ValueError(
                f"Make sure that the first two dimensions of params and indices are identical, \
                but they are params: {params.shape[:2]} vs. indices: {params.shape[:2]}"
            )
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]  # 62x3
        num_indices_to_pick_from = params.shape[2]  # 64 nb

        indices_shift = (
            torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)
            // num_indices_to_gather
            * num_indices_to_pick_from
        )  # bs * nh * 62x3      0...0, 1...1, 23   --->  0...0, 64...64, 1472...1472

        flattened_indices = indices.view(-1) + indices_shift  # smart implementation
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])  # bs, nh12, nb64, 64, c64   --->  bs*nh12*nb64, 64, c64

        out_flattened = flattened_params.index_select(0, flattened_indices)  #   bs * nh * 62x3, 64, c64

        out = out_flattened.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])  # bs, nh, 62x3, 64, c64   1 block has 64 tokens
        return out

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication with transpose"""
        # faster replacement of torch.einsum (bhqd,bhkd->bhqk)
        return torch.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication"""
        # faster replacement of torch.einsum ("bhqk,bhkd->bhqd")
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )
                            
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward_sparse_encoder_SGA(self, hidden_states, rand_attn=None):  #  bs, 4096, C      The implementation follows BigBird
        bsz, seqlen, _ = hidden_states.size()  # bs, 4096
        num_blocks = self.block_h * self.block_w  # default 64 blocks
        block_size = seqlen // num_blocks  # depends on the input sequence length
        query_layer = self.transpose_for_scores(self.q_proj(hidden_states))  # bs, nh, L, headC      by heads
        key_layer = self.transpose_for_scores(self.k_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.v_proj(hidden_states))
        n_rand_blocks = 3
        # ------ guidance
        rand_attn = rand_attn.contiguous()
            

        blocked_query_matrix = query_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)  # bs, nh, nb, nt, 64c
        blocked_key_matrix = key_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)
        # ------ preparing block for randn attn
        gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)   # bs, nh, 64x3, nt, 64c 
        gathered_key = gathered_key.view(bsz, self.num_heads, num_blocks, n_rand_blocks * block_size, -1)  # bs, nh, nb, 3xnt tokens, 64c
        gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
        gathered_value = gathered_value.view(bsz, self.num_heads, num_blocks, n_rand_blocks * block_size, -1)
        # ------ 1st block (not global in this function)
        first_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                gathered_key[:, :, 0],
            ],
            dim=2,
        )  # bs, nh, _, 64c
        first_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                gathered_value[:, :, 0],
            ],
            dim=2,
        )  # bs, nh, _, 64c
        first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], first_key_mat, ndim=4)  # bs, nh, nt, _
        first_product = first_product * self.scaling
        first_attn_weights = nn.functional.softmax(first_product, dim=-1)
        first_context_layer = self.torch_bmm_nd(first_attn_weights, first_value_mat, ndim=4)  # bs, nh, nt, 64c
        first_context_layer.unsqueeze_(2)  # bs, nh, 1, nt, 64C
        
        # ------ Middle blocks (sliding attn is calculated using special trick of shifting tokens as discussed in paper)
        exp_blocked_key_matrix = torch.cat(
            [blocked_key_matrix[:, :, 0:-2], blocked_key_matrix[:, :, 1:-1], blocked_key_matrix[:, :, 2:]], dim=3
        )  #  bs, nh, 62, 3 * nt, 64c
        exp_blocked_value_matrix = torch.cat(
            [blocked_value_matrix[:, :, 0:-2], blocked_value_matrix[:, :, 1:-1], blocked_value_matrix[:, :, 2:]], dim=3,
        ) 
        middle_query_matrix = blocked_query_matrix[:, :, 1:-1]  #  bs, nh, 62, nt, 64c
        # sliding attention scores
        inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)  #  bs, nh, 62, nt, 3 * nt
        inner_band_product = inner_band_product * self.scaling
        # randn attention scores
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)  # bs, nh, 62, nt, 3 * nt
        rand_band_product = rand_band_product * self.scaling
        
        # completing attention scores matrix for all middle blocks
        band_product = torch.cat([inner_band_product, rand_band_product], dim=-1)  # bs, nh, 62, nt, 6 * nt
        # safely doing softmax since attention matrix is completed     1 block, 3 blocks, 3 blocks, 1 block
        attn_weights = nn.functional.softmax(band_product, dim=-1)  # bs, nh, 62, nt, 6 * nt
        # contribution of sliding keys
        context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :,  : 3 * block_size], exp_blocked_value_matrix, ndim=5)
        # adding contribution of random keys
        context_layer += self.torch_bmm_nd(attn_weights[:, :, :, :, 3 * block_size :], gathered_value[:, :, 1:-1], ndim=5)
        
        # ------ last block (global)
        last_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -1],
            ],
            dim=2,
        )  # bs, nh, _, 64c
        last_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -1],
            ],
            dim=2,
        )  # bs, nh, _, 64c
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], last_key_mat, ndim=4)  # bs, nh, nt, _
        last_product = last_product * self.scaling
        last_attn_weights = nn.functional.softmax(last_product, dim=-1)
        last_context_layer = self.torch_bmm_nd(last_attn_weights, last_value_mat, ndim=4)  # bs, nh, nt, 64c
        last_context_layer.unsqueeze_(2)  # bs, nh, 1, nt, 64C
        
        # ------ combining representations of all blocks
        context_layer = torch.cat(
            [first_context_layer, context_layer, last_context_layer],
            dim=2,
        )  # bs, nh, nb, nt, 64C
        context_layer = context_layer.view((bsz, self.num_heads, seqlen, -1))  # bs, nh, L, 64C
        context_layer = torch.transpose(context_layer, 1, 2)  # bs, L, nh, 64C
        assert self.dropout == 0.0
        context_layer = context_layer.contiguous().view(bsz, seqlen, -1)  # bs, L, C

        return context_layer

    def forward_sparse_decoder_causal(self, hidden_states, rand_attn=None):
        start_token_states = hidden_states[:, 0:1, :]  # query is useless
        start_token_key = self.transpose_for_scores(self.k_proj(start_token_states))  # bs, nh, 1, 64c
        start_token_value = self.transpose_for_scores(self.v_proj(start_token_states))
        
        hidden_states = hidden_states[:, 1:, :]
        bsz, seqlen, _ = hidden_states.size()  # bs, 4096
        num_blocks = self.block_h * self.block_w  # default 64 blocks
        block_size = seqlen // num_blocks  # depends on the input sequence length
        query_layer = self.transpose_for_scores(self.q_proj(hidden_states))  # bs, nh, L, headC      by heads
        key_layer = self.transpose_for_scores(self.k_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.v_proj(hidden_states))
        past_key_value = (torch.cat([start_token_key, key_layer], dim=2), torch.cat([start_token_value, value_layer], dim=2))
        n_rand_blocks = 3
        n_neighbor_blocks = 3  # including the block itself
        NBR = num_blocks - (n_rand_blocks + n_neighbor_blocks) + 1   # number of blocks with randomness    why+1:  the 6th block "have" randomness
        # block_(n_rand_blocks + n_neighbor_blocks - 1)  have **full** preceding blocks
        
        # ------ guidance
        rand_attn = rand_attn.contiguous()
                        
        blocked_query_matrix = query_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)  # bs, nh, nb, nt, 64c
        blocked_key_matrix = key_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)
        # ------ preparing block for randn attn
        gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)   # bs, nh, 59x3, nt, 64c 
        gathered_key = gathered_key.view(bsz, self.num_heads, NBR, n_rand_blocks * block_size, -1)  #  bs, nh, 59, 3xnt tokens, 64c
        gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
        gathered_value = gathered_value.view(bsz, self.num_heads, NBR, n_rand_blocks * block_size, -1)
        final_list = []
        # ------ blocks without randomness
        #  bs, nh, nt, 64c x bs, nh, _, 64c  = bs, nh, nt, _
        for si in range(0, n_rand_blocks + n_neighbor_blocks - 1):  # [0, 1, 2, 3, 4]
            # bs, nh, _, 64c
            full_key = torch.cat((start_token_key, blocked_key_matrix[:, :, 0:(si + 1)].view(bsz, self.num_heads, (si + 1) * block_size, -1)), dim=2)
            full_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, si], full_key, ndim=4)  # bs, nh, nt, _
            full_product = full_product * self.scaling
            full_mask = self.generate_causal_mask(full_product)
            full_product += full_mask
            full_attn_weights = nn.functional.softmax(full_product, dim=-1)
            full_value = torch.cat((start_token_value, blocked_value_matrix[:, :, 0:(si + 1)].view(bsz, self.num_heads, (si + 1) * block_size, -1)), dim=2)
            full_context_layer = self.torch_bmm_nd(full_attn_weights, full_value, ndim=4)  # bs, nh, nt, 64C
            final_list.append(full_context_layer.unsqueeze_(2))  # bs, nh, 1, nt, 64C
        
        # ------ blocks with randomness and with fixed length
        exp_blocked_key_matrix = torch.cat(
            [blocked_key_matrix[:, :, (-NBR-2):(-2)], blocked_key_matrix[:, :, (-NBR-1):(-1)], blocked_key_matrix[:, :, (-NBR):]], dim=3
        )  #  bs, nh, 59b, 3 * nt, 64c
        exp_blocked_value_matrix = torch.cat(
            [blocked_value_matrix[:, :, (-NBR-2):(-2)], blocked_value_matrix[:, :, (-NBR-1):(-1)], blocked_value_matrix[:, :, (-NBR):]], dim=3
        )  #  bs, nh, 59b, 3 * nt, 64c
        middle_query_matrix = blocked_query_matrix[:, :, (-NBR):]  #  bs, nh, 59b, nt, 64c
        # randn attention scores
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, torch.cat([gathered_key, exp_blocked_key_matrix], dim=3), ndim=5)  # bs, nh, 59b, nt, 6 * nt
        rand_band_product = rand_band_product * self.scaling
        # [start] attention scores
        start_band_product = torch.einsum("bhlqd,bhkd->bhlqk", middle_query_matrix, start_token_key)  # bs, nh, 59b, nt, 1
        start_band_product = start_band_product * self.scaling
        # completing attention scores matrix for all middle blocks
        band_product = torch.cat([start_band_product, rand_band_product], dim=-1)  # bs, nh, 59b, nt, 1 + 6 * nt
        band_mask = self.generate_causal_mask(band_product[:, :, 0]).unsqueeze(2)
        band_product += band_mask
        # safely doing softmax since attention matrix is completed     1 block, 3 blocks, 3 blocks, 1 block
        attn_weights = nn.functional.softmax(band_product, dim=-1)
        # contribution of sliding keys    # bs, nh, 59b, nt, 64c
        context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :, 1:], torch.cat([gathered_value, exp_blocked_value_matrix], dim=3), ndim=5)
        context_layer += torch.einsum("bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, 0:1], start_token_value)
        final_list.append(context_layer)
        # ------ combining representations of all blocks
        context_layer = torch.cat(final_list, dim=2)  # bs, nh, nb, nt, 64C
        context_layer = context_layer.view((bsz, self.num_heads, seqlen, -1))  # bs, nh, L, 64C
        context_layer = torch.cat([start_token_value, context_layer], dim=2)  # bs, nh, L + 1, 64C
        context_layer = torch.transpose(context_layer, 1, 2)  # bs, L + 1, nh, 64C
        
        assert self.dropout == 0.0
        context_layer = context_layer.contiguous().view(bsz, seqlen + 1, -1)  # bs, L + 1, C

        return context_layer, past_key_value

    def generate_causal_mask(self, product_tensor):
        _, _, ql, kl = product_tensor.size()  # bs, 4096
        causal_mask = torch.full((ql, kl), float("-inf"))
        causal_mask.masked_fill_(torch.arange(kl) <= (torch.arange(ql) + kl - ql).view(ql, 1), 0)
        causal_mask = causal_mask.to(product_tensor.device)
        return causal_mask.unsqueeze(0).unsqueeze(0)
    
    def forward_sparse_decoder_cross_SGA(self, hidden_states, key_value_states, rand_attn=None):   
        
        start_token_states = hidden_states[:, 0:1, :]  # query is necessary
        start_token_query = self.transpose_for_scores(self.q_proj(start_token_states))  # bs, nh, 1, 64c
        
        hidden_states = hidden_states[:, 1:, :]
        bsz, seqlen, _ = hidden_states.size()  # bs, 4096
        num_blocks = self.block_h * self.block_w  # default 64 blocks
        block_size = seqlen // num_blocks  # depends on the input sequence length
        query_layer = self.transpose_for_scores(self.q_proj(hidden_states))  # bs, nh, L, headC      by heads
        key_layer = self.transpose_for_scores(self.k_proj(key_value_states))
        value_layer = self.transpose_for_scores(self.v_proj(key_value_states))
        past_key_value = (key_layer, value_layer)
        n_rand_blocks = 3
        # ------ generate random attention positions
        rand_attn = rand_attn.contiguous()
                  
        blocked_query_matrix = query_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)  # bs, nh, nb, nt, 64c
        blocked_key_matrix = key_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)
        # ------ preparing block for randn attn
        gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)   # bs, nh, 64x3, nt, 64c 
        gathered_key = gathered_key.view(bsz, self.num_heads, num_blocks, n_rand_blocks * block_size, -1)  #  bs, nh, nb, 3xnt tokens, 64c
        gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
        gathered_value = gathered_value.view(bsz, self.num_heads, num_blocks, n_rand_blocks * block_size, -1)
        # ------ [start] token (global)
        start_product = self.torch_bmm_nd_transpose(start_token_query, key_layer, ndim=4)
        start_product = start_product * self.scaling
        start_attn_weights = nn.functional.softmax(start_product, dim=-1)
        start_context_layer = self.torch_bmm_nd(start_attn_weights, value_layer, ndim=4)  # bs, nh, 1, 64C    "hidden features"
        # ------ 1st block (not global)
        first_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                gathered_key[:, :, 0],
            ],
            dim=2,
        )  # bs, nh, _, 64c
        first_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                gathered_value[:, :, 0],
            ],
            dim=2,
        )  # bs, nh, _, 64c
        first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], first_key_mat, ndim=4)  # bs, nh, nt, _
        first_product = first_product * self.scaling
        first_attn_weights = nn.functional.softmax(first_product, dim=-1)
        first_context_layer = self.torch_bmm_nd(first_attn_weights, first_value_mat, ndim=4)  # bs, nh, nt, 64c
        first_context_layer.unsqueeze_(2)  # bs, nh, 1, nt, 64C
        
        # ------ Middle blocks (sliding attn is calculated using special trick of shifting tokens as discussed in paper)
        exp_blocked_key_matrix = torch.cat(
            [blocked_key_matrix[:, :, 0:-2], blocked_key_matrix[:, :, 1:-1], blocked_key_matrix[:, :, 2:]], dim=3
        )  #  bs, nh, 62, 3 * nt, 64c
        exp_blocked_value_matrix = torch.cat(
            [blocked_value_matrix[:, :, 0:-2], blocked_value_matrix[:, :, 1:-1], blocked_value_matrix[:, :, 2:]], dim=3,
        ) 
        middle_query_matrix = blocked_query_matrix[:, :, 1:-1]  #  bs, nh, 62, nt, 64c
        # sliding attention scores
        inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)  #  bs, nh, 62, nt, 3 * nt
        inner_band_product = inner_band_product * self.scaling
        # randn attention scores
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)  # bs, nh, 62, nt, 3 * nt
        rand_band_product = rand_band_product * self.scaling
        
        # completing attention scores matrix for all middle blocks
        band_product = torch.cat([inner_band_product, rand_band_product], dim=-1)  # bs, nh, 62, nt, 6 * nt
        # safely doing softmax since attention matrix is completed     1 block, 3 blocks, 3 blocks, 1 block
        attn_weights = nn.functional.softmax(band_product, dim=-1)  # bs, nh, 62, nt, 6 * nt
        # contribution of sliding keys
        context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :,  : 3 * block_size], exp_blocked_value_matrix, ndim=5)
        # adding contribution of random keys
        context_layer += self.torch_bmm_nd(attn_weights[:, :, :, :, 3 * block_size :], gathered_value[:, :, 1:-1], ndim=5)

        # ------ last block (global)
        last_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -1],
            ],
            dim=2,
        )  # bs, nh, _, 64c
        last_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -1],
            ],
            dim=2,
        )  # bs, nh, _, 64c
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], last_key_mat, ndim=4)  # bs, nh, nt, _
        last_product = last_product * self.scaling
        last_attn_weights = nn.functional.softmax(last_product, dim=-1)
        last_context_layer = self.torch_bmm_nd(last_attn_weights, last_value_mat, ndim=4)  # bs, nh, nt, 64c
        last_context_layer.unsqueeze_(2)  # bs, nh, 1, nt, 64C     
        
        
        # ------ combining representations of all blocks
        context_layer = torch.cat(
            [first_context_layer, context_layer, last_context_layer],
            dim=2,
        )  # bs, nh, nb, nt, 64C
        context_layer = context_layer.view((bsz, self.num_heads, seqlen, -1))  # bs, nh, L, 64C
        context_layer = torch.cat([start_context_layer, context_layer], dim=2)  # bs, nh, L + 1, 64C
        context_layer = torch.transpose(context_layer, 1, 2)  # bs, L + 1, nh, 64C
        assert self.dropout == 0.0
        context_layer = context_layer.contiguous().view(bsz, seqlen + 1, -1)  # bs, L + 1, C

        return context_layer, past_key_value
    
    @torch.no_grad()
    def forward_sparse_decoder_causal_fast(self, hidden_states, rand_attn_full, old_info, care_start, care_end, token_n, do_record):
        if 's_start_token_key' not in old_info.keys():
            assert care_start == 0
            start_token_states = hidden_states[:, 0:1, :]  # query is useless
            old_info['s_start_token_key'] = self.transpose_for_scores(self.k_proj(start_token_states))  # bs, nh, 1, 64c
            old_info['s_start_token_value'] = self.transpose_for_scores(self.v_proj(start_token_states))
        start_token_key = old_info['s_start_token_key']
        start_token_value = old_info['s_start_token_value']
            
        if care_start == 0:
            hidden_states = hidden_states[:, 1:, :]
            
        bsz, partial_l, _ = hidden_states.size()  # bs, 4096
        num_blocks = self.block_h * self.block_w  # default 64 blocks
        block_size = token_n  # depends on the input sequence length
        partial_nb = partial_l // block_size
        
        query_layer = self.transpose_for_scores(self.q_proj(hidden_states))  # bs, nh, L, headC      by heads    not full sequence
        key_layer = self.transpose_for_scores(self.k_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.v_proj(hidden_states))
        
        n_rand_blocks = 3
        n_neighbor_blocks = 3  # including the block itself
        NBR = num_blocks - (n_rand_blocks + n_neighbor_blocks) + 1   # number of blocks with randomness    why+1:  the 6th block "have" randomness   59
        #---- blockify
        blocked_query_matrix_partial = query_layer.view(bsz, self.num_heads, partial_nb, block_size, -1)  # bs, nh, nb, nt, 64c
        blocked_key_matrix_partial = key_layer.view(bsz, self.num_heads, partial_nb, block_size, -1)
        blocked_value_matrix_partial = value_layer.view(bsz, self.num_heads, partial_nb, block_size, -1)
        
        if 's_blocked_key_matrix' in old_info.keys():
            assert care_start != 0
            assert old_info['s_blocked_key_matrix'].shape[2] == care_start  # sanity check
            blocked_key_matrix = torch.cat((old_info['s_blocked_key_matrix'], blocked_key_matrix_partial), dim=2)
            blocked_value_matrix = torch.cat((old_info['s_blocked_value_matrix'], blocked_value_matrix_partial), dim=2)
        else:
            assert care_start == 0
            blocked_key_matrix = blocked_key_matrix_partial
            blocked_value_matrix = blocked_value_matrix_partial
            
        if do_record:  # save time
            old_info['s_blocked_key_matrix'] = blocked_key_matrix
            old_info['s_blocked_value_matrix'] = blocked_value_matrix
                
        final_list = []
        for si in range(0, n_rand_blocks + n_neighbor_blocks - 1):  # [0, 1, 2, 3, 4]
            if si < care_start or si >= care_end:  # not in this partial sequence
                continue
            # si_in_partial = si_in_64 - care_start
            # bs, nh, _, 64c
            full_key = torch.cat((start_token_key, blocked_key_matrix[:, :, 0:(si + 1)].view(bsz, self.num_heads, (si + 1) * block_size, -1)), dim=2)
            full_product = self.torch_bmm_nd_transpose(blocked_query_matrix_partial[:, :, si - care_start], full_key, ndim=4)  # bs, nh, nt, _
            full_product = full_product * self.scaling
            full_mask = self.generate_causal_mask(full_product)
            full_product += full_mask
            full_attn_weights = nn.functional.softmax(full_product, dim=-1)
            full_value = torch.cat((start_token_value, blocked_value_matrix[:, :, 0:(si + 1)].view(bsz, self.num_heads, (si + 1) * block_size, -1)), dim=2)
            full_context_layer = self.torch_bmm_nd(full_attn_weights, full_value, ndim=4)  # bs, nh, nt, 64C
            final_list.append(full_context_layer.unsqueeze_(2))  # bs, nh, 1, nt, 64C  
        remaining_nb = partial_nb - len(final_list)
        start_bid_have_random =  care_end - remaining_nb  # [start_bid_have_random, care_end) in 64 coordinates   have random blocks
        # ------ preparing block for randn attn  # be careful here     
        if remaining_nb > 0:   
            rand_attn = rand_attn_full[:, :, (start_bid_have_random - (num_blocks - NBR)):(care_end - (num_blocks - NBR))].contiguous()  # partial
            gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)   # bs, nh, 59x3, nt, 64c 
            gathered_key = gathered_key.view(bsz, self.num_heads, remaining_nb, n_rand_blocks * block_size, -1)  #  bs, nh, 59, 3xnt tokens, 64c
            gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
            gathered_value = gathered_value.view(bsz, self.num_heads, remaining_nb, n_rand_blocks * block_size, -1)
            
            exp_blocked_key_matrix = torch.cat(
                [blocked_key_matrix[:, :, (start_bid_have_random-2):(care_end-2)], blocked_key_matrix[:, :, (start_bid_have_random-1):(care_end-1)], blocked_key_matrix[:, :, start_bid_have_random:care_end]], dim=3
                )  #  bs, nh, 59b, 3 * nt, 64c
            exp_blocked_value_matrix = torch.cat(
                [blocked_value_matrix[:, :, (start_bid_have_random-2):(care_end-2)], blocked_value_matrix[:, :, (start_bid_have_random-1):(care_end-1)], blocked_value_matrix[:, :, start_bid_have_random:care_end]], dim=3
                )  #  bs, nh, 59b, 3 * nt, 64c
            middle_query_matrix = blocked_query_matrix_partial[:, :, (start_bid_have_random-care_start):(care_end-care_start)]  #  bs, nh, 59b, nt, 64c
            # randn attention scores
            rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, torch.cat([gathered_key, exp_blocked_key_matrix], dim=3), ndim=5)  # bs, nh, 59b, nt, 6 * nt
            rand_band_product = rand_band_product * self.scaling
            # [start] attention scores
            start_band_product = torch.einsum("bhlqd,bhkd->bhlqk", middle_query_matrix, start_token_key)  # bs, nh, 59b, nt, 1
            start_band_product = start_band_product * self.scaling
            # completing attention scores matrix for all middle blocks
            band_product = torch.cat([start_band_product, rand_band_product], dim=-1)  # bs, nh, 59b, nt, 1 + 6 * nt
            band_mask = self.generate_causal_mask(band_product[:, :, 0]).unsqueeze(2)
            band_product += band_mask
            # safely doing softmax since attention matrix is completed     1 block, 3 blocks, 3 blocks, 1 block
            attn_weights = nn.functional.softmax(band_product, dim=-1)
            context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :, 1:], torch.cat([gathered_value, exp_blocked_value_matrix], dim=3), ndim=5)
            context_layer += torch.einsum("bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, 0:1], start_token_value)
            final_list.append(context_layer)

        # ------ combining representations of all blocks
        context_layer = torch.cat(final_list, dim=2)  # bs, nh, nb, nt, 64C
        assert context_layer.shape[2] == partial_nb
        context_layer = context_layer.view((bsz, self.num_heads, partial_l, -1))  # bs, nh, L, 64C
        if care_start == 0:
            context_layer = torch.cat([start_token_value, context_layer], dim=2)  # bs, nh, L + 1, 64C
            context_layer = torch.transpose(context_layer, 1, 2)  # bs, L + 1, nh, 64C
            context_layer = context_layer.contiguous().view(bsz, partial_l + 1, -1)  # bs, L + 1, C
        else:
            context_layer = torch.transpose(context_layer, 1, 2)  # bs, L, nh, 64C
            context_layer = context_layer.contiguous().view(bsz, partial_l, -1)  # bs, L, C
        return context_layer, old_info
    
    @torch.no_grad()
    def forward_sparse_decoder_cross_SGA_fast(self, hidden_states, key_value_states, rand_attn, old_info, care_start, care_end, token_n):   
        if care_start == 0:  # we need start_token_query
            if 'c_start_token_query' not in old_info.keys():
                start_token_states = hidden_states[:, 0:1, :]  # query is necessary
                old_info['c_start_token_query'] = self.transpose_for_scores(self.q_proj(start_token_states))  # bs, nh, 1, 64c
            start_token_query = old_info['c_start_token_query']
            hidden_states = hidden_states[:, 1:, :]
            
        bsz, partial_l, _ = hidden_states.size()  # bs, 4096
        num_blocks = self.block_h * self.block_w  # default 64 blocks
        block_size = token_n  # depends on the input sequence length
        partial_nb = partial_l // block_size
        n_rand_blocks = 3
        if 'c_blocked_key_matrix' not in old_info.keys():  # only computed once
            assert care_start == 0
            rand_attn = rand_attn.contiguous()   # bs, nh, 64, 3
            # ------ blockify
            key_layer = self.transpose_for_scores(self.k_proj(key_value_states))
            value_layer = self.transpose_for_scores(self.v_proj(key_value_states))
            old_info['c_blocked_key_matrix'] = key_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)
            old_info['c_blocked_value_matrix'] = value_layer.view(bsz, self.num_heads, num_blocks, block_size, -1)
            # ------ preparing block for randn attn
            gathered_key = self.torch_gather_b2(old_info['c_blocked_key_matrix'], rand_attn)   # bs, nh, 64x3, nt, 64c 
            old_info['c_gathered_key'] = gathered_key.view(bsz, self.num_heads, num_blocks, n_rand_blocks * block_size, -1)  #  bs, nh, nb, 3xnt tokens, 64c
            gathered_value = self.torch_gather_b2(old_info['c_blocked_value_matrix'], rand_attn)
            old_info['c_gathered_value'] = gathered_value.view(bsz, self.num_heads, num_blocks, n_rand_blocks * block_size, -1)
        blocked_key_matrix = old_info['c_blocked_key_matrix']
        blocked_value_matrix = old_info['c_blocked_value_matrix']
        gathered_key = old_info['c_gathered_key']
        gathered_value = old_info['c_gathered_value']
        
        query_layer = self.transpose_for_scores(self.q_proj(hidden_states))  # bs, nh, L, headC      by heads     partial
        blocked_query_matrix_partial = query_layer.view(bsz, self.num_heads, partial_nb, block_size, -1)  # bs, nh, nb, nt, 64c
        final_list = []
        
        if care_start == 0:
            # ------ [start] token (global)
            if 'c_start_context_layer' not in old_info.keys():  # only computed once
                start_product = self.torch_bmm_nd_transpose(start_token_query, key_layer, ndim=4)
                start_product = start_product * self.scaling
                start_attn_weights = nn.functional.softmax(start_product, dim=-1)
                old_info['c_start_context_layer'] = self.torch_bmm_nd(start_attn_weights, value_layer, ndim=4)  # bs, nh, 1, 64C    "hidden features"
            start_context_layer = old_info['c_start_context_layer']
            # ------ 1st block (not global)
            first_key_mat = torch.cat(
                [
                    blocked_key_matrix[:, :, 0],
                    blocked_key_matrix[:, :, 1],
                    gathered_key[:, :, 0],
                ],
                dim=2,
            )  # bs, nh, _, 64c    
            first_value_mat = torch.cat(
                [
                    blocked_value_matrix[:, :, 0],
                    blocked_value_matrix[:, :, 1],
                    gathered_value[:, :, 0],
                ],
                dim=2,
            )  # bs, nh, _, 64c   
            first_product = self.torch_bmm_nd_transpose(blocked_query_matrix_partial[:, :, 0], first_key_mat, ndim=4)  # bs, nh, nt, _  
            first_product = first_product * self.scaling   
            first_attn_weights = nn.functional.softmax(first_product, dim=-1)
            first_context_layer = self.torch_bmm_nd(first_attn_weights, first_value_mat, ndim=4)  # bs, nh, nt, 64c
            first_context_layer.unsqueeze_(2)  # bs, nh, 1, nt, 64C
            final_list.append(first_context_layer)
        # ------ Middle blocks (sliding attn is calculated using special trick of shifting tokens as discussed in paper)   
        middle_start_bid = max(care_start, 1)
        middle_end_bid = min(care_end, num_blocks - 1)  # [middle_start_bid, middle_end_bid)
        if middle_end_bid > middle_start_bid:  # might be False
            exp_blocked_key_matrix = torch.cat(
                [blocked_key_matrix[:, :, (middle_start_bid-1):(middle_end_bid-1)], blocked_key_matrix[:, :, middle_start_bid:middle_end_bid], blocked_key_matrix[:, :, (middle_start_bid+1):(middle_end_bid+1)]], dim=3
            )  #  bs, nh, 62, 3 * nt, 64c
            exp_blocked_value_matrix = torch.cat(
                [blocked_value_matrix[:, :, (middle_start_bid-1):(middle_end_bid-1)], blocked_value_matrix[:, :, middle_start_bid:middle_end_bid], blocked_value_matrix[:, :, (middle_start_bid+1):(middle_end_bid+1)]], dim=3,
            )
            # si_in_partial = si_in_64 - care_start
            middle_query_matrix = blocked_query_matrix_partial[:, :, (middle_start_bid-care_start):(middle_end_bid-care_start)]  #  bs, nh, 62, nt, 64c  
            # sliding attention scores
            inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)  #  bs, nh, 62, nt, 3 * nt
            inner_band_product = inner_band_product * self.scaling
            # randn attention scores
            rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, middle_start_bid:middle_end_bid], ndim=5)  # bs, nh, 62, nt, 3 * nt
            rand_band_product = rand_band_product * self.scaling
            # completing attention scores matrix for all middle blocks
            band_product = torch.cat([inner_band_product, rand_band_product], dim=-1)  # bs, nh, 62, nt, 6 * nt
            # safely doing softmax since attention matrix is completed     1 block, 3 blocks, 3 blocks, 1 block
            attn_weights = nn.functional.softmax(band_product, dim=-1)  # bs, nh, 62, nt, 6 * nt
            # contribution of sliding keys
            context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :,  : 3 * block_size], exp_blocked_value_matrix, ndim=5)
            # adding contribution of random keys
            context_layer += self.torch_bmm_nd(attn_weights[:, :, :, :, 3 * block_size :], gathered_value[:, :, middle_start_bid:middle_end_bid], ndim=5)
            final_list.append(context_layer)
        # ------ last block (not global)
        if care_end == num_blocks:
            last_key_mat = torch.cat(
                [
                    blocked_key_matrix[:, :, -2],
                    blocked_key_matrix[:, :, -1],
                    gathered_key[:, :, -1],
                ],
                dim=2,
            )  # bs, nh, _, 64c
            last_value_mat = torch.cat(
                [
                    blocked_value_matrix[:, :, -2],
                    blocked_value_matrix[:, :, -1],
                    gathered_value[:, :, -1],
                ],
                dim=2,
            )  # bs, nh, _, 64c
            last_product = self.torch_bmm_nd_transpose(blocked_query_matrix_partial[:, :, -1], last_key_mat, ndim=4)  # bs, nh, nt, _
            last_product = last_product * self.scaling
            last_attn_weights = nn.functional.softmax(last_product, dim=-1)
            last_context_layer = self.torch_bmm_nd(last_attn_weights, last_value_mat, ndim=4)  # bs, nh, nt, 64c
            last_context_layer.unsqueeze_(2)  # bs, nh, 1, nt, 64C     
            final_list.append(last_context_layer)
        # ------ combining representations of all blocks
        context_layer = torch.cat(final_list, dim=2)  # bs, nh, nb, nt, 64C
        context_layer = context_layer.view((bsz, self.num_heads, partial_l, -1))  # bs, nh, L, 64C
        if care_start == 0:
            context_layer = torch.cat([start_context_layer, context_layer], dim=2)  # bs, nh, L + 1, 64C
            context_layer = torch.transpose(context_layer, 1, 2)  # bs, L + 1, nh, 64C
            context_layer = context_layer.contiguous().view(bsz, partial_l + 1, -1)  # bs, L + 1, C
        else:
            context_layer = torch.transpose(context_layer, 1, 2)  # bs, L, nh, 64C
            context_layer = context_layer.contiguous().view(bsz, partial_l, -1)  # bs, L, C
        return context_layer, old_info
    
    @torch.no_grad()   
    def forward_fast(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        rand_attn=None,
        old_info=None,  # a dict storing past states
        care_start=0, 
        care_end=1,  # [start, end)  block indices
        token_n=-1,  # number of tokens in one block
        do_record=False,
    ):
        """Input shape: Batch x Time x Channel"""
        assert self.is_decoder
        is_cross_attention = key_value_states is not None
        if is_cross_attention:
            attn_output, old_info = self.forward_sparse_decoder_cross_SGA_fast(hidden_states, key_value_states, rand_attn, old_info, care_start, care_end, token_n)
        else:
            attn_output, old_info = self.forward_sparse_decoder_causal_fast(hidden_states, rand_attn, old_info, care_start, care_end, token_n, do_record)
        attn_output = self.out_proj(attn_output)
        return attn_output, old_info
            
    @torch.no_grad()   
    def forward_fast256(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        old_info=None,  # a dict storing past states
        care_start=0, 
        care_end=1,  # [start, end)  block indices
    ):
        """Input shape: Batch x Time x Channel"""
        assert self.is_decoder
        is_cross_attention = key_value_states is not None
        bsz, partial_l, _ = hidden_states.size()
        query_states_partial = self.q_proj(hidden_states) * self.scaling
        query_states_partial = self._shape(query_states_partial, -1, bsz)
        
        if is_cross_attention:
            if 'c_key_states' not in old_info.keys():  # only computed once
                assert care_start == 0
                old_info['c_key_states'] = self._shape(self.k_proj(key_value_states), -1, bsz)
                old_info['c_value_states'] = self._shape(self.v_proj(key_value_states), -1, bsz)
            key_states = old_info['c_key_states']
            value_states = old_info['c_value_states']
        else:
            key_states_partial = self._shape(self.k_proj(hidden_states), -1, bsz)  # BS, nh, L, c
            value_states_partial = self._shape(self.v_proj(hidden_states), -1, bsz)
            if 's_key_states' in old_info.keys():
                assert care_start != 0
                assert old_info['s_key_states'].shape[2] == care_start  # sanity check
                key_states = torch.cat((old_info['s_key_states'], key_states_partial), dim=2)
                value_states = torch.cat((old_info['s_value_states'], value_states_partial), dim=2)
            else:
                assert care_start == 0
                key_states = key_states_partial
                value_states = value_states_partial
            old_info['s_key_states'] = key_states
            old_info['s_value_states'] = value_states
        #--- compute attention
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states_partial = query_states_partial.view(*proj_shape)  # bs*nh, L, c
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states_partial, key_states.transpose(1, 2))  # bs*nh, 1, L
        #
        if not is_cross_attention and (care_end - care_start > 1):
            attn_mask = self.generate_causal_mask(attn_weights.unsqueeze(1))  # 1, 1, x, L
            attn_weights += attn_mask.squeeze(1)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, value_states)  # bs*nh, 1, c
        
        attn_output = attn_output.view(bsz, self.num_heads, partial_l, self.head_dim)
        attn_output = attn_output.transpose(1, 2)  # bs, 1, nh, nc
        attn_output = attn_output.reshape(bsz, partial_l, self.embed_dim)
        
        attn_output = self.out_proj(attn_output)
        return attn_output, old_info
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        is_sparse: bool = False,
        rand_attn=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        
        if is_sparse:
            if self.is_decoder:  # decoder
                is_cross_attention = key_value_states is not None
                if is_cross_attention:
                    attn_output, past_key_value = self.forward_sparse_decoder_cross_SGA(hidden_states, key_value_states, rand_attn)     
                else:
                    attn_output, past_key_value = self.forward_sparse_decoder_causal(hidden_states, rand_attn)
            else:
                attn_output = self.forward_sparse_encoder_SGA(hidden_states, rand_attn)
                        
            attn_output = self.out_proj(attn_output)
            return attn_output, None, past_key_value  # ? --> past_key_value
        
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        embed_dim = self.embed_dim

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)  # bs, nh, L, 64c
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)  # bs, nh, L, 64c

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)  # bs*nh, L, 64c
        key_states = key_states.view(*proj_shape)  # bs*nh, L, 64c
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nh, L, L

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)  # bs, nh, L, L   after softmax
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)  # attention dropout

        attn_output = torch.bmm(attn_probs, value_states)  # bs*nh, L, L  *  bs*nh, L, 64c     --->   bs*nh, L, 64c

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PosCNN(nn.Module):  # PEG    https://arxiv.org/abs/2102.10882
    def __init__(self, embed_dim=1024, ks=3):
        super(PosCNN, self).__init__()
        pad_num = (ks - 1) // 2
        self.proj = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, ks, 1, pad_num, bias=True, groups=embed_dim), )

    def forward(self, x, block_h=16, block_w=4):  # input/output is strip order
        B, L, C = x.shape
        H = int(math.sqrt(L))  # 16, 32, 64
        token_h = H // block_h  # 1
        token_w = H // block_w  # 4
        
        feat_token = x
        #-------   
        cnn_feat = feat_token.view(B, block_h * block_w, token_h * token_w, C).view(B * block_h * block_w, token_h, token_w, C).permute(0, 3, 1, 2).contiguous()
        ans = self.proj(cnn_feat)
        ans = ans.permute(0, 2, 3, 1).contiguous().view(B * block_h * block_w, token_h * token_w, C).view(B, block_h * block_w, token_h * token_w, C).view(B, L, C)
        return ans
    
    
class ASSETEncoderLayer(nn.Module):  # based on BartEncoderLayer    Done for 256-resolution
    def __init__(self, config: BartConfig, double_ic: bool = False, use_PEG: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model
        self.double_ic = double_ic
        self.use_PEG = use_PEG
        if self.use_PEG:
            self.PEG_net = PosCNN(self.embed_dim, ks=config.PEG_ks)
        self.self_attn = ASSETAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            double_ic=double_ic,
            block_h=config.block_h,
            block_w=config.block_w)
        if self.double_ic:
            self.residual_ln = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        is_sparse: bool = False,
        e_self_rand_attn=None,
    ):
        if self.use_PEG:  # 3x3 depth-wise convolution
            new_pe = self.PEG_net(hidden_states, block_h=self.config.block_h, block_w=self.config.block_w)
            hidden_states += new_pe
            
        residual = hidden_states
            
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            is_sparse=is_sparse,
            rand_attn=e_self_rand_attn,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.double_ic:
            hidden_states = self.residual_ln(residual) + hidden_states
        else:
            hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)  # tuple

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ASSETEncoder(BartPretrainedModel):  # responsible for positional embeddings      Done for 256-resolution
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None,
                 embed_positions: Optional[nn.Embedding] = None, embed_blocks: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = embed_tokens  # shared image embedding
        self.embed_cond = nn.Embedding(config.vocab_size, embed_dim)  # cond embedding    no padding

        self.embed_positions = embed_positions  # shared pos embedding
        self.embed_blocks = embed_blocks
        
        self.PEG_list = config.PEG_list
        if len(self.PEG_list) == 0:
            use_PEG_list = [False for _ in range(config.encoder_layers)]
        else:
            use_PEG_list = []
            for layer_idx in range(config.encoder_layers):
                to_check_str = 'E_%d' % layer_idx
                if to_check_str in self.PEG_list or 'E_all' in self.PEG_list:
                    use_PEG_list.append(True)
                else:
                    use_PEG_list.append(False)
                  
        self.layers = nn.ModuleList([ASSETEncoderLayer(config, use_PEG=use_PEG_list[layer_idx]) for layer_idx in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
    
    def embed_positions_longer(self, base_L, target_L, bs):  # default bilinear interpolation
        low_res_pos = torch.arange(base_L).unsqueeze(0).expand(bs, -1).to(self.device)
        low_hw = int(math.sqrt(base_L))
        high_hw = int(math.sqrt(target_L))
        low_res_pos_grid = low_res_pos.reshape(bs, low_hw, low_hw)
        low_res_pos_grid_embed_pos = self.embed_positions(low_res_pos_grid) * self.embed_scale
        num_c = low_res_pos_grid_embed_pos.shape[-1]
        high_res_pos_embed_pos = F.interpolate(low_res_pos_grid_embed_pos.permute(0, 3, 1, 2), size=high_hw, mode='bilinear').permute(0, 2, 3, 1).reshape(bs, -1, num_c)
        
        high_res_pos_embed_condpos = None
        return high_res_pos_embed_pos, high_res_pos_embed_condpos
    
    def forward(
        self,
        input_ids=None,
        input_pos=None,  # Make it None
        cond_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_sparse=False,
        e_self_rand_attn=None,
    ):
        output_attentions = output_attentions
        output_hidden_states = (False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # default True

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()  #  2, 256
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            assert 0
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale  # bs, L, 1024
        cond_embeds = self.embed_cond(cond_ids) * self.embed_scale
        
        assert input_pos is None
        if len(self.PEG_list) == 0:  # original PE
            if input_shape[1] > self.config.max_position_embeddings:  # bilinear interpolation of PE  TODO bicubic interpolation
                embed_pos, embed_condposv = self.embed_positions_longer(self.config.max_position_embeddings, input_shape[1], input_shape[0])
            else:  # the original length
                input_pos = torch.arange(input_shape[1]).unsqueeze(0).expand(input_shape[0], -1).to(self.device)
                embed_pos = self.embed_positions(input_pos) * self.embed_scale
                embed_condposv = None
        else:  # compute block embedding
            block_ids = generate_block_ids(inputs_embeds, self.config.block_h, self.config.block_w)  # output normal order
            embed_pos = self.embed_blocks(block_ids) * self.embed_scale
        
        hidden_states = inputs_embeds + embed_pos + cond_embeds
        
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # change scan order  Input BS, L, C
        hidden_states = normal_to_strip_order(hidden_states, block_h=self.config.block_h, block_w=self.config.block_w)
            
        
        encoder_states = () if output_hidden_states else None  # default None
        all_attentions = () if output_attentions else None  # default None  a empty tuple

        # check if head_mask has a correct number of layers specified if desired

        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        is_sparse=is_sparse,
                        e_self_rand_attn=(e_self_rand_attn[idx] if e_self_rand_attn is not None else None)
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class ASSETDecoderLayer(nn.Module):  # Done for 256-resolution
    def __init__(self, config: BartConfig, use_PEG: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model

        self.use_PEG = use_PEG
        if self.use_PEG:
            self.PEG_net = PosCNN(self.embed_dim, ks=config.PEG_ks)
            
        self.self_attn = ASSETAttention(  # causal self-attention
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            block_h=config.block_h,
            block_w=config.block_w,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = ASSETAttention(  # cross attention
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            block_h=config.block_h,
            block_w=config.block_w,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        is_sparse: Optional[bool] = False,
        d_causal_rand_attn=None,
        d_cross_rand_attn=None,
    ):
        if self.use_PEG:  # 3x3 depth-wise convolution
            new_pe = self.PEG_net(encoder_hidden_states, block_h=self.config.block_h, block_w=self.config.block_w)
            hidden_states[:, 1:, :] = hidden_states[:, 1:, :] + new_pe
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple       causal self-attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            is_sparse=is_sparse,
            rand_attn=d_causal_rand_attn,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                is_sparse=is_sparse,
                rand_attn=d_cross_rand_attn,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    @torch.no_grad()
    def forward_fast256(
        self,
        hidden_states: torch.Tensor,  # not full sequence
        encoder_hidden_states: Optional[torch.Tensor] = None,
        old_info=None,  # a dict storing past states
        care_start=0, 
        care_end=1,  # [start, end)  block indices
    ):
        if self.use_PEG:  # 3x3 depth-wise convolution
            if 'new_pe' not in old_info.keys():
                old_info['new_pe'] = self.PEG_net(encoder_hidden_states, block_h=self.config.block_h, block_w=self.config.block_w)  # L not L + 1
            if care_start == 0 and care_end == 1:
                pass
            elif care_start == 0 and care_end > 1:
                hidden_states[:, 1:, :] = hidden_states[:, 1:, :] + old_info['new_pe'][:, 0:(care_end - 1), :]
            else:
                hidden_states = hidden_states + old_info['new_pe'][:, (care_start - 1):(care_end - 1), :]
            
        residual = hidden_states
        # Self Attention
        # add present self-attn cache to positions 1,2 of present_key_value tuple       causal self-attention
        hidden_states, old_info = self.self_attn.forward_fast256(
            hidden_states=hidden_states,
            old_info=old_info, care_start=care_start, care_end=care_end,
        )  # Implementation is done!!!!
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, old_info = self.encoder_attn.forward_fast256(
                hidden_states=hidden_states, key_value_states=encoder_hidden_states,
                old_info=old_info, care_start=care_start, care_end=care_end,
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, old_info
    
    @torch.no_grad()
    def forward_fast(
        self,
        hidden_states: torch.Tensor,  # not full sequence
        encoder_hidden_states: Optional[torch.Tensor] = None,
        is_sparse: Optional[bool] = False,
        d_causal_rand_attn=None,
        d_cross_rand_attn=None,
        old_info=None,  # a dict storing past states
        care_start=0, 
        care_end=1,  # [start, end)  block indices
        token_n=-1,  # number of tokens in one block
        do_record=False,
    ):
        assert is_sparse
        if self.use_PEG:  # 3x3 depth-wise convolution
            if 'new_pe' not in old_info.keys():
                old_info['new_pe'] = self.PEG_net(encoder_hidden_states, block_h=self.config.block_h, block_w=self.config.block_w)  # L not L + 1
            if care_start == 0:
                hidden_states[:, 1:, :] = hidden_states[:, 1:, :] + old_info['new_pe'][:, care_start*token_n : care_end*token_n]
            else:
                hidden_states = hidden_states + old_info['new_pe'][:, care_start*token_n : care_end*token_n]
            
        residual = hidden_states

        # Self Attention
        # add present self-attn cache to positions 1,2 of present_key_value tuple       causal self-attention
        hidden_states, old_info = self.self_attn.forward_fast(
            hidden_states=hidden_states, rand_attn=d_causal_rand_attn,
            old_info=old_info, care_start=care_start, care_end=care_end, token_n=token_n, do_record=do_record,
        )  # Implementation is done!!!!
        
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, old_info = self.encoder_attn.forward_fast(
                hidden_states=hidden_states, key_value_states=encoder_hidden_states, rand_attn=d_cross_rand_attn,
                old_info=old_info, care_start=care_start, care_end=care_end, token_n=token_n, do_record=do_record,
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, old_info
    

class ASSETDecoder(BartPretrainedModel):  # Done for 256-resolution
    """
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None,
                 embed_positions: Optional[nn.Embedding] = None, embed_blocks: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = embed_tokens

        self.embed_positions = embed_positions

        self.embed_blocks = embed_blocks
        
        self.PEG_list = config.PEG_list
        if len(self.PEG_list) == 0:
            use_PEG_list = [False for _ in range(config.decoder_layers)]
        else:
            use_PEG_list = []
            for layer_idx in range(config.decoder_layers):
                to_check_str = 'D_%d' % layer_idx
                if to_check_str in self.PEG_list or 'D_all' in self.PEG_list:
                    use_PEG_list.append(True)
                else:
                    use_PEG_list.append(False)
            
            
        self.layers = nn.ModuleList([ASSETDecoderLayer(config, use_PEG=use_PEG_list[layer_idx]) for layer_idx in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:  # sequence length >= 2
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)  # bs, 1, L, L   causal part 0.0   other parts -inf

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def embed_positions_longer(self, base_L, target_L, bs):  # default bilinear interpolation
        low_res_pos = torch.arange(base_L).unsqueeze(0).expand(bs, -1).to(self.device)
        low_hw = int(math.sqrt(base_L))
        high_hw = int(math.sqrt(target_L))
        low_res_pos_grid = low_res_pos.reshape(bs, low_hw, low_hw)
        low_res_pos_grid_embed_pos = self.embed_positions(low_res_pos_grid) * self.embed_scale
        num_c = low_res_pos_grid_embed_pos.shape[-1]
        high_res_pos_embed_pos = F.interpolate(low_res_pos_grid_embed_pos.permute(0, 3, 1, 2), size=high_hw, mode='bilinear').permute(0, 2, 3, 1).reshape(bs, -1, num_c)
        
        start_tokens = (torch.arange(1).unsqueeze(0).expand(bs, -1).to(self.device) + 1) * self.max_target_positions
        start_tokens_embed = self.embed_positions(start_tokens) * self.embed_scale
        return torch.cat((start_tokens_embed, high_res_pos_embed_pos), dim=1)
    
    def forward(
        self,
        input_ids=None,  # Not None
        input_pos=None,
        attention_mask=None,
        encoder_hidden_states=None,  # Not None
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_sparse=False,
        d_causal_rand_attn=None,
        d_cross_rand_attn=None,
    ):
        r"""
        Args:
        """
        output_attentions = output_attentions
        output_hidden_states = (False)
        use_cache = use_cache if use_cache is not None else self.config.use_cache  # default True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # default True

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()  # bs, 257
            assert input_shape[1] % 2 == 1
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0  # default 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )  # a causal mask    bs, 1, L, L   causal part 0.0   other parts -inf (remove attention)

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        assert input_pos is None
        if len(self.PEG_list) == 0:  # original PE
            if input_shape[1] - 1 > self.config.max_position_embeddings:  # bilinear interpolation of PE
                positions = self.embed_positions_longer(self.config.max_position_embeddings, input_shape[1] - 1, input_shape[0])
            else:  # the original length    the first one is [start] token      be careful here
                input_pos = torch.arange(input_shape[1]).unsqueeze(0).expand(input_shape[0], -1).to(self.device)  #  [0, ..., 256]
                input_pos = input_pos - 1  # -1, 0, ..., 255
                input_pos[:, 0] = self.max_target_positions  # -->  256,   0, ..., 255
                positions = self.embed_positions(input_pos) * self.embed_scale
        else:
            block_ids = generate_block_ids(inputs_embeds[:, 1:, :], self.config.block_h, self.config.block_w)  # normal order
            block_positions = self.embed_blocks(block_ids) * self.embed_scale
            start_ids = torch.ones(inputs_embeds.shape[0], 1, dtype=torch.int64).to(self.device) * self.max_target_positions
            start_positions = self.embed_positions(start_ids) * self.embed_scale
            positions = torch.cat((start_positions, block_positions), dim=1)

            
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # change scan order  Input BS, L, C
        hidden_states = torch.cat((hidden_states[:, 0:1, :], normal_to_strip_order(hidden_states[:, 1:, :], block_h=self.config.block_h, block_w=self.config.block_w)), dim=1)
            
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None  # default None
        all_self_attns = () if output_attentions else None  # default None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None  # default None
        next_decoder_cache = () if use_cache else None  # default ()

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None  # default None and always None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,  # Not None
                    encoder_hidden_states=encoder_hidden_states,  # Not None
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    is_sparse=is_sparse,
                    d_causal_rand_attn=(d_causal_rand_attn[idx] if d_causal_rand_attn is not None else None),
                    d_cross_rand_attn=(d_cross_rand_attn[idx] if d_cross_rand_attn is not None else None),
                )  # self-attention is the same as the self-attention in the encoder
            hidden_states = layer_outputs[0]  # layer_outputs[1] or [3] is key_values of self-attention and cross-attention   self then cross

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)  # length-1 then length-4

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    @torch.no_grad()
    def get_pe_fast(self, encoder_feats):
        input_L = encoder_feats.shape[1]  # 1024, 4096
        BS = encoder_feats.shape[0]
        
        if len(self.PEG_list) == 0:  # original PE
            if input_L > self.config.max_position_embeddings:  # bilinear interpolation of PE
                positions = self.embed_positions_longer(self.config.max_position_embeddings, input_L, BS)
            else:  # the original length    the first one is [start] token      be careful here
                input_pos = torch.arange(input_L + 1).unsqueeze(0).expand(BS, -1).to(self.device)  #  [0, ..., 256]
                input_pos = input_pos - 1  # -1, 0, ..., 255
                input_pos[:, 0] = self.max_target_positions  # -->  256,   0, ..., 255
                positions = self.embed_positions(input_pos) * self.embed_scale
        else:
            block_ids = generate_block_ids(encoder_feats, self.config.block_h, self.config.block_w)  # normal order
            block_positions = self.embed_blocks(block_ids) * self.embed_scale
            start_ids = torch.ones(BS, 1, dtype=torch.int64).to(self.device) * self.max_target_positions
            start_positions = self.embed_positions(start_ids) * self.embed_scale
            positions = torch.cat((start_positions, block_positions), dim=1)
        return positions
    
    @torch.no_grad()
    def forward_fast256(
        self,
        input_ids=None,  # Not None  1 + all ids
        encoder_hidden_states=None,  # Not None
        decoder_positions=None,
        previous_info=None,  # a dict storing past states
        care_start=0, 
        care_end=1,  # [start, end)  block indices
    ):  # fast decoding   everything is in z_order    this function only cares about feature transformation, it doesn't care about masking/sampling

        # retrieve input_ids and inputs_embeds
        input_shape = input_ids.size()  # bs, 257
        assert input_shape[1] % 2 == 1
        #  care_start :care_end

        inputs_embeds = self.embed_tokens(input_ids[:, care_start:care_end]) * self.embed_scale
        hidden_states = inputs_embeds + decoder_positions[:, care_start:care_end]

        hidden_states = self.layernorm_embedding(hidden_states)  # not full sequence,  just some blocks
        for idx, decoder_layer in enumerate(self.layers):  # loop over decoder layers
            if idx in previous_info.keys():
                old_info = previous_info[idx]
            else:
                old_info = {}
            hidden_states, old_info = decoder_layer.forward_fast256(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,  # Not None
                    old_info=old_info, care_start=care_start, care_end=care_end,
            )  # self-attention is the same as the self-attention in the encoder
            previous_info[idx] = old_info
        
        return hidden_states, previous_info # input to lm_head
    
    @torch.no_grad()
    def forward_fast(
        self,
        input_ids=None,  # Not None  1 + all ids
        encoder_hidden_states=None,  # Not None
        decoder_positions=None,
        is_sparse=False,
        d_causal_rand_attn=None,
        d_cross_rand_attn=None,
        previous_info=None,  # a dict storing past states
        care_start=0, 
        care_end=1,  # [start, end)  block indices
        token_n=-1,  # number of tokens in one block
        do_record=False,
    ):  # fast decoding   everything is in z_order    this function only cares about feature transformation, it doesn't care about masking/sampling

        # retrieve input_ids and inputs_embeds
        input_shape = input_ids.size()  # bs, 257
        assert input_shape[1] % 2 == 1
        #  care_start * token_n  + 1 :   care_end   * token_n   + 1
        if care_start == 0:
            inputs_embeds = self.embed_tokens(input_ids[:, :(care_end * token_n + 1)]) * self.embed_scale
            hidden_states = inputs_embeds + decoder_positions[:, :(care_end * token_n + 1)]
        else:
            inputs_embeds = self.embed_tokens(input_ids[:, (care_start * token_n + 1):(care_end * token_n + 1)]) * self.embed_scale
            hidden_states = inputs_embeds + decoder_positions[:, (care_start * token_n + 1):(care_end * token_n + 1)]
        hidden_states = self.layernorm_embedding(hidden_states)  # not full sequence,  just some blocks

        for idx, decoder_layer in enumerate(self.layers):  # loop over decoder layers
            if idx in previous_info.keys():
                old_info = previous_info[idx]
            else:
                old_info = {}
            hidden_states, old_info = decoder_layer.forward_fast(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,  # Not None
                    is_sparse=is_sparse,
                    d_causal_rand_attn=(d_causal_rand_attn[idx] if d_causal_rand_attn is not None else None),
                    d_cross_rand_attn=(d_cross_rand_attn[idx] if d_cross_rand_attn is not None else None),
                    old_info=old_info, care_start=care_start, care_end=care_end, token_n=token_n, do_record=do_record,
            )  # self-attention is the same as the self-attention in the encoder
            previous_info[idx] = old_info

        return hidden_states, previous_info # input to lm_head
        

class ASSET(BartPretrainedModel):
    def __init__(self, vocab_size=1024, d_model=1024, max_position_embeddings=256, dropout=0.1, attention_dropout=0.0,
                 encoder_layers=12, decoder_layers=12, encoder_attention_heads=16, decoder_attention_heads=16,
                 encoder_ffn_dim=4096, decoder_ffn_dim=4096, 
                 block_h=8, block_w=8, PEG_list=[], PEG_ks=3):
        config = ASSETConfig(block_h=block_h,
                               block_w=block_w, PEG_list=PEG_list, PEG_ks=PEG_ks,
                               vocab_size=vocab_size, d_model=d_model, max_position_embeddings=max_position_embeddings, 
                               encoder_layers=encoder_layers, dropout=dropout, attention_dropout=attention_dropout,
                               encoder_attention_heads=encoder_attention_heads, 
                               decoder_layers=decoder_layers, decoder_attention_heads=decoder_attention_heads,
                               encoder_ffn_dim=encoder_ffn_dim, decoder_ffn_dim=decoder_ffn_dim)
        super().__init__(config)

        vocab_size = config.vocab_size
        self.shared = nn.Embedding(vocab_size + 2, config.d_model)  # words embedding plus [mask] [start]      no padding
        self.shared_pos = nn.Embedding(config.max_position_embeddings + 1, config.d_model)  # pos embedding  plus pos for [start]   for "normal" order
        self.shared_block = nn.Embedding(config.block_h * config.block_w, config.d_model)  # block embedding
        
        self.encoder = ASSETEncoder(config, self.shared, self.shared_pos, self.shared_block)
        self.decoder = ASSETDecoder(config, self.shared, self.shared_pos, self.shared_block)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        self.init_weights()
        

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def forward(
        self,
        input_ids=None,  # Not None
        input_pos=None,
        cond_ids=None,  # Not None
        attention_mask=None,
        decoder_input_ids=None,  # Not None
        decoder_input_pos=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=None,
        is_sparse=False,
        e_self_rand_attn=None,
        d_causal_rand_attn=None,
        d_cross_rand_attn=None,
    ):

        output_attentions = output_attentions
        output_hidden_states = (False)
        use_cache = use_cache if use_cache is not None else self.config.use_cache  # default True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # default True

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            input_pos=input_pos,
            cond_ids=cond_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_sparse=is_sparse,
            e_self_rand_attn=e_self_rand_attn,
        )  # to_block_order inside

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            input_pos=decoder_input_pos,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_sparse=is_sparse,
            d_causal_rand_attn=d_causal_rand_attn,
            d_cross_rand_attn=d_cross_rand_attn,
        )  # to_block_order inside
        logits = self.lm_head(decoder_outputs[0])
        
        # to_normal_order
        logits = torch.cat((logits[:, 0:1, :], strip_to_normal_order(logits[:, 1:, :], block_h=self.config.block_h, block_w=self.config.block_w)), dim=1)
            
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=logits,  # Very important
            past_key_values=decoder_outputs.past_key_values,  # might be used during testing
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,  # important
            cross_attentions=decoder_outputs.cross_attentions,  # important
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,  # important
        )

