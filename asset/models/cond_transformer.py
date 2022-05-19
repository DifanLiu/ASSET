import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
from skimage import measure
from main import instantiate_from_config
from asset.modules.util import SOSProvider
from asset.modules.transformer.asset import normal_to_strip_order, strip_to_normal_order
import numpy as np


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 VQGAN_freq=-1,  # -1 means DO NOT split
                 max_unknown=-1,
                 d_size=[256],
                 is_SGA=False,
                 guiding_ckpt_path=None,
                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "asset.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.is_SGA = is_SGA
        if self.is_SGA:
            self.guiding_transformer = instantiate_from_config(config=transformer_config)
            assert guiding_ckpt_path is not None
            self.load_guiding_transformer(guiding_ckpt_path)
            
        self.downsample_cond_size = downsample_cond_size  # -1
        self.pkeep = pkeep  # 1.0
        self.VQGAN_freq = VQGAN_freq
        self.max_unknown = max_unknown
        self.d_size = d_size
        #---- 

        self.block_h = transformer_config['params']['block_h']
        self.block_w = transformer_config['params']['block_w']
        self.el = transformer_config['params']['encoder_layers']  # number of transformer encoder layers
        self.dl = transformer_config['params']['decoder_layers']  # number of transformer decoder layers
        self.NN = 3
        self.NK = 3
        if self.is_SGA:
            self.rand_full_mask = self.generate_rand_full_mask()
            self.rand_causal_mask = self.generate_rand_causal_mask()

        self.num_downsampling = len(first_stage_config['params']['ddconfig']['ch_mult']) - 1
        self.mask_token_id = transformer_config['params']['vocab_size']
        self.start_token_id = transformer_config['params']['vocab_size'] + 1
        self.start_pos_id = transformer_config['params']['max_position_embeddings']

    def generate_rand_causal_mask(self, ):  # in causal attention, prevent attending to subsequent blocks  /  also ignore 3 neighboring blocks
        num_blocks = self.block_h * self.block_w
        NBR = num_blocks - (self.NN + self.NK) + 1  # 59
        
        ans = torch.zeros(num_blocks, num_blocks, dtype=torch.float32)
        
        for bid in range((self.NN + self.NK) + 1, num_blocks):
            ans[bid, (bid-(self.NN - 1)):] = -10000.0 
        return ans
        
    def generate_rand_full_mask(self, ):  # ignore 3 neighboring blocks when selecting top-K blocks
        num_blocks = self.block_h * self.block_w  # default 64
        
        ans = torch.zeros(num_blocks, num_blocks, dtype=torch.float32)

        ans[0, 0:2] = -10000.0
        ans[-1, -2:] = -10000.0
        for bid in range(1, num_blocks - 1):
            ans[bid, (bid-1):(bid+2)] = -10000.0              
        return ans
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd.keys():
            sd = sd["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print("missing_keys", len(missing_keys))
        print("unexpected_keys", len(unexpected_keys))
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model 

    def top_k_logits(self, logits, k):  # bs, 1024
        if isinstance(k, int):  # top-k
            v, ix = torch.topk(logits, k)
            out = logits.clone()
            out[out < v[..., [-1]]] = -float('Inf') 
        else:  # top-p     p=1 means all effective      at least one token is effective
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # larger --> smaller
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # bs 1024     0--->1
            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > k
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)  # boolean
            #print(torch.sum(indices_to_remove) / indices_to_remove.shape[0])
            out = logits.clone()
            out = out.masked_fill(indices_to_remove, -float('Inf'))
        return out

    @torch.no_grad()
    def encode_to_z(self, x, mask_tensor=None):
        quant_z, _, info = self.first_stage_model.encode(x, mask_in=mask_tensor)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, info = self.cond_stage_model.encode(c)
        #if len(indices.shape) > 2:
        #    indices = indices.view(c.shape[0], -1)
        indices = info[2].view(quant_c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def training_step(self, batch, batch_idx):
        if not self.is_SGA:  # guiding transformer
            mask_tensor = self.get_input('mask', batch)  # bs, 1, 256, 256, on cuda   
            loss = self.shared_step_guiding(batch, batch_idx, mask_tensor=mask_tensor)
        else:  # SGA transformer
            loss = self.shared_step_SGA(batch, batch_idx)
            
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.is_SGA:
            mask_tensor = self.get_input('mask', batch)  # bs, 1, 256, 256, on cuda
            loss = self.shared_step_guiding(batch, batch_idx, mask_tensor=mask_tensor)
        else:
            loss = self.shared_step_SGA(batch, batch_idx)

        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def get_rough_attn_map(self, batch, batch_idx, z_indices=None, c_indices=None, resized_mask_tensor=None):
        # ------ get guidance from the guiding transformer
        if batch is not None:  # used during training
            low_size_idx = 1  # low-res images
            x = self.get_input('image_%d' % low_size_idx, batch)
            c = self.get_input('segmentation_%d' % low_size_idx, batch)
            xh = x.shape[2]
            xw = x.shape[3]
            mask_tensor = self.get_input('mask_%d' % low_size_idx, batch)  # bs, 1, 256, 256, on cuda
            resized_mask_tensor = F.interpolate(mask_tensor, size=(xh // (2 ** self.num_downsampling), xw // (2 ** self.num_downsampling)))
            # one step to produce the logits
            _, z_indices = self.encode_to_z(x, mask_tensor=mask_tensor)  # bs, 256
            _, c_indices = self.encode_to_c(c)  # bs, 256

        resized_mask = resized_mask_tensor[:, 0, :, :].cpu().numpy()  # bs, 16, 16
        single_T = z_indices.shape[1] # 256

        # mask input image tokens
        a_indices = z_indices.clone()
        for bid in range(resized_mask.shape[0]):
            flatten_np = resized_mask[bid].flatten()
            indices_unknown = np.nonzero(flatten_np)[0]  # positions in z_indices
            a_indices[bid, indices_unknown] = self.mask_token_id

        # decoder_input_ids
        decoder_input_ids = torch.arange(single_T + 1).unsqueeze(0).expand(resized_mask.shape[0], -1).to(a_indices.device)  # B, 257
        decoder_input_ids[:, 1:] = z_indices.clone()
        decoder_input_ids[:, 0] = self.start_token_id  # [start]   
        temp = self.guiding_transformer(input_ids=a_indices, cond_ids=c_indices, decoder_input_ids=decoder_input_ids, output_attentions=True)  # No Sparse attention
            
        e_self_weights = temp.encoder_attentions
        e_self_rand_attn = self.get_topK_full(e_self_weights)  # nl, bs, nh, 62, 3
        
        d_causal_weights = temp.decoder_attentions  # 257 x 257
        d_causal_rand_attn = self.get_topK_causal(d_causal_weights)
        
        d_cross_weights = temp.cross_attentions  # 257x256
        d_cross_rand_attn = self.get_topK_full(d_cross_weights)  # nl, bs, nh, 62, 3

        return e_self_rand_attn, d_causal_rand_attn, d_cross_rand_attn
    
    @torch.no_grad()
    def get_topK_full(self, attn_tuple):
        num_blocks = self.block_h * self.block_w
        num_t = 256 // num_blocks
        bs = attn_tuple[0].shape[0]
        nh = attn_tuple[0].shape[1]
        nl = len(attn_tuple)
        all_attn_weights = torch.stack(attn_tuple, dim=0)  # nl, bs, nh, 256, 256
        if all_attn_weights.shape[3] == all_attn_weights.shape[4] + 1:  # decoder cross attention
            all_attn_weights = all_attn_weights[:, :, :, 1:, :]
            
        all_attn_weights = all_attn_weights.view(nl * bs * nh, 256, 256).unsqueeze(1)  # nl*bs*nh, 1, 256, 256      [0, 1]
        all_attn_weights_block = F.avg_pool2d(all_attn_weights, num_t, stride=num_t)  # nl*bs*nh, 1, 64, 64  [0, 1]
        all_attn_weights_block += self.rand_full_mask.unsqueeze(0).unsqueeze(0).to(self.device)
        _, indices = torch.topk(all_attn_weights_block.squeeze(1), 3, dim=-1)  # # nl*bs*nh, 64, 3
        
        return indices.view(nl, bs, nh, num_blocks, 3)
            
    @torch.no_grad()
    def get_topK_causal(self, attn_tuple):
        num_blocks = self.block_h * self.block_w
        num_t = 256 // num_blocks
        NBR = num_blocks - (self.NN + self.NK) + 1  # 59
        
        bs = attn_tuple[0].shape[0]
        nh = attn_tuple[0].shape[1]
        nl = len(attn_tuple)
        all_attn_weights = torch.stack(attn_tuple, dim=0)  # nl, bs, nh, 257, 257
        all_attn_weights = all_attn_weights[:, :, :, 1:, 1:]
            
        all_attn_weights = all_attn_weights.view(nl * bs * nh, 256, 256).unsqueeze(1)  # nl*bs*nh, 1, 256, 256      [0, 1]
        all_attn_weights_block = F.avg_pool2d(all_attn_weights, num_t, stride=num_t)  # nl*bs*nh, 1, 64, 64  [0, 1]
        all_attn_weights_block += self.rand_causal_mask.unsqueeze(0).unsqueeze(0).to(self.device)
        _, indices = torch.topk(all_attn_weights_block.squeeze(1), 3, dim=-1)  # # nl*bs*nh, 64, 3
        indices = indices[:, (-NBR):, :]
        return indices.view(nl, bs, nh, NBR, 3)
            
    def shared_step_SGA(self, batch, batch_idx):
        e_self_rand_attn, d_causal_rand_attn, d_cross_rand_attn = self.get_rough_attn_map(batch, batch_idx)  # no grad, guidance from the guiding transformer
        
        high_size_idx = 0  # index of the high-resolution data
        
        x = self.get_input('image_%d' % high_size_idx, batch)
        c = self.get_input('segmentation_%d' % high_size_idx, batch)
        xh = x.shape[2]
        xw = x.shape[3]
        mask_tensor = self.get_input('mask_%d' % high_size_idx, batch)  # bs, 1, 256, 256, on cuda
        
        resized_mask_tensor = F.interpolate(mask_tensor, size=(xh // (2 ** self.num_downsampling), xw // (2 ** self.num_downsampling)))
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x, mask_tensor=mask_tensor)  # bs, 256
        _, c_indices = self.encode_to_c(c)  # bs, 256
        
        resized_mask = resized_mask_tensor[:, 0, :, :].cpu().numpy()  # bs, 16, 16
        single_T = z_indices.shape[1] # 256
        
        # mask input image tokens
        a_indices = z_indices.clone()
        for bid in range(resized_mask.shape[0]):
            flatten_np = resized_mask[bid].flatten()
            indices_unknown = np.nonzero(flatten_np)[0]  # positions in z_indices
            a_indices[bid, indices_unknown] = self.mask_token_id
        
        # decoder_input_ids
        decoder_input_ids = torch.arange(single_T + 1).unsqueeze(0).expand(resized_mask.shape[0], -1).to(self.device)  # B, 257
        decoder_input_ids[:, 1:] = z_indices.clone()
        decoder_input_ids[:, 0] = self.start_token_id  # [start]
        
        is_sparse=True
           
        # no randomness
        temp = self.transformer(input_ids=a_indices, cond_ids=c_indices, decoder_input_ids=decoder_input_ids, is_sparse=is_sparse,
                                e_self_rand_attn=e_self_rand_attn, d_causal_rand_attn=d_causal_rand_attn, d_cross_rand_attn=d_cross_rand_attn)
        logits = temp[0]  # normal order  BS, L + 1, 1024

        logits = torch.cat((logits[:, 0:1, :], normal_to_strip_order(logits[:, 1:, :], block_h=self.block_h, block_w=self.block_w)), dim=1)
        logits = logits[:, :-1, :]  # the last one is a redundant one    BS, L, 1024   z order
        gt_indices = normal_to_strip_order(z_indices.unsqueeze(-1), block_h=self.block_h, block_w=self.block_w)
        gt_indices = gt_indices.squeeze(-1)  #   BS, L   z order
        small_bs = resized_mask_tensor.shape[0]
        small_h = resized_mask_tensor.shape[2]
        small_w = resized_mask_tensor.shape[3]
        normal_mask_tensor = resized_mask_tensor[:, 0, :, :].reshape(small_bs, small_h * small_w)  # BS, L  normal order
        z_mask_tensor = normal_to_strip_order(normal_mask_tensor.unsqueeze(-1), block_h=self.block_h, block_w=self.block_w)
        z_mask_array = z_mask_tensor.squeeze(-1).cpu().numpy()  # BS, L  z order
        # compute losses
        logits_list = []
        target_list = []
        for bid in range(resized_mask.shape[0]):
            flatten_np = z_mask_array[bid]  # L
            indices_unknown = np.nonzero(flatten_np)[0]  # 50, 
            logits_list.append(logits[bid, indices_unknown, :])  # 50, 1024
            target_list.append(gt_indices[bid, indices_unknown])   

        logits = torch.cat(logits_list, 0)
        target = torch.cat(target_list, 0)     
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))    
        
        return loss
            
    @torch.no_grad()
    def autoregressive_sample_fast256(self, z_indices, c_indices, H, W, latent_mask, batch_size=None,
                              temperature=1.0, top_k=100, output_nll=False):  # autoregressive fast testing
        # guiding transformer for 256x256 images
        assert self.is_SGA
        the_transformer = self.guiding_transformer  # guiding transformer
            
        assert H == W  # support square image
        assert H == 16
        idx = z_indices.clone()  # 1, L
        idx = idx.reshape(idx.shape[0], H, W)  # 1, h//16, w//16
        cidx = c_indices.clone()
        cidx = cidx.reshape(cidx.shape[0], H, W)
        if batch_size is not None:  # parallel sampling
            idx = idx.expand(batch_size, -1, -1).clone()
            cidx = cidx.expand(batch_size, -1, -1).clone()
            likelihood_single = np.zeros((batch_size, ))
        
        #------ get encoder representation  just once
        patch = idx.clone()  # bs, h, w
        patch = patch.reshape(patch.shape[0],-1)  # bs, hw
        cpatch = cidx.clone()  # bs, h, w
        cpatch = cpatch.reshape(cpatch.shape[0], -1)  # bs, hw
        
        flatten_np = latent_mask.flatten()
        indices_unknown = np.nonzero(flatten_np)[0]  # positions in z_indices
        # mask input image tokens
        a_indices = patch.clone()
        a_indices[:, indices_unknown] = self.mask_token_id
        
        encoder_outputs =  the_transformer.encoder(input_ids=a_indices,
                                                   cond_ids=cpatch,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   return_dict=True)
        encoder_feats = encoder_outputs[0]  # bs, L, 1024
        #------ get decoder care-about list
        z_flatten_np = np.zeros((flatten_np.shape[0] + 1, ), dtype=flatten_np.dtype)
        z_flatten_np[1:] = flatten_np   # 257
        labeled_block_mask_np = measure.label(z_flatten_np[np.newaxis, :], background=2, connectivity=1)[0, :]  # label start from 1
        number_components = np.amax(labeled_block_mask_np)
        assert number_components >= 2
        care_about_list = []
        for comp_id in range(1, number_components + 1):  # [1, number_components]
            temp_indices = np.nonzero(labeled_block_mask_np == comp_id)[0]
            if z_flatten_np[temp_indices[0]] == 1:  # mask block
                for temp_b_id in list(temp_indices):
                    care_about_list.append([temp_b_id, temp_b_id + 1])
            elif z_flatten_np[temp_indices[0]] == 0 and comp_id != number_components:
                care_about_list.append([np.amin(temp_indices), np.amax(temp_indices) + 1])
            else:
                pass
        #------ get decoder position embeddings
        t_config = the_transformer.config
        decoder_positions = the_transformer.decoder.get_pe_fast(encoder_feats)  # bs, L + 1, 1024    in normal order
        
        #-------- decoder_input_ids 
        #  will be changed in-place
        single_T = patch.shape[1]  # 256
        decoder_input_ids = torch.arange(single_T + 1).unsqueeze(0).expand(patch.shape[0], -1).to(patch.device)  # B, 257
        decoder_input_ids[:, 1:] = patch.clone()
        decoder_input_ids[:, 0] = self.start_token_id  # [start]  BS, 257
        
        #-------- sampling
        previous_info = {}  # important!!!!
        hidden_states = None
        for care_tuple in care_about_list:  # loop over blocks
            care_start = care_tuple[0]
            care_end = care_tuple[1]  # exclusive
            if z_flatten_np[care_start] == 1:  #  do_record = True for last token
                assert care_end == care_start + 1
                # DO SAMPLINNG 
                assert hidden_states is not None
                logits = the_transformer.lm_head(hidden_states)
                logits = logits[:, -1, :]  # bs, 1024
                #----- sample tokens
                logits = logits/temperature
                logits = self.top_k_logits(logits, top_k)
                probs = torch.nn.functional.softmax(logits, dim=-1)  # BS, 1024
                the_selected_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)  # BS
                
                decoder_input_ids[:, care_start] = the_selected_idx
                #----- compute - log likelihood
                if output_nll:
                    the_selected_idx = the_selected_idx.cpu().numpy()
                    probs = probs.cpu().numpy()
                    likelihood_single -= np.log(probs[np.arange(probs.shape[0]), the_selected_idx])
                
            hidden_states, previous_info = the_transformer.decoder.forward_fast256(input_ids=decoder_input_ids, 
                                                        encoder_hidden_states=encoder_feats, decoder_positions=decoder_positions,
                                                        previous_info=previous_info, care_start=care_start, care_end=care_end)
        #-------- output
        ans_idx = decoder_input_ids[:, 1:].reshape(decoder_input_ids.shape[0], H, W)
        if output_nll:
            return ans_idx, likelihood_single
        else:
            return ans_idx
                     
    @torch.no_grad()
    def autoregressive_sample_fast(self, z_indices, c_indices, H, W, latent_mask, batch_size=None,
                                   e_self_rand_attn=None, d_causal_rand_attn=None, d_cross_rand_attn=None, 
                                   temperature=1.0, top_k=100, output_nll=False):  # autoregressive testing
        assert self.is_SGA
        assert H == W  # square image
        idx = z_indices.clone()  # 1, L
        idx = idx.reshape(idx.shape[0], H, W)  # 1, H//16, W//16
        cidx = c_indices.clone()
        cidx = cidx.reshape(cidx.shape[0], H, W)
        if batch_size is not None:  # parallel sampling
            idx = idx.expand(batch_size, -1, -1).clone()
            cidx = cidx.expand(batch_size, -1, -1).clone()
            likelihood_single = np.zeros((batch_size, ))
        
        #------ get encoder representation  just once
        patch = idx.clone()  # bs, h, w
        patch = patch.reshape(patch.shape[0],-1)  # bs, hw
        cpatch = cidx.clone()  # bs, h, w
        cpatch = cpatch.reshape(cpatch.shape[0], -1)  # bs, hw
        
        flatten_np = latent_mask.flatten()
        indices_unknown = np.nonzero(flatten_np)[0]  # positions in z_indices
        # mask input image tokens
        a_indices = patch.clone()
        a_indices[:, indices_unknown] = self.mask_token_id
        
        encoder_outputs =  self.transformer.encoder(input_ids=a_indices,
                                                    cond_ids=cpatch,
                                                    output_attentions=False,
                                                    output_hidden_states=False,
                                                    return_dict=True,
                                                    is_sparse=True,
                                                    e_self_rand_attn=e_self_rand_attn)
        encoder_feats = encoder_outputs[0]  # bs, L, 1024    in z order
        #------ get decoder care-about list
        BLOCK_H = self.block_h
        BLOCK_W = self.block_w
        TOKEN_H = H // BLOCK_H
        TOKEN_W = W // BLOCK_W    
        TOKEN_N = TOKEN_H * TOKEN_W  # number of tokens inside each block
        flatten_mask_tensor = torch.from_numpy(flatten_np.astype(np.float32)).unsqueeze(0).unsqueeze(-1)  # 1, L, 1  in normal order
        flatten_mask_tensor = normal_to_strip_order(flatten_mask_tensor, block_h=BLOCK_H, block_w=BLOCK_W).permute(0, 2, 1)  # 1, 1, L  in z order
        z_flatten_np = flatten_mask_tensor.squeeze().cpu().numpy()  # L     {0, 1}       IMPORTANT z ordr
        block_mask_tensor = F.max_pool1d(flatten_mask_tensor, kernel_size=TOKEN_N, stride=TOKEN_N)  # 1, 1, 64
        block_mask_np = np.uint8(block_mask_tensor.squeeze().cpu().numpy())  #   64,   {0, 1}    1 means masked      IMPORTANT z order
        labeled_block_mask_np = measure.label(block_mask_np[np.newaxis, :], background=2, connectivity=1)[0, :]  # label start from 1
        number_components = np.amax(labeled_block_mask_np)
        assert number_components >= 2
        care_about_list = []
        for comp_id in range(1, number_components + 1):  # [1, number_components]
            temp_indices = np.nonzero(labeled_block_mask_np == comp_id)[0]
            if block_mask_np[temp_indices[0]] == 1:  # mask block
                for temp_b_id in list(temp_indices):
                    care_about_list.append([temp_b_id, temp_b_id + 1])
            elif block_mask_np[temp_indices[0]] == 0 and comp_id != number_components:
                care_about_list.append([np.amin(temp_indices), np.amax(temp_indices) + 1])
            else:
                pass
        #------ get decoder position embeddings
        t_config = self.transformer.config
        decoder_positions = self.transformer.decoder.get_pe_fast(encoder_feats)  # bs, L + 1, 1024    in normal order
        # block position embedding    now in z order
        decoder_positions = torch.cat((decoder_positions[:, 0:1, :], normal_to_strip_order(decoder_positions[:, 1:, :], block_h=BLOCK_H, block_w=BLOCK_W)), dim=1)
        
        #-------- decoder_input_ids   
        #  will be changed in-place
        single_T = patch.shape[1]  # 256, 1024, 4096
        decoder_input_ids = torch.arange(single_T + 1).unsqueeze(0).expand(patch.shape[0], -1).to(patch.device)  # B, 257
        decoder_input_ids[:, 1:] = patch.clone()
        decoder_input_ids[:, 0] = self.start_token_id  # [start] 
        decoder_input_ids = decoder_input_ids.unsqueeze(-1)  # bs, L + 1, 1
        decoder_input_ids = torch.cat((decoder_input_ids[:, 0:1, :], normal_to_strip_order(decoder_input_ids[:, 1:, :], block_h=BLOCK_H, block_w=BLOCK_W)), dim=1)
        decoder_input_ids = decoder_input_ids.squeeze(-1)   #  in z order

        previous_info = {}  # important!!!!
        hidden_states = None
        for care_tuple in care_about_list:  # loop over blocks
            care_start = care_tuple[0]
            care_end = care_tuple[1]  # exclusive
            if block_mask_np[care_start] == 1:  #  do_record = True for last token
                assert care_end == care_start + 1
                # DO SAMPLINNG   loop over masked tokens
                chuncked_z_mask_np = z_flatten_np[care_start*TOKEN_N:care_end*TOKEN_N]
                list_tid_chuncked = list(np.nonzero(chuncked_z_mask_np)[0])
                if 0 not in list_tid_chuncked:
                    hidden_states = None
                for tid_chuncked in list_tid_chuncked:  # [0, TOKEN_N)

                    #----- get the distribution for tid_chuncked
                    if hidden_states is None:
                        hidden_states, previous_info = self.transformer.decoder.forward_fast(input_ids=decoder_input_ids, 
                                                        encoder_hidden_states=encoder_feats, decoder_positions=decoder_positions,
                                                        is_sparse=True, d_causal_rand_attn=d_causal_rand_attn, d_cross_rand_attn=d_cross_rand_attn,
                                                        previous_info=previous_info, care_start=care_start, care_end=care_end, token_n=TOKEN_N, do_record=False)
                    logits = self.transformer.lm_head(hidden_states)
                    if tid_chuncked == 0 and care_start > 0:  # need to use previous block
                        logits = logits[:, -1, :]  # bs, 1024
                    else:  # need to use the same block hidden_states
                        if care_start == 0:
                            logits = logits[:, tid_chuncked, :]  # bs, 1024
                        else:
                            logits = logits[:, tid_chuncked - 1, :]  # bs, 1024
                    
                    #----- sample tokens for tid_chuncked
                    logits = logits/temperature
                    logits = self.top_k_logits(logits, top_k)
                    probs = torch.nn.functional.softmax(logits, dim=-1)  # BS, 1024
                    the_selected_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)  # BS

                    decoder_input_ids[:, care_start*TOKEN_N+tid_chuncked+1] = the_selected_idx
                    #----- compute - log likelihood
                    if output_nll:
                        the_selected_idx = the_selected_idx.cpu().numpy()
                        probs = probs.cpu().numpy()
                        likelihood_single -= np.log(probs[np.arange(probs.shape[0]), the_selected_idx])
                    #----- prepare hidden_states for next token
                    if tid_chuncked == list_tid_chuncked[-1]:
                        d_record = True
                    else:
                        d_record = False
                    hidden_states, previous_info = self.transformer.decoder.forward_fast(input_ids=decoder_input_ids, 
                                                        encoder_hidden_states=encoder_feats, decoder_positions=decoder_positions,
                                                        is_sparse=True, d_causal_rand_attn=d_causal_rand_attn, d_cross_rand_attn=d_cross_rand_attn,
                                                        previous_info=previous_info, care_start=care_start, care_end=care_end, token_n=TOKEN_N, do_record=d_record)
         
            else:  # all known, no sampling    all in z order    might provide distribution for the first token in next block
                hidden_states, previous_info = self.transformer.decoder.forward_fast(input_ids=decoder_input_ids, 
                                                        encoder_hidden_states=encoder_feats, decoder_positions=decoder_positions,
                                                        is_sparse=True, d_causal_rand_attn=d_causal_rand_attn, d_cross_rand_attn=d_cross_rand_attn,
                                                        previous_info=previous_info, care_start=care_start, care_end=care_end, token_n=TOKEN_N, do_record=True)
                
        #-------- output
        decoder_input_ids = strip_to_normal_order(decoder_input_ids[:, 1:].unsqueeze(-1), block_h=BLOCK_H, block_w=BLOCK_W)
        ans_idx = decoder_input_ids.squeeze(-1).reshape(decoder_input_ids.shape[0], H, W)
        if output_nll:
            return ans_idx, likelihood_single
        else:
            return ans_idx
                   
    def shared_step_guiding(self, batch, batch_idx, mask_tensor):
        x, c = self.get_xc(batch)
        xh = x.shape[2]
        xw = x.shape[3]
        resized_mask_tensor = F.interpolate(mask_tensor, size=(xh // (2 ** self.num_downsampling), xw // (2 ** self.num_downsampling)))
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x, mask_tensor=mask_tensor)  # bs, 256
        _, c_indices = self.encode_to_c(c)  # bs, 256
        
        resized_mask = resized_mask_tensor[:, 0, :, :].cpu().numpy()  # bs, 16, 16
        single_T = z_indices.shape[1] # 256
        
        # mask input image tokens
        a_indices = z_indices.clone()
        for bid in range(resized_mask.shape[0]):
            flatten_np = resized_mask[bid].flatten()
            indices_unknown = np.nonzero(flatten_np)[0]  # positions in z_indices
            a_indices[bid, indices_unknown] = self.mask_token_id
        
        # decoder_input_ids
        decoder_input_ids = torch.arange(single_T + 1).unsqueeze(0).expand(resized_mask.shape[0], -1).to(self.device)  # B, 257
        decoder_input_ids[:, 1:] = z_indices.clone()
        decoder_input_ids[:, 0] = self.start_token_id  # [start]
        
        temp = self.transformer(input_ids=a_indices, cond_ids=c_indices, decoder_input_ids=decoder_input_ids)
        logits = temp[0]
        logits = logits[:, :-1, :]  # the last one is a redundant one
        # compute losses
        logits_list = []
        target_list = []
        for bid in range(resized_mask.shape[0]):
            flatten_np = resized_mask[bid].flatten()
            indices_unknown = np.nonzero(flatten_np)[0]
            logits_list.append(logits[bid, indices_unknown, :])
            target_list.append(z_indices[bid, indices_unknown])
        logits = torch.cat(logits_list, 0)
        target = torch.cat(target_list, 0)     
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))    
        
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif 'PEG_net' in pn:  # PEG no decay
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        if 'pos_emb' in param_dict.keys():
            no_decay.add('pos_emb')
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer

    def load_guiding_transformer(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        own_state = self.guiding_transformer.state_dict()
        for name, param in state_dict.items():
            if 'transformer' in name:
                new_name = '.'.join(name.split('.')[1:])
                if new_name not in own_state.keys():
                    assert 0
                own_state[new_name].copy_(param)
            else:
                continue