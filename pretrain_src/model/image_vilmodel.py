import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from .vision_transformer import vit_base_patch16_224

from .vilmodel import BertEmbeddings, ImageEmbeddings, HistoryEmbeddings, LxmertEncoder


class NavTHORImagePreTrainedModel(BertPreTrainedModel):
    r""" Modification of LXMERT Model """
    def __init__(self, config):
        super().__init__(config)
        self.vision_backbone = vit_base_patch16_224(pretrained=True,
            drop_rate=config.hidden_dropout_prob, 
            attn_drop_rate=config.attention_probs_dropout_prob, 
            drop_path_rate=0.)

        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)
        # share image encoding
        self.hist_embeddings = HistoryEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.init_weights()

    def forward_vision_backbone(self, images, detach=False):
        # due to memory issue, we cannot propagate to pano images in the history
        is_pano = len(images.size()) == 6
        if is_pano:
            N, T, P, C, H, W = images.size()    # pano images
            images = images.view(N*T*P, C, H, W)
        else:
            N, T, C, H, W = images.size()
            images = images.view(N*T, C, H, W)
        # feats = self.vision_backbone.forward_features(images)
        if is_pano:
            with torch.no_grad():
                feats = self.vision_backbone.forward_features(images)
            feats = feats.view(N, T, P, -1)
        else:
            feats = self.vision_backbone.forward_features(images)
            feats = feats.view(N, T, -1)
        if detach:
            feats = feats.detach()
        return feats

    def forward(self, txt_ids, txt_masks, 
                hist_images, hist_ang_feats, hist_pano_images, hist_pano_ang_feats, hist_masks,
                ob_images, ob_ang_feats, ob_nav_types, ob_masks,
                hist_mrc_masks=None, ob_v_exists=None):
        batch_size = txt_ids.size(0)

        # text embedding
        extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
        extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
        extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)

        # history embedding
        extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
        extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
        extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0
        if hist_images is not None:
            hist_max_len = hist_images.size(1)
            hist_img_feats = self.forward_vision_backbone(hist_images)
            hist_pano_img_feats = self.forward_vision_backbone(hist_pano_images, detach=True)
            hist_step_ids = torch.arange(hist_max_len).expand((1, -1)).to(self.device)
            if hist_mrc_masks is not None:  # (N, T)
                hist_img_feats = hist_img_feats.masked_fill(hist_mrc_masks.unsqueeze(-1), 0)
                hist_pano_img_feats = hist_pano_img_feats.masked_fill(hist_mrc_masks.unsqueeze(-1).unsqueeze(-1), 0)
        else:
            hist_step_ids = None
            hist_img_feats = None
            hist_pano_img_feats = None
        hist_cls_embeds, hist_vp_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, 
            hist_pano_img_feats, hist_pano_ang_feats, hist_step_ids,
            batch_size=batch_size)
        if hist_vp_embeds is None:
            hist_embeds = hist_cls_embeds
        else:
            hist_embeds = torch.cat([hist_cls_embeds, hist_vp_embeds], dim=1)

        # image embedding
        if ob_images is not None:
            ob_img_feats = self.forward_vision_backbone(ob_images)
            if ob_v_exists is not None: # [N, ]
                ob_img_feats = ob_img_feats.masked_fill(ob_v_exists.logical_not().unsqueeze(-1).unsqueeze(-1), 0)
            # add STOP token
            ob_img_feats= torch.cat([ob_img_feats, \
                torch.zeros(batch_size, 1, ob_img_feats.size(2), dtype=ob_img_feats.dtype, device=ob_img_feats.device)],
                dim=1)
            ob_token_type_ids = torch.ones(batch_size, 1).long().to(self.device)
            ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats, 
                self.embeddings.token_type_embeddings(ob_token_type_ids), 
                nav_types=ob_nav_types)
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0
        else:
            ob_embeds, extended_ob_masks = None, None

        # multi-modal encoding
        txt_embeds, hist_embeds, ob_embeds = self.encoder(
            txt_embeds, extended_txt_masks, 
            hist_embeds, extended_hist_masks,
            ob_embeds, extended_ob_masks)

        return txt_embeds, hist_embeds, ob_embeds

    def forward_itm(self, txt_ids, txt_masks, 
                hist_images, hist_ang_feats, hist_pano_images, hist_pano_ang_feats, hist_masks,
                num_neg_trajs=4):
        batch_size, hist_max_len = hist_images.size()[:2]

        # text encoding
        extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
        extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
        extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        for layer_module in self.encoder.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        # copy txt_embeds
        batch_size = txt_embeds.size(0)
        txt_embeds = txt_embeds.repeat(1+num_neg_trajs, 1, 1)
        extended_txt_masks = extended_txt_masks.repeat(1+num_neg_trajs, 1, 1, 1)

        # history encoding
        extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
        extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
        extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

        hist_img_feats = self.forward_vision_backbone(hist_images)
        hist_pano_img_feats = self.forward_vision_backbone(hist_pano_images, detach=True)

        hist_cls_embeds, hist_vp_embeds_no_pos = self.hist_embeddings(hist_img_feats, hist_ang_feats, 
            hist_pano_img_feats, hist_pano_ang_feats, pos_ids=None,
            batch_size=batch_size)
        hist_step_ids = torch.arange(hist_max_len).expand((1, -1)).to(self.device)
        hist_vp_embeds = self.hist_embeddings.dropout(self.hist_embeddings.layer_norm(
            hist_vp_embeds_no_pos + self.hist_embeddings.position_embeddings(hist_step_ids)))
        hist_embeds = torch.cat([hist_cls_embeds, hist_vp_embeds], dim=1)
        
        if self.encoder.h_layers is not None:
            for layer_module in self.encoder.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]

        # random negs in batch
        K = num_neg_trajs // 2
        neg_idxs = []
        for i in range(batch_size):
            neg_idxs.append(np.random.choice(np.arange(0, i).tolist() + np.arange(i+1, batch_size).tolist(), K))
        neg_idxs = torch.from_numpy(np.stack(neg_idxs, 0)).to(self.device)

        neg_hist_embeds, neg_hist_masks = [], []
        for k in range(K):
            neg_hist_embeds.append(hist_embeds[neg_idxs[:, k]])
            neg_hist_masks.append(extended_hist_masks[neg_idxs[:, k]])

        # shuffled negs
        hist_lens = torch.sum(hist_masks, 1) - 1
        for _ in range(K):
            shuffled_pos_ids = []
            for i in range(batch_size):
                shuffled_idxs = torch.randperm(hist_lens[i])
                shuffled_idxs = torch.cat([shuffled_idxs, torch.arange(hist_lens[i], hist_max_len, dtype=torch.long)], 0).to(self.device)
                shuffled_pos_ids.append(shuffled_idxs)
            shuffled_pos_ids = torch.stack(shuffled_pos_ids, 0)
            shuffled_hist_embeds = torch.cat([hist_cls_embeds, \
                self.hist_embeddings.dropout(self.hist_embeddings.layer_norm(
                hist_vp_embeds_no_pos + self.hist_embeddings.position_embeddings(shuffled_pos_ids)))], dim=1)

            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers:
                    temp_output = layer_module(shuffled_hist_embeds, extended_hist_masks)
                    shuffled_hist_embeds = temp_output[0]
            neg_hist_embeds.append(shuffled_hist_embeds)
            neg_hist_masks.append(extended_hist_masks)
        
        pos_neg_hist_embeds = torch.cat([hist_embeds] + neg_hist_embeds, 0)
        pos_neg_hist_masks = torch.cat([extended_hist_masks] + neg_hist_masks, 0)
            
        # multi-modal encoding
        for layer_module in self.encoder.x_layers:
            txt_embeds, pos_neg_hist_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                pos_neg_hist_embeds, pos_neg_hist_masks)

        fused_embeds = txt_embeds[:, 0] * pos_neg_hist_embeds[:, 0]
        fused_embeds = torch.stack(torch.split(fused_embeds, batch_size), 1)
        return fused_embeds

