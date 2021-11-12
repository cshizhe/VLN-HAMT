
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel

from models.vilmodel_cmt import (
    BertLayerNorm, BertEmbeddings, ImageEmbeddings,
    HistoryEmbeddings, LxmertEncoder, NextActionPrediction,
    )

class ObjectEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
       
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size) 
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.pos_linear = nn.Linear(5, config.hidden_size)
        self.pos_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, obj_feat, obj_ang, obj_pos, type_embeddings, nav_type_embeddings):
        batch_size = obj_feat.size(0)
        device = obj_feat.device

        embeddings = self.img_layer_norm(self.img_linear(obj_feat)) + \
                     self.ang_layer_norm(self.ang_linear(obj_ang)) + \
                     self.pos_layer_norm(self.pos_linear(obj_pos)) + \
                     nav_type_embeddings + type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NavRefCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)
        self.obj_embeddings = ObjectEmbeddings(config)
        self.hist_embeddings = HistoryEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)
        self.ref_object = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        self.init_weights()

    def forward(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                hist_img_feats=None, hist_ang_feats=None, 
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, ob_step_ids=None, hist_masks=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, ob_masks=None,
                obj_feats=None, obj_angles=None, obj_poses=None, obj_masks=None):
        
        # text embedding            
        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            txt_token_type_ids = torch.zeros_like(txt_ids)
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            for layer_module in self.encoder.layer:
                temp_output = layer_module(txt_embeds, extended_txt_masks)
                txt_embeds = temp_output[0]
            if self.config.fix_lang_embedding:
                txt_embeds = txt_embeds.detach()
            return txt_embeds

        if mode == 'history':
            hist_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, ob_step_ids,
                pano_img_feats=hist_pano_img_feats, pano_ang_feats=hist_pano_ang_feats)
            if self.config.fix_hist_embedding:
                hist_embeds = hist_embeds.detach()
            return hist_embeds

        elif mode == 'visual':
            ''' LXMERT visual branch'''
            # history embedding
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
            extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers:
                    temp_output = layer_module(hist_embeds, extended_hist_masks)
                    hist_embeds = temp_output[0]

            # image embedding
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0

            ob_token_type_ids = torch.ones(ob_img_feats.size(0), ob_img_feats.size(1), dtype=torch.long, device=self.device)
            ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats, 
                self.embeddings.token_type_embeddings(ob_token_type_ids), 
                nav_types=ob_nav_types)
            if self.encoder.r_layers is not None:
                for layer_module in self.encoder.r_layers:
                    temp_output = layer_module(ob_embeds, extended_ob_masks)
                    ob_embeds = temp_output[0]
            if self.config.fix_obs_embedding:
                ob_embeds = ob_embeds.detach()

            # object embedding
            extended_obj_masks = obj_masks.unsqueeze(1).unsqueeze(2)
            extended_obj_masks = extended_obj_masks.to(dtype=self.dtype)
            extended_obj_masks = (1.0 - extended_obj_masks) * -10000.0

            obj_token_type_ids = torch.ones(obj_feats.size(0), obj_feats.size(1), dtype=torch.long, device=self.device)
            # STOP nav_type
            obj_nav_type_ids = torch.zeros(obj_feats.size(0), obj_feats.size(1), dtype=torch.long, device=self.device) + 2
            obj_embeds = self.obj_embeddings(obj_feats, obj_angles, obj_poses,
                self.embeddings.token_type_embeddings(obj_token_type_ids),
                self.img_embeddings.nav_type_embedding(obj_nav_type_ids))

            # multi-modal encoding
            hist_max_len = hist_embeds.size(1)
            ob_max_len = ob_embeds.size(1)
            obj_max_len = obj_embeds.size(1)

            vision_embeds = torch.cat([hist_embeds, ob_embeds, obj_embeds], 1)
            extended_vision_masks = torch.cat([extended_hist_masks, extended_ob_masks, extended_obj_masks], -1)

            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            for layer_module in self.encoder.x_layers:
                txt_embeds, vision_embeds = layer_module(
                    txt_embeds, extended_txt_masks, 
                    vision_embeds, extended_vision_masks,
                )

            hist_embeds = vision_embeds[:, :hist_max_len]
            ob_embeds = vision_embeds[:, hist_max_len: hist_max_len+ob_max_len]
            obj_embeds = vision_embeds[:, hist_max_len+ob_max_len:]

            act_logits = self.next_action(ob_embeds * hist_embeds[:, :1]).squeeze(-1)
            obj_logits = self.ref_object(obj_embeds * txt_embeds[:, :1]).squeeze(-1)

            act_logits.masked_fill_(ob_nav_types==0, -float('inf'))
            obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
            
            return act_logits, obj_logits, txt_embeds, hist_embeds, ob_embeds, obj_embeds

