from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead

from .pretrain import (NextActionPrediction, NextActionRegression,
                       SpatialRelRegression, RegionClassification,
                       ItmPrediction)

from .image_vilmodel import NavTHORImagePreTrainedModel


class MultiStepNavImagePreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = NavTHORImagePreTrainedModel(config)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'sap' in config.pretrain_tasks:
            self.next_action = NextActionPrediction(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'sar' in config.pretrain_tasks:
            self.regress_action = NextActionRegression(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'sprel' in config.pretrain_tasks:
            self.sprel_head = SpatialRelRegression(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
        if 'itm' in config.pretrain_tasks:
            self.itm_head = ItmPrediction(self.config.hidden_size)
        
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(batch['txt_ids'], batch['txt_masks'], 
                                    batch['hist_images'], batch['hist_ang_fts'],
                                    batch['hist_pano_images'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['txt_labels'], compute_loss)
        elif task.startswith('sap'):
            return self.forward_sap(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_images'], batch['hist_ang_fts'],
                                    batch['hist_pano_images'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_images'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['ob_action_viewindex'], batch['ob_v_exists'], compute_loss)
        elif task.startswith('sar'):
            return self.forward_sar(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_images'], batch['hist_ang_fts'],
                                    batch['hist_pano_images'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_images'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['ob_action_angles'], batch['ob_progress'], 
                                    batch['ob_v_exists'], compute_loss)
        elif task.startswith('sprel'):
            return self.forward_sprel(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_images'], batch['hist_ang_fts'],
                                    batch['hist_pano_images'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_images'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['sp_anchor_idxs'], batch['sp_targets'], 
                                    batch['ob_v_exists'], compute_loss)
        elif task.startswith('mrc'):
            return self.forward_mrc(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_images'], batch['hist_ang_fts'],
                                    batch['hist_pano_images'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['hist_mrc_masks'], batch['hist_img_probs'], compute_loss)
        elif task.startswith('itm'):
            return self.forward_itm(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_images'], batch['hist_ang_fts'],
                                    batch['hist_pano_images'], batch['hist_pano_ang_fts'], batch['hist_masks'], 4, compute_loss)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, txt_ids, txt_masks, 
                    hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks, 
                    txt_labels, compute_loss):
        txt_embeds, _, _ = self.bert(txt_ids, txt_masks, 
            hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks,
            None, None, None, None)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(prediction_scores, 
                                        txt_labels[txt_labels != -1], 
                                        reduction='none')
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_sap(self, txt_ids, txt_masks, 
                    hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks, 
                    ob_images, ob_ang_fts, ob_nav_types, ob_masks, 
                    act_labels, ob_v_exists, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks, 
            hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks,
            ob_images, ob_ang_fts, ob_nav_types, ob_masks, ob_v_exists=ob_v_exists)
        
        # combine text and visual to predict next action
        prediction_scores = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
        prediction_scores.masked_fill_(ob_nav_types == 0, -float('inf'))

        if compute_loss:
            act_loss = F.cross_entropy(prediction_scores, act_labels, reduction='none')
            return act_loss
        else:
            return prediction_scores

    def forward_sar(self, txt_ids, txt_masks, 
                    hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks, 
                    ob_images, ob_ang_fts, ob_nav_types, ob_masks, 
                    ob_act_angles, ob_progress, ob_v_exists, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks, 
            hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks,
            ob_images, ob_ang_fts, ob_nav_types, ob_masks, ob_v_exists=ob_v_exists)

        prediction_scores = self.regress_action(txt_embeds[:, 0])   # [CLS] token

        if compute_loss:
            act_targets = torch.cat([ob_act_angles, ob_progress.unsqueeze(1)], dim=1)
            act_loss = F.mse_loss(prediction_scores, act_targets, reduction='none')
            return act_loss
        else:
            return prediction_scores

    def forward_sprel(self, txt_ids, txt_masks, 
                    hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks, 
                    ob_images, ob_ang_fts, ob_nav_types, ob_masks, 
                    sp_anchor_idxs, sp_targets, ob_v_exists, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks, 
            hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks,
            ob_images, ob_ang_fts, ob_nav_types, ob_masks, ob_v_exists=ob_v_exists)

        # img_embeds: (batch, views, dim), sp_anchor_idxs: (batch)
        anchor_ob_embeds = torch.gather(ob_embeds, 1, 
            sp_anchor_idxs.unsqueeze(1).unsqueeze(2).repeat(1, 36, ob_embeds.size(-1)))
        # (batch, 1, dim)
        cat_ob_embeds = torch.cat([anchor_ob_embeds, ob_embeds[:, :-1]], -1)
        
        prediction_scores = self.sprel_head(cat_ob_embeds) # (batch, 36, 2)

        if compute_loss:
            sprel_loss = F.mse_loss(prediction_scores, sp_targets, reduction='none')
            return sprel_loss
        else:
            return prediction_scores

    def forward_mrc(self, txt_ids, txt_masks, 
                    hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks, 
                    hist_mrc_masks, hist_img_probs, compute_loss=True):
        txt_embeds, hist_embeds, _ = self.bert(txt_ids, txt_masks, 
            hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks,
            None, None, None, None, hist_mrc_masks=hist_mrc_masks)

        # only compute masked regions for better efficient=cy
        hist_embeds = hist_embeds[:, 1:] # remove global embedding
        masked_output = self._compute_masked_hidden(hist_embeds, hist_mrc_masks)
        prediction_soft_labels = self.image_classifier(masked_output)

        hist_mrc_targets = self._compute_masked_hidden(hist_img_probs, hist_mrc_masks)

        if compute_loss:
            prediction_soft_labels = F.log_softmax(prediction_soft_labels, dim=-1)
            mrc_loss = F.kl_div(prediction_soft_labels, hist_mrc_targets, reduction='none').sum(dim=1)
            return mrc_loss
        else:
            return prediction_soft_labels, hist_mrc_targets

    def forward_itm(self, txt_ids, txt_masks, 
                    hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks, 
                    num_neg_trajs, compute_loss):
        # (batch_size, 1+num_negs, dim)
        fused_embeds = self.bert.forward_itm(
            txt_ids, txt_masks, 
            hist_images, hist_ang_fts, hist_pano_images, hist_pano_ang_fts, hist_masks,
            num_neg_trajs=num_neg_trajs)

        prediction_scores = self.itm_head(fused_embeds).squeeze(2) # (batch, 1+num_negs, 1)
        # The first is positive
        itm_targets = torch.zeros(fused_embeds.size(0), dtype=torch.long).to(self.device)

        if compute_loss:
            sprel_loss = F.cross_entropy(prediction_scores, itm_targets, reduction='none')
            return sprel_loss
        else:
            return prediction_scores, itm_targets
