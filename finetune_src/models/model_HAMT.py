import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import length2mask

from models.vlnbert_init import get_vlnbert_models

class VLNBertCMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(self, mode, txt_ids=None, txt_masks=None, txt_embeds=None, 
                hist_img_feats=None, hist_ang_feats=None, 
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, hist_lens=None, ob_step=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
                ob_masks=None, return_states=False):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)
            if ob_step is not None:
                ob_step_ids = torch.LongTensor([ob_step]).cuda()
            else:
                ob_step_ids = None
            hist_embeds = self.vln_bert(mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, ob_step_ids=ob_step_ids,
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats)
            return hist_embeds

        elif mode == 'visual':
            hist_embeds = torch.stack(hist_embeds, 1)
            hist_masks = length2mask(hist_lens, size=hist_embeds.size(1)).logical_not()
            
            ob_img_feats = self.drop_env(ob_img_feats)
            
            act_logits, txt_embeds, hist_embeds, ob_embeds = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_embeds=hist_embeds, hist_masks=hist_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks)

            if return_states:
                if self.args.no_lang_ca:
                    states = hist_embeds[:, 0]
                else:
                    states = txt_embeds[:, 0] * hist_embeds[:, 0]   # [CLS]
                return act_logits, states
            return (act_logits, )


class VLNBertCausalCMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(
        self, mode, txt_ids=None, txt_masks=None, txt_embeds=None,
        hist_img_feats=None, hist_ang_feats=None, 
        hist_pano_img_feats=None, hist_pano_ang_feats=None, ob_step=0,
        new_hist_embeds=None, new_hist_masks=None,
        prefix_hiddens=None, prefix_masks=None,
        ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None,
        ob_masks=None, return_states=False, batch_size=None,
    ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            if ob_step == 0:
                hist_step_ids = torch.arange(1).long()
            else:
                hist_step_ids = torch.arange(2).long() + ob_step - 1
            hist_step_ids = hist_step_ids.unsqueeze(0)

            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)

            hist_embeds = self.vln_bert(
                mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, 
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats,
                hist_step_ids=hist_step_ids,
                batch_size=batch_size
            )
            return hist_embeds

        elif mode == 'visual':
            ob_img_feats = self.drop_env(ob_img_feats)
            
            act_logits, prefix_hiddens, states = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks,
                new_hist_embeds=new_hist_embeds, new_hist_masks=new_hist_masks,
                prefix_hiddens=prefix_hiddens, prefix_masks=prefix_masks
            )

            if return_states:
                return act_logits, prefix_hiddens, states
            return (act_logits, prefix_hiddens)


class VLNBertMMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(
        self, mode, txt_ids=None, txt_masks=None, txt_embeds=None, 
        hist_img_feats=None, hist_ang_feats=None, 
        hist_pano_img_feats=None, hist_pano_ang_feats=None,
        hist_embeds=None, hist_masks=None, ob_step=None,
        ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
        ob_masks=None, return_states=False, batch_size=None,
        prefix_embeds=None, prefix_masks=None
    ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            if hist_img_feats is None:
                # only encode [sep] token
                hist_step_ids = torch.zeros((batch_size, 1), dtype=torch.long)
            else:
                # encode the new observation and [sep]
                hist_step_ids = torch.arange(2).long().expand(batch_size, -1) + ob_step - 1
            
            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)
            
            new_hist_embeds = self.vln_bert(
                mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, 
                hist_step_ids=hist_step_ids,
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats,
                batch_size=batch_size,
            )
            return new_hist_embeds

        elif mode == 'visual':
            ob_img_feats = self.drop_env(ob_img_feats)
            
            outs = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_embeds=hist_embeds, hist_masks=hist_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks,
                prefix_embeds=prefix_embeds, prefix_masks=prefix_masks
            )

            act_logits, hist_state = outs[:2]

            if return_states:
                return (act_logits, hist_state) + outs[2:]

            return (act_logits, ) + outs[2:]


class VLNBertCMT3(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(
        self, mode, txt_ids=None, txt_masks=None,
        hist_img_feats=None, hist_ang_feats=None, 
        hist_pano_img_feats=None, hist_pano_ang_feats=None, ob_step=0,
        ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None,
        ob_masks=None, return_states=False, 
        txt_embeds=None, hist_in_embeds=None, hist_out_embeds=None, hist_masks=None
    ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            if ob_step == 0:
                hist_step_ids = torch.arange(1).long()
            else:
                hist_step_ids = torch.arange(2).long() + ob_step - 1
            hist_step_ids = hist_step_ids.unsqueeze(0)

            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)

            hist_in_embeds, hist_out_embeds = self.vln_bert(
                mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, 
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats,
                hist_step_ids=hist_step_ids,
                hist_in_embeds=hist_in_embeds,
                hist_out_embeds=hist_out_embeds,
                hist_masks=hist_masks
            )
            return hist_in_embeds, hist_out_embeds

        elif mode == 'visual':
            ob_img_feats = self.drop_env(ob_img_feats)
            
            act_logits, states = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_out_embeds=hist_out_embeds, hist_masks=hist_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks,
            )

            if return_states:
                return act_logits, states
            return (act_logits, )


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
