import random
import math
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .common import pad_tensors, gen_seq_masks

############### Masked Language Modeling ###############
def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_tokens, output_label = [], []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_tokens.append(mask)

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_tokens.append(random.choice(list(range(*vocab_range))))

            # -> rest 10% randomly keep current token
            else:
                output_tokens.append(token)

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            output_tokens.append(token)
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        output_tokens[0] = mask

    return output_tokens, output_label    

class MlmDataset(Dataset):
    def __init__(self, nav_db, tok):
        self.nav_db = nav_db
        self.tok = tok

        self.vocab_range = [1996, 29611] #TODO: manually checked in bert-base-uncased
        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.mask_token_id = self.tok.mask_token_id # 103
        self.pad_token_id = self.tok.pad_token_id   # 0

    def __len__(self):
        return len(self.nav_db.traj_refer)

    def __getitem__(self, i):
        i_traj, j_instr, path_len = self.nav_db.traj_refer[i]

        inputs = self.nav_db.get_input(i_traj, j_instr, path_len,
            return_ob=False, return_ob_action=False, 
            return_hist_img_probs=False, return_ob_progress=False)

        output = {}

        # prepare text tensor
        txt_ids, txt_labels = random_word(inputs['instr_encoding'], 
            self.vocab_range, self.mask_token_id)
        output['txt_ids'] = torch.LongTensor(txt_ids)
        output['txt_labels'] = torch.LongTensor(txt_labels)
        output['txt_lens'] = output['txt_ids'].size(0)

        # prepare history tensor
        output['hist_img_fts'] = torch.from_numpy(inputs['hist_img_fts'])
        output['hist_ang_fts'] = torch.from_numpy(inputs['hist_ang_fts'])
        if 'hist_pano_img_fts' in inputs:
            output['hist_pano_img_fts'] = torch.from_numpy(inputs['hist_pano_img_fts'])
            output['hist_pano_ang_fts'] = torch.from_numpy(inputs['hist_pano_ang_fts'])
        output['hist_lens'] = inputs['hist_lens']

        return output

def mlm_collate(inputs):
    '''
    Return: 
    :txt_ids    (n, max_L) padded with 0
    :txt_labels (n, max_L) padded with -1
    :txt_masks  (n, max_L) padded with 0
    :txt_lens   (n, )
    :img_fts    (n, max_R) padded with 0
    :img_masks  (n, max_R) padded with 0
    :img_lens   (n, )
    '''
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)
    batch['txt_labels'] = pad_sequence(batch['txt_labels'], batch_first=True, padding_value=-1)
    batch['txt_masks'] = torch.BoolTensor(gen_seq_masks(batch['txt_lens']))
    batch['txt_lens'] = torch.LongTensor(batch['txt_lens'])

    # history batches
    batch['hist_img_fts'] = pad_tensors(batch['hist_img_fts'], lens=batch['hist_lens'], pad=0)
    batch['hist_ang_fts'] = pad_tensors(batch['hist_ang_fts'], lens=batch['hist_lens'], pad=0)
    if 'hist_pano_img_fts' in batch:
        batch['hist_pano_img_fts'] = pad_tensors(batch['hist_pano_img_fts'], lens=batch['hist_lens'], pad=0)
        batch['hist_pano_ang_fts'] = pad_tensors(batch['hist_pano_ang_fts'], lens=batch['hist_lens'], pad=0)

    batch['hist_lens'] = [x + 1 for x in batch['hist_lens']] # added a special token
    batch['hist_masks'] = torch.BoolTensor(gen_seq_masks(batch['hist_lens']))
    batch['hist_lens'] = torch.LongTensor(batch['hist_lens'])
    return batch


############### Masked Region Modeling ###############
def _get_img_mask(mask_prob, num_images):
    # do not mask the last [STOP] token
    img_mask = [np.random.rand() < mask_prob for _ in range(num_images)]
    if not any(img_mask):
        # at least mask 1
        img_mask[np.random.randint(num_images)] = True
    img_mask = torch.tensor(img_mask)
    return img_mask

def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked

def _mask_pano_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked

def _get_targets(img_soft_label, img_masks):
    soft_label_dim = img_soft_label.size(-1)
    img_masks_ext_for_label = img_masks.unsqueeze(-1).expand_as(img_soft_label)
    label_targets = img_soft_label[img_masks_ext_for_label].contiguous().view(
        -1, soft_label_dim)
    return label_targets

class MrcDataset(Dataset):
    def __init__(self, nav_db, tok, mask_prob):
        self.nav_db = nav_db
        self.tok = tok
        self.mask_prob = mask_prob

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

    def __len__(self):
        return len(self.nav_db.traj_refer)

    def __getitem__(self, i):
        i_traj, j_instr, path_len = self.nav_db.traj_refer[i]

        inputs = self.nav_db.get_input(i_traj, j_instr, path_len,
            return_ob=False, return_ob_action=False, 
            return_hist_img_probs=True, return_ob_progress=False)

        output = {}

        # prepare text tensor
        txt_ids = inputs['instr_encoding']
        output['txt_ids'] = torch.LongTensor(txt_ids)
        output['txt_lens'] = output['txt_ids'].size(0)

        # prepare history tensor: masked history image
        hist_mrc_masks = _get_img_mask(self.mask_prob, inputs['hist_img_probs'].shape[0])
        output['hist_img_fts'] = _mask_img_feat(
            torch.from_numpy(inputs['hist_img_fts']),
            hist_mrc_masks)
        if 'hist_pano_img_fts' in inputs:
            output['hist_pano_img_fts'] = _mask_pano_img_feat(
                torch.from_numpy(inputs['hist_pano_img_fts']),
                hist_mrc_masks)

        output['hist_img_probs'] = torch.from_numpy(inputs['hist_img_probs'])
        output['hist_mrc_masks'] = hist_mrc_masks

        output['hist_ang_fts'] = torch.from_numpy(inputs['hist_ang_fts'])
        if 'hist_pano_ang_fts' in inputs:
            output['hist_pano_ang_fts'] = torch.from_numpy(inputs['hist_pano_ang_fts'])
        output['hist_lens'] = inputs['hist_lens'] 

        return output

def mrc_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)
    batch['txt_masks'] = torch.BoolTensor(gen_seq_masks(batch['txt_lens']))
    batch['txt_lens'] = torch.LongTensor(batch['txt_lens'])

    # history batches
    batch['hist_img_fts'] = pad_tensors(batch['hist_img_fts'], lens=batch['hist_lens'], pad=0)
    batch['hist_ang_fts'] = pad_tensors(batch['hist_ang_fts'], lens=batch['hist_lens'], pad=0)

    if 'hist_pano_img_fts' in batch:
        batch['hist_pano_img_fts'] = pad_tensors(batch['hist_pano_img_fts'], lens=batch['hist_lens'], pad=0)
        batch['hist_pano_ang_fts'] = pad_tensors(batch['hist_pano_ang_fts'], lens=batch['hist_lens'], pad=0)
    
    # labels
    batch['hist_mrc_masks'] = pad_sequence(batch['hist_mrc_masks'], batch_first=True, padding_value=0)
    batch['hist_img_probs'] = pad_tensors(batch['hist_img_probs'], lens=batch['hist_lens'], pad=0)

    batch['hist_lens'] = [x + 1 for x in batch['hist_lens']] # added a special token
    batch['hist_masks'] = torch.BoolTensor(gen_seq_masks(batch['hist_lens']))
    batch['hist_lens'] = torch.LongTensor(batch['hist_lens'])
    return batch


############### Insturction Trajectory Matching ###############
class ItmDataset(Dataset):
    def __init__(self, nav_db, tok):
        '''Instruction Trajectory Matching'''
        self.nav_db = nav_db
        self.tok = tok

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

    def __len__(self):
        return len(self.nav_db.traj_refer)

    def __getitem__(self, i):
        i_traj, j_instr, path_len = self.nav_db.traj_refer[i]

        inputs = self.nav_db.get_input(i_traj, j_instr, path_len, 
            return_ob=False, return_ob_action=False, 
            return_hist_img_probs=False, return_ob_progress=False)

        output = {}

        # prepare text tensor
        txt_ids = inputs['instr_encoding']
        output['txt_ids'] = torch.LongTensor(txt_ids)
        output['txt_lens'] = output['txt_ids'].size(0)

        # prepare history tensor
        output['hist_img_fts'] = torch.from_numpy(inputs['hist_img_fts'])
        output['hist_ang_fts'] = torch.from_numpy(inputs['hist_ang_fts'])

        if 'hist_pano_img_fts' in inputs:
            output['hist_pano_img_fts'] = torch.from_numpy(inputs['hist_pano_img_fts'])
            output['hist_pano_ang_fts'] = torch.from_numpy(inputs['hist_pano_ang_fts'])

        output['hist_lens'] = inputs['hist_lens']
        return output

def itm_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)
    batch['txt_masks'] = torch.BoolTensor(gen_seq_masks(batch['txt_lens']))
    batch['txt_lens'] = torch.LongTensor(batch['txt_lens'])

    # history batches
    batch['hist_img_fts'] = pad_tensors(batch['hist_img_fts'], lens=batch['hist_lens'], pad=0)
    batch['hist_ang_fts'] = pad_tensors(batch['hist_ang_fts'], lens=batch['hist_lens'], pad=0)
    if 'hist_pano_img_fts' in batch:
        batch['hist_pano_img_fts'] = pad_tensors(batch['hist_pano_img_fts'], lens=batch['hist_lens'], pad=0)
        batch['hist_pano_ang_fts'] = pad_tensors(batch['hist_pano_ang_fts'], lens=batch['hist_lens'], pad=0)

    batch['hist_lens'] = [x + 1 for x in batch['hist_lens']] # added a special token
    batch['hist_masks'] = torch.BoolTensor(gen_seq_masks(batch['hist_lens']))
    batch['hist_lens'] = torch.LongTensor(batch['hist_lens'])
    return batch

############### Single-step Action Prediction ###############
class SapDataset(Dataset):
    def __init__(self, nav_db, tok, random_kill_v, random_kill_a):
        '''Single Step Action Prediction'''
        self.nav_db = nav_db
        self.tok = tok
        self.random_kill_v = random_kill_v
        self.random_kill_a = random_kill_a

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

    def __len__(self):
        return len(self.nav_db.traj_step_refer)

    def __getitem__(self, i):
        i_traj, j_instr, t_cur = self.nav_db.traj_step_refer[i]

        inputs = self.nav_db.get_input(i_traj, j_instr, t_cur,
            return_ob=True, return_ob_action=True, 
            return_hist_img_probs=False, return_ob_progress=False)

        output = {}

        # prepare text tensor
        txt_ids = inputs['instr_encoding']
        output['txt_ids'] = torch.LongTensor(txt_ids)
        output['txt_lens'] = output['txt_ids'].size(0)

        # prepare vision tensor
        output['ob_img_fts'] = torch.from_numpy(inputs['ob_img_fts'])
        v_exists = True
        if random.random() < self.random_kill_v:
            output['ob_img_fts'][...] = 0
            v_exists = False
        output['ob_ang_fts'] = torch.from_numpy(inputs['ob_ang_fts'])
        if v_exists and random.random() < self.random_kill_a:
            output['ob_ang_fts'][...] = 0
        output['ob_nav_types'] = torch.LongTensor(inputs['ob_nav_types'])
        output['ob_lens'] = output['ob_img_fts'].size(0)

        # prepare action
        output['ob_action_viewindex'] = inputs['ob_action_viewindex']

        # prepare history tensor
        output['hist_img_fts'] = torch.from_numpy(inputs['hist_img_fts'])
        output['hist_ang_fts'] = torch.from_numpy(inputs['hist_ang_fts'])
        if 'hist_pano_img_fts' in inputs:
            output['hist_pano_img_fts'] = torch.from_numpy(inputs['hist_pano_img_fts'])
            output['hist_pano_ang_fts'] = torch.from_numpy(inputs['hist_pano_ang_fts'])
        output['hist_lens'] = inputs['hist_lens'] 
        return output

def sap_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)
    batch['txt_masks'] = torch.BoolTensor(gen_seq_masks(batch['txt_lens']))
    batch['txt_lens'] = torch.LongTensor(batch['txt_lens'])

    # image batches
    batch['ob_img_fts'] = pad_tensors(batch['ob_img_fts'], lens=batch['ob_lens'], pad=0)
    batch['ob_ang_fts'] = pad_tensors(batch['ob_ang_fts'], lens=batch['ob_lens'], pad=0)
    batch['ob_nav_types'] = pad_sequence(batch['ob_nav_types'], batch_first=True, padding_value=0)
    batch['ob_masks'] = torch.BoolTensor(gen_seq_masks(batch['ob_lens']))
    batch['ob_lens'] = torch.LongTensor(batch['ob_lens'])

    # history batches
    if max(batch['hist_lens']) == 0:
        # all are in first step
        batch['hist_img_fts'] = None
        batch['hist_ang_fts'] = None
        if 'hist_pano_img_fts' in batch:
            batch['hist_pano_img_fts'] = None
            batch['hist_pano_ang_fts'] = None
    else:
        batch['hist_img_fts'] = pad_tensors(batch['hist_img_fts'], lens=batch['hist_lens'], pad=0)
        batch['hist_ang_fts'] = pad_tensors(batch['hist_ang_fts'], lens=batch['hist_lens'], pad=0)
        if 'hist_pano_img_fts' in batch:
            batch['hist_pano_img_fts'] = pad_tensors(batch['hist_pano_img_fts'], lens=batch['hist_lens'], pad=0)
            batch['hist_pano_ang_fts'] = pad_tensors(batch['hist_pano_ang_fts'], lens=batch['hist_lens'], pad=0)
    batch['hist_lens'] = [x + 1 for x in batch['hist_lens']] # added a special token
    batch['hist_masks'] = torch.BoolTensor(gen_seq_masks(batch['hist_lens']))
    batch['hist_lens'] = torch.LongTensor(batch['hist_lens'])

    # action batches
    batch['ob_action_viewindex'] = torch.LongTensor(batch['ob_action_viewindex'])
    return batch


############### Single-step Action Regression ###############
class SarDataset(Dataset):
    def __init__(self, nav_db, tok, random_kill_v, random_kill_a):
        '''Single Step Action Regression'''
        self.nav_db = nav_db
        self.tok = tok
        self.random_kill_v = random_kill_v
        self.random_kill_a = random_kill_a

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

    def __len__(self):
        return len(self.nav_db.traj_step_refer)

    def __getitem__(self, i):
        i_traj, j_instr, t_cur = self.nav_db.traj_step_refer[i]
        
        inputs = self.nav_db.get_input(i_traj, j_instr, t_cur,
            return_ob=True, return_ob_action=True, 
            return_hist_img_probs=False, return_ob_progress=True)

        output = {}

        # prepare text tensor
        txt_ids = inputs['instr_encoding']
        output['txt_ids'] = torch.LongTensor(txt_ids)
        output['txt_lens'] = output['txt_ids'].size(0)

        # prepare vision tensor
        output['ob_img_fts'] = torch.from_numpy(inputs['ob_img_fts'])
        v_exists = True
        if random.random() < self.random_kill_v:
            output['ob_img_fts'][...] = 0
            v_exists = False
        output['ob_ang_fts'] = torch.from_numpy(inputs['ob_ang_fts'])
        if v_exists and random.random() < self.random_kill_a:
            output['ob_ang_fts'][...] = 0
        output['ob_nav_types'] = torch.LongTensor(inputs['ob_nav_types'])
        output['ob_lens'] = output['ob_img_fts'].size(0)

        # prepare action
        output['ob_action_angles'] = self._standardize_radians(inputs['ob_action_angles'])
        output['ob_progress'] = inputs['ob_progress']

        # prepare history tensor
        output['hist_img_fts'] = torch.from_numpy(inputs['hist_img_fts'])
        output['hist_ang_fts'] = torch.from_numpy(inputs['hist_ang_fts'])
        if 'hist_pano_img_fts' in inputs:
            output['hist_pano_img_fts'] = torch.from_numpy(inputs['hist_pano_img_fts'])
            output['hist_pano_ang_fts'] = torch.from_numpy(inputs['hist_pano_ang_fts'])
        output['hist_lens'] = inputs['hist_lens']

        return output

    def _standardize_radians(self, x):
        # [-pi, pi]
        x = np.mod(x, 2 * np.pi)
        x = np.where(x >= np.pi, x - 2 * np.pi, x)
        return  x

def sar_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)
    batch['txt_masks'] = torch.BoolTensor(gen_seq_masks(batch['txt_lens']))
    batch['txt_lens'] = torch.LongTensor(batch['txt_lens'])

    # image batches
    batch['ob_img_fts'] = pad_tensors(batch['ob_img_fts'], lens=batch['ob_lens'], pad=0)
    batch['ob_ang_fts'] = pad_tensors(batch['ob_ang_fts'], lens=batch['ob_lens'], pad=0)
    batch['ob_nav_types'] = pad_sequence(batch['ob_nav_types'], batch_first=True, padding_value=0)
    batch['ob_masks'] = torch.BoolTensor(gen_seq_masks(batch['ob_lens']))
    batch['ob_lens'] = torch.LongTensor(batch['ob_lens'])

    # history batches
    if max(batch['hist_lens']) == 0:
        # all are in first step
        batch['hist_img_fts'] = None
        batch['hist_ang_fts'] = None
        if 'hist_pano_img_fts' in batch:
            batch['hist_pano_img_fts'] = None
            batch['hist_pano_ang_fts'] = None
    else:
        batch['hist_img_fts'] = pad_tensors(batch['hist_img_fts'], lens=batch['hist_lens'], pad=0)
        batch['hist_ang_fts'] = pad_tensors(batch['hist_ang_fts'], lens=batch['hist_lens'], pad=0)
        if 'hist_pano_img_fts' in batch:
            batch['hist_pano_img_fts'] = pad_tensors(batch['hist_pano_img_fts'], lens=batch['hist_lens'], pad=0)
            batch['hist_pano_ang_fts'] = pad_tensors(batch['hist_pano_ang_fts'], lens=batch['hist_lens'], pad=0)
    batch['hist_lens'] = [x + 1 for x in batch['hist_lens']] # added a special token
    batch['hist_masks'] = torch.BoolTensor(gen_seq_masks(batch['hist_lens']))
    batch['hist_lens'] = torch.LongTensor(batch['hist_lens'])

    # action batches
    batch['ob_action_angles'] = torch.FloatTensor(batch['ob_action_angles'])
    batch['ob_progress'] = torch.FloatTensor(batch['ob_progress'])
    return batch


############### Spatial Relationship Regression ###############

class SprelDataset(Dataset):
    def __init__(self, nav_db, tok, random_kill_v, random_kill_a):
        '''Spatial Relationship Regression'''
        self.nav_db = nav_db
        self.tok = tok
        self.random_kill_v = random_kill_v
        self.random_kill_a = random_kill_a

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

        self.sp_targets = np.zeros((36, 36, 2))    # each row: anchor viewindex (to all views)
        for i in range(36):
            anchor_heading = (i % 12) * math.radians(30)
            anchor_elevation = (i // 12 - 1) * math.radians(30) 
            for j in range(36):
                cur_heading = (j % 12) * math.radians(30)
                cur_elevation = (j // 12 - 1) * math.radians(30)
                self.sp_targets[i, j] = self._standardize_radians(
                    [cur_heading - anchor_heading, cur_elevation - anchor_elevation])

    def __len__(self):
        return len(self.nav_db.traj_step_refer)

    def __getitem__(self, i):
        i_traj, j_instr, t_cur = self.nav_db.traj_step_refer[i]

        inputs = self.nav_db.get_input(i_traj, j_instr, t_cur, 
            return_ob=True, return_ob_action=False, 
            return_hist_img_probs=False, return_ob_progress=False,
            ob_cand_pano_view=False)

        output = {}

        # prepare text tensor
        txt_ids = inputs['instr_encoding']
        output['txt_ids'] = torch.LongTensor(txt_ids)
        output['txt_lens'] = output['txt_ids'].size(0)

        # prepare vision tensor
        output['ob_img_fts'] = torch.from_numpy(inputs['ob_img_fts'])
        v_exists = True
        if random.random() < self.random_kill_v:
            output['ob_img_fts'][...] = 0
            v_exists = False
        output['ob_ang_fts'] = torch.from_numpy(inputs['ob_ang_fts'])
        if v_exists and random.random() < self.random_kill_a:
            output['ob_ang_fts'][...] = 0
        output['ob_nav_types'] = torch.LongTensor(inputs['ob_nav_types'])
        output['ob_lens'] = output['ob_img_fts'].size(0)

        # prepare history tensor
        output['hist_img_fts'] = torch.from_numpy(inputs['hist_img_fts'])
        output['hist_ang_fts'] = torch.from_numpy(inputs['hist_ang_fts'])
        if 'hist_pano_img_fts' in inputs:
            output['hist_pano_img_fts'] = torch.from_numpy(inputs['hist_pano_img_fts'])
            output['hist_pano_ang_fts'] = torch.from_numpy(inputs['hist_pano_ang_fts'])
        output['hist_lens'] = inputs['hist_lens'] 

        # prepare labels
        sp_anchor_idx = np.random.randint(36)   # select a view as anchor
        output['sp_anchor_idxs'] = sp_anchor_idx
        output['sp_targets'] = self.sp_targets[sp_anchor_idx]
        
        return output

    def _standardize_radians(self, x):
        # [-pi, pi]
        x = np.mod(x, 2 * np.pi)
        x = np.where(x >= np.pi, x - 2 * np.pi, x)
        return  x

def sprel_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)
    batch['txt_masks'] = torch.BoolTensor(gen_seq_masks(batch['txt_lens']))
    batch['txt_lens'] = torch.LongTensor(batch['txt_lens'])

    # image batches
    batch['ob_img_fts'] = pad_tensors(batch['ob_img_fts'], lens=batch['ob_lens'], pad=0)
    batch['ob_ang_fts'] = pad_tensors(batch['ob_ang_fts'], lens=batch['ob_lens'], pad=0)
    batch['ob_nav_types'] = pad_sequence(batch['ob_nav_types'], batch_first=True, padding_value=0)
    batch['ob_masks'] = torch.BoolTensor(gen_seq_masks(batch['ob_lens']))
    batch['ob_lens'] = torch.LongTensor(batch['ob_lens'])

    # history batches
    if max(batch['hist_lens']) == 0:
        # all are in first step
        batch['hist_img_fts'] = None
        batch['hist_ang_fts'] = None
        if 'hist_pano_img_fts' in batch:
            batch['hist_pano_img_fts'] = None
            batch['hist_pano_ang_fts'] = None
    else:
        batch['hist_img_fts'] = pad_tensors(batch['hist_img_fts'], lens=batch['hist_lens'], pad=0)
        batch['hist_ang_fts'] = pad_tensors(batch['hist_ang_fts'], lens=batch['hist_lens'], pad=0)
        if 'hist_pano_img_fts' in batch:
            batch['hist_pano_img_fts'] = pad_tensors(batch['hist_pano_img_fts'], lens=batch['hist_lens'], pad=0)
            batch['hist_pano_ang_fts'] = pad_tensors(batch['hist_pano_ang_fts'], lens=batch['hist_lens'], pad=0)
    batch['hist_lens'] = [x + 1 for x in batch['hist_lens']] # added a special token
    batch['hist_masks'] = torch.BoolTensor(gen_seq_masks(batch['hist_lens']))
    batch['hist_lens'] = torch.LongTensor(batch['hist_lens'])

    # action batches
    batch['sp_anchor_idxs'] = torch.LongTensor(batch['sp_anchor_idxs'])
    batch['sp_targets'] = torch.FloatTensor(batch['sp_targets'])
    return batch

