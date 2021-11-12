import random
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from .data import pad_tensors, gen_seq_masks

from .mlm import random_word, MlmDataset
from .mrc import _get_img_mask, MrcDataset
from .sap import SapDataset
from .sar import SarDataset
from .sprel import SprelDataset
from .itm import ItmDataset


class MlmImageDataset(MlmDataset):
    def __getitem__(self, i):
        i_traj, j_instr, path_len = self.nav_db.traj_refer[i]

        inputs = self.nav_db.get_input(
            i_traj,
            j_instr,
            path_len,
            return_ob=False,
            return_ob_action=False,
            return_hist_img_probs=False,
            return_ob_progress=False,
        )

        output = {}

        # prepare text tensor
        txt_ids, txt_labels = random_word(
            inputs["instr_encoding"], self.vocab_range, self.mask_token_id
        )
        output["txt_ids"] = torch.LongTensor(txt_ids)
        output["txt_labels"] = torch.LongTensor(txt_labels)
        output["txt_lens"] = output["txt_ids"].size(0)

        # prepare history tensor
        output["hist_images"] = inputs["hist_images"]
        output["hist_ang_fts"] = torch.from_numpy(inputs["hist_ang_fts"])
        output["hist_pano_images"] = inputs["hist_pano_images"]
        output["hist_pano_ang_fts"] = torch.from_numpy(inputs["hist_pano_ang_fts"])
        output["hist_lens"] = inputs["hist_lens"]

        return output


def mlm_image_collate(inputs):
    batch = {k: [x[k] for x in inputs] for k in inputs[0].keys()}
    # text batches
    batch["txt_ids"] = pad_sequence(batch["txt_ids"], batch_first=True, padding_value=0)
    batch["txt_labels"] = pad_sequence(
        batch["txt_labels"], batch_first=True, padding_value=-1
    )
    batch["txt_masks"] = torch.BoolTensor(gen_seq_masks(batch["txt_lens"]))
    batch["txt_lens"] = torch.LongTensor(batch["txt_lens"])

    # history batches
    batch["hist_images"] = pad_tensors(
        batch["hist_images"], lens=batch["hist_lens"], pad=0
    )
    batch["hist_ang_fts"] = pad_tensors(
        batch["hist_ang_fts"], lens=batch["hist_lens"], pad=0
    )
    batch["hist_pano_images"] = pad_tensors(
        batch["hist_pano_images"], lens=batch["hist_lens"], pad=0
    )
    batch["hist_pano_ang_fts"] = pad_tensors(
        batch["hist_pano_ang_fts"], lens=batch["hist_lens"], pad=0
    )

    batch["hist_lens"] = [x + 1 for x in batch["hist_lens"]]  # added a global token
    batch["hist_masks"] = torch.BoolTensor(gen_seq_masks(batch["hist_lens"]))
    batch["hist_lens"] = torch.LongTensor(batch["hist_lens"])

    return batch


class MrcImageDataset(MrcDataset):
    def __getitem__(self, i):
        i_traj, j_instr, path_len = self.nav_db.traj_refer[i]

        inputs = self.nav_db.get_input(
            i_traj,
            j_instr,
            path_len,
            return_ob=False,
            return_ob_action=False,
            return_hist_img_probs=True,
            return_ob_progress=False,
        )

        output = {}

        # prepare text tensor
        txt_ids = inputs["instr_encoding"]
        output["txt_ids"] = torch.LongTensor(txt_ids)
        output["txt_lens"] = output["txt_ids"].size(0)

        # prepare history tensor: masked history image
        hist_mrc_masks = _get_img_mask(
            self.mask_prob, inputs["hist_img_probs"].shape[0]
        )  # (T, )
        # masked the feature after backbone (not now)
        output["hist_images"] = inputs["hist_images"]
        output["hist_pano_images"] = inputs["hist_pano_images"]

        output["hist_img_probs"] = torch.from_numpy(inputs["hist_img_probs"])
        output["hist_mrc_masks"] = hist_mrc_masks

        output["hist_ang_fts"] = torch.from_numpy(inputs["hist_ang_fts"])
        output["hist_pano_ang_fts"] = torch.from_numpy(inputs["hist_pano_ang_fts"])
        output["hist_lens"] = inputs["hist_lens"]

        return output


def mrc_image_collate(inputs):
    batch = {k: [x[k] for x in inputs] for k in inputs[0].keys()}
    # text batches
    batch["txt_ids"] = pad_sequence(batch["txt_ids"], batch_first=True, padding_value=0)
    batch["txt_masks"] = torch.BoolTensor(gen_seq_masks(batch["txt_lens"]))
    batch["txt_lens"] = torch.LongTensor(batch["txt_lens"])

    # history batches
    batch["hist_images"] = pad_tensors(
        batch["hist_images"], lens=batch["hist_lens"], pad=0
    )
    batch["hist_ang_fts"] = pad_tensors(
        batch["hist_ang_fts"], lens=batch["hist_lens"], pad=0
    )

    batch["hist_pano_images"] = pad_tensors(
        batch["hist_pano_images"], lens=batch["hist_lens"], pad=0
    )
    batch["hist_pano_ang_fts"] = pad_tensors(
        batch["hist_pano_ang_fts"], lens=batch["hist_lens"], pad=0
    )

    # labels
    batch["hist_mrc_masks"] = pad_sequence(
        batch["hist_mrc_masks"], batch_first=True, padding_value=0
    )
    batch["hist_img_probs"] = pad_tensors(
        batch["hist_img_probs"], lens=batch["hist_lens"], pad=0
    )

    batch["hist_lens"] = [x + 1 for x in batch["hist_lens"]]  # added a global token
    batch["hist_masks"] = torch.BoolTensor(gen_seq_masks(batch["hist_lens"]))
    batch["hist_lens"] = torch.LongTensor(batch["hist_lens"])

    return batch


class SapImageDataset(SapDataset):
    def __getitem__(self, i):
        i_traj, j_instr, t_cur = self.nav_db.traj_step_refer[i]

        inputs = self.nav_db.get_input(
            i_traj,
            j_instr,
            t_cur,
            return_ob=True,
            return_ob_action=True,
            return_hist_img_probs=False,
            return_ob_progress=False,
        )

        output = {}

        # prepare text tensor
        txt_ids = inputs["instr_encoding"]
        output["txt_ids"] = torch.LongTensor(txt_ids)
        output["txt_lens"] = output["txt_ids"].size(0)

        # prepare vision tensor
        output["ob_images"] = inputs["ob_images"]
        v_exists = True
        if random.random() < self.random_kill_v:
            output["ob_images"][...] = 0
            v_exists = False
        output["ob_v_exists"] = v_exists
        output["ob_ang_fts"] = torch.from_numpy(inputs["ob_ang_fts"])
        if v_exists and random.random() < self.random_kill_a:
            output["ob_ang_fts"][...] = 0
        output["ob_nav_types"] = torch.LongTensor(inputs["ob_nav_types"])
        output["ob_lens"] = output["ob_images"].size(0) + 1  # STOP

        # prepare action
        output["ob_action_viewindex"] = inputs["ob_action_viewindex"]

        # prepare history tensor
        output["hist_images"] = inputs["hist_images"]
        output["hist_ang_fts"] = torch.from_numpy(inputs["hist_ang_fts"])
        output["hist_pano_images"] = inputs["hist_pano_images"]
        output["hist_pano_ang_fts"] = torch.from_numpy(inputs["hist_pano_ang_fts"])
        output["hist_lens"] = inputs["hist_lens"]

        return output


def sap_image_collate(inputs):
    batch = {k: [x[k] for x in inputs] for k in inputs[0].keys()}
    # text batches
    batch["txt_ids"] = pad_sequence(batch["txt_ids"], batch_first=True, padding_value=0)
    batch["txt_masks"] = torch.BoolTensor(gen_seq_masks(batch["txt_lens"]))
    batch["txt_lens"] = torch.LongTensor(batch["txt_lens"])

    # image batches
    batch["ob_images"] = pad_tensors(
        batch["ob_images"], lens=[x - 1 for x in batch["ob_lens"]], pad=0
    )
    batch["ob_v_exists"] = torch.BoolTensor(batch["ob_v_exists"])
    batch["ob_ang_fts"] = pad_tensors(batch["ob_ang_fts"], lens=batch["ob_lens"], pad=0)
    batch["ob_nav_types"] = pad_sequence(
        batch["ob_nav_types"], batch_first=True, padding_value=0
    )
    batch["ob_masks"] = torch.BoolTensor(gen_seq_masks(batch["ob_lens"]))
    batch["ob_lens"] = torch.LongTensor(batch["ob_lens"])

    # history batches
    if max(batch["hist_lens"]) == 0:
        # all are in first step
        batch["hist_images"] = None
        batch["hist_anages"] = None
        batch["hist_pano_images"] = None
        batch["hist_pano_ang_fts"] = None
    else:
        batch["hist_images"] = pad_tensors(
            batch["hist_images"], lens=batch["hist_lens"], pad=0
        )
        batch["hist_ang_fts"] = pad_tensors(
            batch["hist_ang_fts"], lens=batch["hist_lens"], pad=0
        )
        batch["hist_pano_images"] = pad_tensors(
            batch["hist_pano_images"], lens=batch["hist_lens"], pad=0
        )
        batch["hist_pano_ang_fts"] = pad_tensors(
            batch["hist_pano_ang_fts"], lens=batch["hist_lens"], pad=0
        )
    batch["hist_lens"] = [x + 1 for x in batch["hist_lens"]]  # added a global token
    batch["hist_masks"] = torch.BoolTensor(gen_seq_masks(batch["hist_lens"]))
    batch["hist_lens"] = torch.LongTensor(batch["hist_lens"])

    # action batches
    batch["ob_action_viewindex"] = torch.LongTensor(batch["ob_action_viewindex"])

    return batch


class SarImageDataset(SarDataset):
    def __getitem__(self, i):
        i_traj, j_instr, t_cur = self.nav_db.traj_step_refer[i]

        inputs = self.nav_db.get_input(
            i_traj,
            j_instr,
            t_cur,
            return_ob=True,
            return_ob_action=True,
            return_hist_img_probs=False,
            return_ob_progress=True,
        )

        output = {}

        # prepare text tensor
        txt_ids = inputs["instr_encoding"]
        output["txt_ids"] = torch.LongTensor(txt_ids)
        output["txt_lens"] = output["txt_ids"].size(0)

        # prepare vision tensor
        output["ob_images"] = inputs["ob_images"]
        v_exists = True
        if random.random() < self.random_kill_v:
            output["ob_images"][...] = 0
            v_exists = False
        output["ob_v_exists"] = v_exists
        output["ob_ang_fts"] = torch.from_numpy(inputs["ob_ang_fts"])
        if v_exists and random.random() < self.random_kill_a:
            output["ob_ang_fts"][...] = 0
        output["ob_nav_types"] = torch.LongTensor(inputs["ob_nav_types"])
        output["ob_lens"] = output["ob_images"].size(0) + 1  # STOP

        # prepare action
        output["ob_action_angles"] = self._standardize_radians(
            inputs["ob_action_angles"]
        )
        output["ob_progress"] = inputs["ob_progress"]

        # prepare history tensor
        output["hist_images"] = inputs["hist_images"]
        output["hist_ang_fts"] = torch.from_numpy(inputs["hist_ang_fts"])
        output["hist_pano_images"] = inputs["hist_pano_images"]
        output["hist_pano_ang_fts"] = torch.from_numpy(inputs["hist_pano_ang_fts"])
        output["hist_lens"] = inputs["hist_lens"]

        return output


def sar_image_collate(inputs):
    batch = {k: [x[k] for x in inputs] for k in inputs[0].keys()}
    # text batches
    batch["txt_ids"] = pad_sequence(batch["txt_ids"], batch_first=True, padding_value=0)
    batch["txt_masks"] = torch.BoolTensor(gen_seq_masks(batch["txt_lens"]))
    batch["txt_lens"] = torch.LongTensor(batch["txt_lens"])

    # image batches
    batch["ob_images"] = pad_tensors(
        batch["ob_images"], lens=[x - 1 for x in batch["ob_lens"]], pad=0
    )
    batch["ob_v_exists"] = torch.BoolTensor(batch["ob_v_exists"])
    batch["ob_ang_fts"] = pad_tensors(batch["ob_ang_fts"], lens=batch["ob_lens"], pad=0)
    batch["ob_nav_types"] = pad_sequence(
        batch["ob_nav_types"], batch_first=True, padding_value=0
    )
    batch["ob_masks"] = torch.BoolTensor(gen_seq_masks(batch["ob_lens"]))
    batch["ob_lens"] = torch.LongTensor(batch["ob_lens"])

    # history batches
    if max(batch["hist_lens"]) == 0:
        # all are in first step
        batch["hist_images"] = None
        batch["hist_anages"] = None
        batch["hist_pano_images"] = None
        batch["hist_pano_ang_fts"] = None
    else:
        batch["hist_images"] = pad_tensors(
            batch["hist_images"], lens=batch["hist_lens"], pad=0
        )
        batch["hist_ang_fts"] = pad_tensors(
            batch["hist_ang_fts"], lens=batch["hist_lens"], pad=0
        )
        batch["hist_pano_images"] = pad_tensors(
            batch["hist_pano_images"], lens=batch["hist_lens"], pad=0
        )
        batch["hist_pano_ang_fts"] = pad_tensors(
            batch["hist_pano_ang_fts"], lens=batch["hist_lens"], pad=0
        )
    batch["hist_lens"] = [x + 1 for x in batch["hist_lens"]]  # added a global token
    batch["hist_masks"] = torch.BoolTensor(gen_seq_masks(batch["hist_lens"]))
    batch["hist_lens"] = torch.LongTensor(batch["hist_lens"])

    # action batches
    batch["ob_action_angles"] = torch.FloatTensor(batch["ob_action_angles"])
    batch["ob_progress"] = torch.FloatTensor(batch["ob_progress"])

    return batch


class SprelImageDataset(SprelDataset):
    def __getitem__(self, i):
        i_traj, j_instr, t_cur = self.nav_db.traj_step_refer[i]

        inputs = self.nav_db.get_input(
            i_traj,
            j_instr,
            t_cur,
            return_ob=True,
            return_ob_action=False,
            return_hist_img_probs=False,
            return_ob_progress=False,
        )

        output = {}

        # prepare text tensor
        txt_ids = inputs["instr_encoding"]
        output["txt_ids"] = torch.LongTensor(txt_ids)
        output["txt_lens"] = output["txt_ids"].size(0)

        # prepare vision tensor
        output["ob_images"] = inputs["ob_images"]
        v_exists = True
        if random.random() < self.random_kill_v:
            output["ob_images"][...] = 0
            v_exists = False
        output["ob_v_exists"] = v_exists
        output["ob_ang_fts"] = torch.from_numpy(inputs["ob_ang_fts"])
        if v_exists and random.random() < self.random_kill_a:
            output["ob_ang_fts"][...] = 0
        output["ob_nav_types"] = torch.LongTensor(inputs["ob_nav_types"])
        output["ob_lens"] = output["ob_images"].size(0) + 1  # STOP

        # prepare history tensor
        output["hist_images"] = inputs["hist_images"]
        output["hist_ang_fts"] = torch.from_numpy(inputs["hist_ang_fts"])
        output["hist_pano_images"] = inputs["hist_pano_images"]
        output["hist_pano_ang_fts"] = torch.from_numpy(inputs["hist_pano_ang_fts"])
        output["hist_lens"] = inputs["hist_lens"]

        # prepare labels
        sp_anchor_idx = np.random.randint(36)  # select a view as anchor
        output["sp_anchor_idxs"] = sp_anchor_idx
        output["sp_targets"] = self.sp_targets[sp_anchor_idx]

        return output


def sprel_image_collate(inputs):
    batch = {k: [x[k] for x in inputs] for k in inputs[0].keys()}
    # text batches
    batch["txt_ids"] = pad_sequence(batch["txt_ids"], batch_first=True, padding_value=0)
    batch["txt_masks"] = torch.BoolTensor(gen_seq_masks(batch["txt_lens"]))
    batch["txt_lens"] = torch.LongTensor(batch["txt_lens"])

    # image batches
    batch["ob_images"] = pad_tensors(
        batch["ob_images"], lens=[x - 1 for x in batch["ob_lens"]], pad=0
    )
    batch["ob_v_exists"] = torch.BoolTensor(batch["ob_v_exists"])
    batch["ob_ang_fts"] = pad_tensors(batch["ob_ang_fts"], lens=batch["ob_lens"], pad=0)
    batch["ob_nav_types"] = pad_sequence(
        batch["ob_nav_types"], batch_first=True, padding_value=0
    )
    batch["ob_masks"] = torch.BoolTensor(gen_seq_masks(batch["ob_lens"]))
    batch["ob_lens"] = torch.LongTensor(batch["ob_lens"])

    # history batches
    if max(batch["hist_lens"]) == 0:
        # all are in first step
        batch["hist_images"] = None
        batch["hist_ang_fts"] = None
        batch["hist_pano_images"] = None
        batch["hist_pano_ang_fts"] = None
    else:
        batch["hist_images"] = pad_tensors(
            batch["hist_images"], lens=batch["hist_lens"], pad=0
        )
        batch["hist_ang_fts"] = pad_tensors(
            batch["hist_ang_fts"], lens=batch["hist_lens"], pad=0
        )
        batch["hist_pano_images"] = pad_tensors(
            batch["hist_pano_images"], lens=batch["hist_lens"], pad=0
        )
        batch["hist_pano_ang_fts"] = pad_tensors(
            batch["hist_pano_ang_fts"], lens=batch["hist_lens"], pad=0
        )
    batch["hist_lens"] = [x + 1 for x in batch["hist_lens"]]  # added a global token
    batch["hist_masks"] = torch.BoolTensor(gen_seq_masks(batch["hist_lens"]))
    batch["hist_lens"] = torch.LongTensor(batch["hist_lens"])

    # action batches
    batch["sp_anchor_idxs"] = torch.LongTensor(batch["sp_anchor_idxs"])
    batch["sp_targets"] = torch.FloatTensor(batch["sp_targets"])

    return batch


class ItmImageDataset(ItmDataset):
    def __getitem__(self, i):
        i_traj, j_instr, path_len = self.nav_db.traj_refer[i]

        inputs = self.nav_db.get_input(
            i_traj,
            j_instr,
            path_len,
            return_ob=False,
            return_ob_action=False,
            return_hist_img_probs=False,
            return_ob_progress=False,
        )

        output = {}

        # prepare text tensor
        txt_ids = inputs["instr_encoding"]
        output["txt_ids"] = torch.LongTensor(txt_ids)
        output["txt_lens"] = output["txt_ids"].size(0)

        # prepare history tensor
        output["hist_images"] = inputs["hist_images"]
        output["hist_ang_fts"] = torch.from_numpy(inputs["hist_ang_fts"])
        output["hist_pano_images"] = inputs["hist_pano_images"]
        output["hist_pano_ang_fts"] = torch.from_numpy(inputs["hist_pano_ang_fts"])
        output["hist_lens"] = inputs["hist_lens"]

        return output


def itm_image_collate(inputs):
    batch = {k: [x[k] for x in inputs] for k in inputs[0].keys()}
    # text batches
    batch["txt_ids"] = pad_sequence(batch["txt_ids"], batch_first=True, padding_value=0)
    batch["txt_masks"] = torch.BoolTensor(gen_seq_masks(batch["txt_lens"]))
    batch["txt_lens"] = torch.LongTensor(batch["txt_lens"])

    # history batches
    batch["hist_images"] = pad_tensors(
        batch["hist_images"], lens=batch["hist_lens"], pad=0
    )
    batch["hist_ang_fts"] = pad_tensors(
        batch["hist_ang_fts"], lens=batch["hist_lens"], pad=0
    )
    batch["hist_pano_images"] = pad_tensors(
        batch["hist_pano_images"], lens=batch["hist_lens"], pad=0
    )
    batch["hist_pano_ang_fts"] = pad_tensors(
        batch["hist_pano_ang_fts"], lens=batch["hist_lens"], pad=0
    )

    batch["hist_lens"] = [x + 1 for x in batch["hist_lens"]]  # added a global token
    batch["hist_masks"] = torch.BoolTensor(gen_seq_masks(batch["hist_lens"]))
    batch["hist_lens"] = torch.LongTensor(batch["hist_lens"])

    return batch
