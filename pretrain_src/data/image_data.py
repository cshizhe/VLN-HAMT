"""
R2R-style dataset: load images
"""
import os
import json
import jsonlines
import numpy as np
import h5py
import math
import lmdb
import networkx as nx
from PIL import Image

import torch

from timm.data.transforms_factory import create_transform

from .data import angle_feature, softmax, MultiStepNavData

HEIGHT = 248
WIDTH = 330
IMGSIZE = 224


class MultiStepNavImageData(MultiStepNavData):
    """
    Load image features stored on an LMDB file.
    """

    def __init__(
        self,
        traj_files,
        img_db_file,
        img_ft_file,
        scanvp_cands_file,
        connectivity_dir,
        auto_augment=None,
        re_mode="const",
        re_prob=0.0,
        is_training=False,
        image_prob_size=1000,
        image_feat_size=2048,
        angle_feat_size=4,
        max_txt_len=80,
        in_memory=False,
    ):
        super().__init__(
            traj_files,
            img_ft_file,
            scanvp_cands_file,
            connectivity_dir,
            image_prob_size=image_prob_size,
            image_feat_size=image_feat_size,
            angle_feat_size=angle_feat_size,
            max_txt_len=max_txt_len,
            in_memory=in_memory,
        )
        self.img_db_file = img_db_file

        self.img_db_env = lmdb.open(
            self.img_db_file,
            map_size=int(1e12),
            readonly=True,
            create=False,
            readahead=False,
            max_readers=2000,
        )
        self.img_db_txn = self.img_db_env.begin()

        self.img_transform = create_transform(
            (3, 224, 224),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
            crop_pct=0.9,
            is_training=is_training,
            auto_augment=auto_augment,
            re_mode=re_mode,
            re_prob=re_prob,
        )

        if not is_training:
            # cannot evaluate all the samples as it takes too much time
            # sample K data for validation
            self.traj_step_refer, self.traj_refer = [], []
            sel_idxs = np.random.permutation(len(self.traj_data))
            for sel_idx in sel_idxs:
                item = self.traj_data[sel_idx]
                path_len = len(item["path"])
                j_rand = np.random.randint(len(item["instr_encodings"]))
                t_rand = np.random.randint(path_len)
                self.traj_refer.append((sel_idx, j_rand, path_len))
                self.traj_step_refer.append((sel_idx, j_rand, t_rand))

    def __del__(self):
        self.img_db_env.close()

    def get_input(
        self,
        i_path,
        j_instr,
        t_cur,
        return_ob=False,
        return_hist_img_probs=False,
        return_ob_action=False,
        return_ob_progress=False,
    ):

        traj_data = self.traj_data[i_path]
        scan = traj_data["scan"]
        path = traj_data["path"]
        path_viewindex = traj_data["path_viewindex"]
        action_viewindex = traj_data["action_viewindex"]
        abs_pos_angles = traj_data["abs_pos_angles"]
        rel_act_angles = traj_data["rel_act_angles"]

        instr_id = traj_data["instr_ids"][j_instr]
        instr_encoding = traj_data["instr_encodings"][j_instr][: self.max_txt_len]

        hist_inputs = self.get_history_feature(
            scan,
            path,
            path_viewindex,
            rel_act_angles,
            t_cur,
            return_img_probs=return_hist_img_probs,
        )

        outs = {
            "instr_id": instr_id,
            "instr_encoding": instr_encoding,
            "hist_images": hist_inputs[0],
            "hist_ang_fts": hist_inputs[1],
            "hist_pano_images": hist_inputs[2],
            "hist_pano_ang_fts": hist_inputs[3],
            "hist_lens": t_cur,
        }
        if return_hist_img_probs:
            outs["hist_img_probs"] = hist_inputs[4]

        if return_ob:
            ob_images = self.get_image(scan, path[t_cur])  # NO STOP TOKEN
            ob_ang_feats = self.get_angle_feature(
                path_viewindex[t_cur], pad_stop_token=True
            )
            ob_len = ob_images.shape[0] + 1  # add STOP

            ob_nav_types = np.zeros((ob_len,), dtype=np.int64)
            ob_nav_types[-1] = 2  # 2 for [STOP]
            ob_nav_cands = self.scanvp_cands["%s_%s" % (scan, path[t_cur])]
            ob_nav_viewindexes = np.array(list(ob_nav_cands.values()))
            ob_nav_types[ob_nav_viewindexes] = 1

            outs.update(
                {
                    "ob_images": ob_images,
                    "ob_ang_fts": ob_ang_feats,
                    "ob_nav_types": ob_nav_types,
                }
            )
            if return_ob_action:
                if action_viewindex[t_cur] != -1:
                    outs["ob_action_viewindex"] = action_viewindex[t_cur]
                    outs["ob_action_angles"] = rel_act_angles[t_cur]
                else:  # stop
                    outs["ob_action_viewindex"] = ob_len - 1
                    outs["ob_action_angles"] = np.zeros((2,), dtype=np.float32)
            if return_ob_progress:
                outs["ob_progress"] = self.get_progress(
                    scan, path[0], path[t_cur], path[-1]
                )

        return outs

    def get_history_feature(
        self, scan, path, path_viewindex, rel_act_angles, t_cur, return_img_probs=False
    ):
        # get history features before the step t_cur
        images, angle_feats, image_probs = [], [], []
        pano_images, pano_angle_feats = [], []

        for t in range(0, t_cur):
            vp = path[t]
            viewidx = path_viewindex[t]
            heading, elevation = rel_act_angles[t]

            if t == len(path) - 1:  # STOP Action
                angle_feats.append(np.zeros((self.angle_feat_size,), dtype=np.float32))
            else:
                angle_feats.append(
                    angle_feature(heading, elevation, self.angle_feat_size)
                )

            t_pano_images = self.get_image(scan, vp)
            t_image = t_pano_images[viewidx]

            images.append(t_image)
            pano_images.append(t_pano_images)
            pano_angle_feats.append(self.angle_features[viewidx])

            if return_img_probs:
                image_probs.append(self.get_image_feature(scan, vp)[viewidx])

        if t_cur > 0:
            images = torch.stack(images, 0)  # (T, 3, H, W)
            angle_feats = np.stack(angle_feats)
            pano_images = torch.stack(pano_images, 0)  # (T, P, 3, H, W)
            pano_angle_feats = np.stack(pano_angle_feats, 0)

            if return_img_probs:
                image_probs = np.stack(image_probs, 0)
                image_probs = softmax(image_probs)
        else:
            images = torch.zeros(0, 3, IMGSIZE, IMGSIZE, dtype=torch.float)
            angle_feats = np.zeros((0, self.angle_feat_size), dtype=np.float32)
            pano_images = torch.zeros(0, 36, 3, IMGSIZE, IMGSIZE, dtype=torch.float)
            pano_angle_feats = np.zeros((0, 36, self.angle_feat_size), dtype=np.float32)
            image_probs = np.zeros((0, self.image_prob_size), dtype=np.float32)

        if return_img_probs:
            return images, angle_feats, pano_images, pano_angle_feats, image_probs

        return images, angle_feats, pano_images, pano_angle_feats

    def get_image(self, scan, viewpoint):
        key = "%s_%s" % (scan, viewpoint)
        buf = self.img_db_txn.get(key.encode("ascii"))

        images = np.frombuffer(buf, dtype=np.uint8)
        images = images.reshape(36, HEIGHT, WIDTH, 3)  # fixed image size

        # (36, 3, IMGSIZE, IMGSIZE)
        images = torch.stack(
            [self.img_transform(Image.fromarray(image)) for image in images], 0
        )

        return images

    def get_image_feature(self, scan, viewpoint, pad_stop_token=False):
        # only need to get prob feature
        key = "%s_%s" % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            fts = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, "r") as f:
                fts = f[key][...].astype(np.float32)[:, -self.image_prob_size :]
            if self.in_memory:
                self._feature_store[key] = fts

        if pad_stop_token:
            fts = np.vstack([fts, np.zeros((1, fts.shape[-1]), dtype=fts.dtype)])
        return fts

    def get_angle_feature(self, viewindex, pad_stop_token=False):
        fts = self.angle_features[viewindex]
        if pad_stop_token:
            fts = np.vstack([fts, np.zeros((1, fts.shape[-1]), dtype=fts.dtype)])
        return fts

    def get_progress(self, scan, start_vp, cur_vp, end_vp):
        if cur_vp == end_vp:
            return 1
        elif start_vp == cur_vp:
            return 0
        else:
            total_dist = self.shortest_distances[scan][start_vp][end_vp]
            remained_dist = self.shortest_distances[scan][cur_vp][end_vp]
            return 1 - remained_dist / total_dist
