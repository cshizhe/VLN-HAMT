''' Batched NDH navigation environment '''

import json
import os
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict
import copy

import MatterSim

from r2r.data_utils import load_nav_graphs
from r2r.data_utils import new_simulator
from r2r.data_utils import angle_feature, get_all_point_angle_feature
from r2r.env import R2RBatch


class NDHNavBatch(R2RBatch):
    def __init__(self, feat_db, instr_data, connectivity_dir,
        batch_size=64, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None,
        use_player_path=False):
        super().__init__(feat_db, instr_data, connectivity_dir, batch_size=batch_size,
            angle_feat_size=angle_feat_size, seed=seed, name=name, sel_data_idxs=sel_data_idxs)
        self.use_player_path = use_player_path

    def _get_gt_trajs(self, data):
        return {x['instr_id']: (x['scan'], x['end_panos']) for x in data if 'end_panos' in x}

    def _next_minibatch(self, batch_size=None, **kwargs):
        super()._next_minibatch(batch_size=batch_size, **kwargs)
        batch = copy.deepcopy(self.batch)
        for item in batch:
            scan = item['scan']
            if 'end_panos' in item:
                use_player_path = self.use_player_path and np.random.rand() > 0.5
                if use_player_path:
                    item['path'] = item['nav_steps'][item['nav_idx']:]
                else:
                    end_pano = np.random.choice(item['end_panos'])
                    item['path'] = self.shortest_paths[scan][item['start_pano']][end_pano]
            else:
                item['path'] = [item['start_pano']]
            item['heading'] = item['start_heading']
        self.batch = batch

    def _get_obs(self, t=None, shortest_teacher=False):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            if feature is None:
                feature = np.zeros((36, 2048))

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            scan = state.scanId
            viewpoint = state.location.viewpointId

            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : scan,
                'viewpoint' : viewpoint,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instr_encoding': item['instr_encoding'],
                'teacher' : self._teacher_path_action(state, item['path'], t=t, shortest_teacher=shortest_teacher),
                'gt_path' : item['path'],
            })
            
            # A3C reward. There are multiple gt end viewpoints on REVERIE. 
            if 'end_panos' in item:
                min_dist = np.inf
                for end_pano in item['end_panos']:
                    min_dist = min(min_dist, self.shortest_distances[scan][viewpoint][end_pano])
            else:
                min_dist = 0
            obs[-1]['distance'] = min_dist
        return obs

    ############### Nav Evaluation ###############
    def _eval_item(self, scan, path, end_panos):
        scores = {}

        start_pano = path[0]
        end_panos = set(end_panos)
        shortest_distances = self.shortest_distances[scan]

        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])
        gt_lengths = np.min([shortest_distances[start_pano][end_pano] for end_pano in end_panos])

        # navigation: success is to arrive to a viewpoint in end_panos
        scores['success'] = float(path[-1] in end_panos)
        scores['oracle_success'] = float(any(x in end_panos for x in path))
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['gp'] = gt_lengths - np.min([shortest_distances[path[-1]][end_pano] for end_pano in end_panos])

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = [x[0] for x in item['trajectory']]
            scan, end_panos = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, end_panos)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'gp': np.mean(metrics['gp']),
        }
        return avg_metrics, metrics

