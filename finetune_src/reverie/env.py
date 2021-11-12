''' Batched REVERIE navigation environment '''

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

from reverie.data_utils import get_obj_local_pos


class ReverieNavBatch(R2RBatch):
    def __init__(self, feat_db, instr_data, connectivity_dir, anno_dir,
        batch_size=64, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None,
        multi_endpoints=False, multi_startpoints=False):
        super().__init__(feat_db, instr_data, connectivity_dir, batch_size=batch_size,
          angle_feat_size=angle_feat_size, seed=seed, name=name, sel_data_idxs=sel_data_idxs)
        self.multi_endpoints = multi_endpoints
        self.multi_startpoints = multi_startpoints

        self.gt_trajs = {
            x['instr_id']: (x['scan'], x['path'], x['objId']) for x in self.data if 'objId' in x and x['objId'] is not None
        }   # to evaluate full data

        # load: the object can be viewed at which viewpoint
        self.obj2viewpoint = {}    # {scan_objid: viewpoint_list}
            
        bbox_data = json.load(open(os.path.join(anno_dir, 'BBoxes.json')))
        for scanvp, value in bbox_data.items():
            scan, vp = scanvp.split('_')
            # for all visible objects at that viewpoint
            for objid, objinfo in value.items():
                if objinfo['visible_pos']:
                    # if such object not already in the dict
                    self.obj2viewpoint.setdefault(scan+'_'+objid, [])
                    self.obj2viewpoint[scan+'_'+objid].append(vp)

    def _next_minibatch(self, batch_size=None, **kwargs):
        super()._next_minibatch(batch_size=batch_size, **kwargs)
        if self.multi_endpoints:
            batch = copy.deepcopy(self.batch)
            for item in batch:
                scan = item['scan']
                gt_objid = str(item['objId'])
                end_vps = self.obj2viewpoint['%s_%s'%(scan, gt_objid)]
                end_vp = np.random.choice(end_vps)
                start_vp = item['path'][0]
                if self.multi_startpoints:
                    start_vps = []
                    for cvp, cpath in self.shortest_paths[scan][end_vp].items():
                        if len(cpath) >= 4 and len(cpath) <= 7:
                            start_vps.append(cvp)
                    if len(start_vps) > 0:
                        start_vp = start_vps[np.random.randint(len(start_vps))]                        
                item['path'] = self.shortest_paths[scan][start_vp][end_vp]
            self.batch = batch

    def _get_obs(self, t=None, shortest_teacher=False):
        obs = super()._get_obs(t=t, shortest_teacher=shortest_teacher)
        for i, ob in enumerate(obs):
            if ob['instr_id'] in self.gt_trajs:
                # A3C reward. There are multiple gt end viewpoints on REVERIE. 
                gt_objid = self.gt_trajs[ob['instr_id']][-1]
                min_dist = np.inf
                for vp in self.obj2viewpoint['%s_%s'%(ob['scan'], str(gt_objid))]:
                    try:
                        min_dist = min(min_dist, self.shortest_distances[ob['scan']][ob['viewpoint']][vp])
                    except:
                        print(ob['scan'], ob['viewpoint'], vp)
                        exit(0)
                ob['distance'] = min_dist
            else:
                ob['distance'] = 0
        return obs

    ############### Nav Evaluation ###############
    def _eval_item(self, scan, path, gt_path, gt_objid):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])
        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        # navigation: success is to arrive to a viewpoint where the object is visible
        goal_viewpoints = set(self.obj2viewpoint['%s_%s'%(scan, str(gt_objid))])
        assert len(goal_viewpoints) > 0, '%s_%s'%(scan, str(gt_objid))

        scores['success'] = float(path[-1] in goal_viewpoints)
        scores['oracle_success'] = float(any(x in goal_viewpoints for x in path))
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = [x[0] for x in item['trajectory']]
            scan, gt_traj, gt_objid = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, gt_traj, gt_objid)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
        }
        return avg_metrics, metrics

class ReverieNavRefBatch(R2RBatch):
    def __init__(self, feat_db, obj_db, instr_data, connectivity_dir, anno_dir,
        batch_size=64, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None,
        multi_endpoints=False, multi_startpoints=False, max_objects=20):
        super().__init__(feat_db, instr_data, connectivity_dir, batch_size=batch_size,
            angle_feat_size=angle_feat_size, seed=seed, name=name, sel_data_idxs=sel_data_idxs)

        self.obj_db = obj_db
        self.max_objects = max_objects
        self.multi_endpoints = multi_endpoints
        self.multi_startpoints = multi_startpoints

        self.gt_trajs = {
            x['instr_id']: (x['scan'], x['path'], x['objId']) for x in self.data
        }   # to evaluate full data

        # load: the object can be viewed at which viewpoint
        self.obj2viewpoint = {}    # {scan_objid: viewpoint_list}
            
        bbox_data = json.load(open(os.path.join(anno_dir, 'BBoxes.json')))
        for scanvp, value in bbox_data.items():
            scan, vp = scanvp.split('_')
            # for all visible objects at that viewpoint
            for objid, objinfo in value.items():
                if objinfo['visible_pos']:
                    # if such object not already in the dict
                    self.obj2viewpoint.setdefault(scan+'_'+objid, [])
                    self.obj2viewpoint[scan+'_'+objid].append(vp)

    def _next_minibatch(self, batch_size=None, **kwargs):
        super()._next_minibatch(batch_size=batch_size, **kwargs)
        if self.multi_endpoints:
            batch = copy.deepcopy(self.batch)
            for item in batch:
                scan = item['scan']
                gt_objid = str(item['objId'])
                end_vps = self.obj2viewpoint['%s_%s'%(scan, gt_objid)]
                end_vp = np.random.choice(end_vps)
                start_vp = item['path'][0]
                if self.multi_startpoints:
                    start_vps = []
                    for cvp, cpath in self.shortest_paths[scan][end_vp].items():
                        if len(cpath) >= 4 and len(cpath) <= 7:
                            start_vps.append(cvp)
                    if len(start_vps) > 0:
                        start_vp = start_vps[np.random.randint(len(start_vps))]                        
                item['path'] = self.shortest_paths[scan][start_vp][end_vp]
            self.batch = batch

    def _get_obs(self, t=None, shortest_teacher=False):
        obs = super()._get_obs(t=t, shortest_teacher=shortest_teacher)
        for i, ob in enumerate(obs):
            item = self.batch[i]
            scan = ob['scan']
            viewpoint = ob['viewpoint']
            # object feature
            base_view_id = ob['viewIndex']
            directional_feature = self.angle_feature[base_view_id]
            scan_vp = '%s_%s' % (scan, viewpoint)
            if scan_vp in self.obj_db:
                obj_viewindexs = self.obj_db[scan_vp]['viewindexs']
                obj_angle_fts = np.vstack([directional_feature[vidx] for vidx in obj_viewindexs])

                obj_local_pos = get_obj_local_pos(self.obj_db[scan_vp]['bboxes']) # xywh
                obj_features = np.concatenate([self.obj_db[scan_vp]['fts'], obj_angle_fts], 1)
                candidate_objId = self.obj_db[scan_vp]['obj_ids']
            else:
                obj_local_pos, obj_features, candidate_objId = [], [], []
            ob.update({
                'id': item['id'],
                'objId': str(item['objId']),
                'candidate_obj': (obj_local_pos[:self.max_objects], obj_features[:self.max_objects], candidate_objId[:self.max_objects])
            })
                
            # A3C reward. There are multiple gt end viewpoints on REVERIE. 
            gt_objid = self.gt_trajs[ob['instr_id']][-1]
            if gt_objid is None:
                min_dist = 0
            else:
                min_dist = np.inf
                for vp in self.obj2viewpoint['%s_%s'%(scan, str(gt_objid))]:
                    min_dist = min(min_dist, self.shortest_distances[scan][viewpoint][vp])
            ob['distance'] = min_dist
        return obs

    ############### Nav Evaluation ###############
    def _eval_item(self, scan, path, gt_path, pred_objid, gt_objid):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])
        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        # navigation: success is to arrive to a viewpoint where the object is visible
        goal_viewpoints = set(self.obj2viewpoint['%s_%s'%(scan, str(gt_objid))])
        assert len(goal_viewpoints) > 0, '%s_%s'%(scan, str(gt_objid))

        scores['success'] = float(path[-1] in goal_viewpoints)
        scores['oracle_success'] = float(any(x in goal_viewpoints for x in path))
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

        # REF success
        pred_objid = str(pred_objid)
        gt_objid = str(gt_objid)
        scores['rgs'] = float(pred_objid == gt_objid)
        scores['rgspl'] = scores['rgs'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = [x[0] for x in item['trajectory']]
            scan, gt_traj, gt_objid = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, gt_traj, item['predObjId'], gt_objid)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'rgs': np.mean(metrics['rgs']) * 100,
            'rgspl': np.mean(metrics['rgspl']) * 100,
        }
        return avg_metrics, metrics
