import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.misc import length2mask

from r2r.agent_cmt import Seq2SeqCMTAgent
from reverie.model_navref import NavRefModel, Critic


class NavRefCMTAgent(Seq2SeqCMTAgent):

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v, 'predObjId': r} for k, (v, r) in self.results.items()]
        return output

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()

        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout():
                    self.loss = 0
                    self.results[traj['instr_id']] = (traj['path'], traj['predObjId'])
        else:   # Do a full round
            while True:
                for traj in self.rollout():
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = (traj['path'], traj['predObjId'])
                if looped:
                    break

    def _build_model(self):
        self.vln_bert = NavRefModel(self.args).cuda()
        self.critic = Critic(self.args).cuda()

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) for ob in obs]
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types = [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types = [], [], []
            cand_pointids = np.zeros((self.args.views, ), dtype=np.bool)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)
            
            # add pano context
            pano_fts = ob['feature'][~cand_pointids]
            cand_pano_img_fts = np.concatenate([cand_img_fts, pano_fts[:, :self.args.image_feat_size]], 0)
            cand_pano_ang_fts = np.concatenate([cand_ang_fts, pano_fts[:, self.args.image_feat_size:]], 0)
            cand_nav_types.extend([0] * (self.args.views - np.sum(cand_pointids)))

            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)

        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()

        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens

    def _candidate_variable(self, obs):
        cand_lens = [len(ob['candidate']) for ob in obs]
        max_len = max(cand_lens)
        cand_img_feats = np.zeros((len(obs), max_len, self.args.image_feat_size), dtype=np.float32)
        cand_ang_feats = np.zeros((len(obs), max_len, self.args.angle_feat_size), dtype=np.float32)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                cand_img_feats[i, j] = cc['feature'][:self.feature_size]
                cand_ang_feats[i, j] = cc['feature'][self.feature_size:]
                cand_nav_types[i, j] = 1

        cand_img_feats = torch.from_numpy(cand_img_feats).cuda()
        cand_ang_feats = torch.from_numpy(cand_ang_feats).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()
        return cand_img_feats, cand_ang_feats, cand_nav_types, cand_lens

    def _object_variable(self, obs):
        obj_lens = [max(len(ob['candidate_obj'][2]), 1) for ob in obs] # in case no object in a vp
        obj_feats = np.zeros((len(obs), max(obj_lens), self.args.obj_feat_size + self.args.angle_feat_size), dtype=np.float32)
        obj_poses = np.zeros((len(obs), max(obj_lens), 5), dtype=np.float32)

        for i, ob in enumerate(obs):
            obj_local_pos, obj_features, candidate_objId = ob['candidate_obj']
            if len(obj_features) > 0:
                obj_feats[i, :obj_lens[i]] = obj_features
                obj_poses[i, :obj_lens[i]] = obj_local_pos

        obj_angles = torch.from_numpy(obj_feats[..., -self.args.angle_feat_size:]).cuda()
        obj_feats = torch.from_numpy(obj_feats[..., :-self.args.angle_feat_size]).cuda()
        obj_poses = torch.from_numpy(obj_poses).cuda()
        return obj_feats, obj_angles, obj_poses, obj_lens

    def _teacher_action(self, obs, ended, ob_img_max_len):
        # navigate or select an object as stop
        a = np.zeros(len(obs), dtype=np.int64)
        ref = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
                ref[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['teacher']:   # select object
                    a[i] = ob_img_max_len
                    candidate_objs = ob['candidate_obj'][2]
                    for k, kid in enumerate(candidate_objs):
                        if str(kid) == str(ob['objId']):
                            ref[i] = k
                            break
                    else:
                        ref[i] = self.args.ignoreid
                else:   # navigate
                    ref[i] = self.args.ignoreid
                    for k, candidate in enumerate(ob['candidate']):
                        if candidate['viewpointId'] == ob['teacher']:   # Next view point
                            a[i] = k
                            break
        return torch.from_numpy(a).cuda(), torch.from_numpy(ref).cuda()

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        batch_size = len(obs)

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)
        
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'predObjId': str(None),
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size) # mark the stop action

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        ref_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]

        hist_embeds = [self.vln_bert('history').expand(batch_size, -1)]  # global embedding
        hist_lens = [1 for _ in range(batch_size)]

        for t in range(self.args.max_action_len):
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()
            
            obj_feats, obj_angles, obj_poses, obj_lens = self._object_variable(obs)
            obj_masks = length2mask(obj_lens).logical_not()
            
            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'obj_feats': obj_feats,
                'obj_poses': obj_poses,
                'obj_angles': obj_angles,
                'obj_masks': obj_masks,
                'return_states': True if self.feedback == 'sample' else False,
            }
            ob_img_max_len = ob_img_feats.size(1)

            t_outputs = self.vln_bert(**visual_inputs)
            act_logits = t_outputs['act_logits']
            obj_logits = t_outputs['obj_logits']
            _, max_obj_logits = torch.max(obj_logits, 1)
            act_logits = torch.cat([act_logits, max_obj_logits.unsqueeze(1)], 1)
            
            if self.feedback == 'sample':
                h_t = t_outputs['states']
                hidden_states.append(h_t)

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:
                bt_masks = torch.zeros(act_logits.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                act_logits.masked_fill_(bt_masks, -float('inf'))

            if train_ml is not None:
                # Supervised training
                target, ref_target = self._teacher_action(obs, ended, ob_img_max_len)
                ml_loss += self.criterion(act_logits, target)
                ref_loss += self.criterion(obj_logits, ref_target)
            
            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = act_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(act_logits, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(act_logits, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if ((next_id >= ob_img_max_len) or (t == self.args.max_action_len-1)) and (not ended[i]):  # just stopped and forced stopped
                    if len(obs[i]['candidate_obj'][2]) == 0:
                        traj[i]['predObjId'] = str(None)
                    else:
                        _, ref_t = obj_logits[i, :obj_lens[i]].max(0)
                        traj[i]['predObjId'] = obs[i]['candidate_obj'][2][ref_t]

                if (next_id >= ob_img_max_len) or (next_id == self.args.ignoreid) or (ended[i]):    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # get history input embeddings
            hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
            prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
            for i, next_id in enumerate(cpu_a_t):
                if next_id != -1:
                    prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
            prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

            t_hist_inputs = {
                'mode': 'history',
                'hist_img_feats': hist_img_feats,
                'hist_ang_feats': prev_act_angle,
                'hist_pano_img_feats': hist_pano_img_feats,
                'hist_pano_ang_feats': hist_pano_ang_feats,
                'ob_step': t,
            }
            t_hist_embeds = self.vln_bert(**t_hist_inputs)
            hist_embeds.append(t_hist_embeds)

            for i, i_ended in enumerate(ended):
                if not i_ended:
                    hist_lens[i] += 1
                
            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs()

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:                              # If the action now is end
                            if dist[i] == 0.:                             # Correct
                                reward[i] = 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            if reward[i] > 0.0:                           # Quantification
                                reward[i] = 1.0
                            elif reward[i] < 0.0:
                                reward[i] = -1.0
                            else:
                                reward[i] = 0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()
            
            obj_feats, obj_angles, obj_poses, obj_lens = self._object_variable(obs)
            obj_masks = length2mask(obj_lens).logical_not()
            
            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'obj_feats': obj_feats,
                'obj_poses': obj_poses,
                'obj_angles': obj_angles,
                'obj_masks': obj_masks,
                'return_states': True,
            }
            last_h_ = self.vln_bert(**visual_inputs)['states']
            
            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5 # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item()) # critic loss + policy loss + entropy loss

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size + ref_loss / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())
            self.logs['REF_loss'].append(ref_loss / batch_size)

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj
