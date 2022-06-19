import numpy as np
from utils.arg.model import ArgsModel
from utils.trainer.model import TrainerModel

from utils.trainer.fwd import TrainerFwd
from utils.trainer.goal_fwd import TrainerGoalFwd

import torch
from torch.autograd import Variable


class TrainerBackprop(object):
    
    # Compute labels and backpropagate
    def backprop(self, fwd: TrainerFwd, gfwd: TrainerGoalFwd, a: ArgsModel, m: TrainerModel, 
        color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap=None):

        if a.stage == 'grasp_only':
            loss_value = self._grasp_only(fwd, gfwd, a, m, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap)
        elif a.stage == 'push_only':
            loss_value = self._push_only(fwd, gfwd, a, m, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap)
        elif a.stage == 'push_grasp':
            # Compute labels
            label = np.zeros((1,320,320))
            action_area = np.zeros((224,224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            tmp_label = np.zeros((224,224))
            tmp_label[action_area > 0] = label_value
            label[0,48:(320-48),48:(320-48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224,224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

            # Compute loss and backward pass
            m.optimizer.zero_grad()
            loss_value = 0
            if primitive_action == 'push':
                loss_value = self._push_grasp_push(label, label_weights,
                    color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap)

            elif primitive_action == 'grasp':
                loss_value = self._push_grasp_grasp(label, label_weights,
                    color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap)

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()
        return loss_value

    def _grasp_only(self, fwd: TrainerFwd, gfwd: TrainerGoalFwd, a: ArgsModel, m: TrainerModel, 
        color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap=None):

        # Compute labels (here all vars are local except label_value - it is arg)
        label = np.zeros((1,320,320))
        action_area = np.zeros((224,224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)
        tmp_label = np.zeros((224,224))
        tmp_label[action_area > 0] = label_value
        label[0,48:(320-48),48:(320-48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224,224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

        # Compute loss and backward pass
        m.optimizer.zero_grad()
        loss_value = 0

        # Do forward pass with specified rotation (to save gradients)
        if not a.grasp_goal_conditioned:
            push_predictions, grasp_predictions, state_feat = fwd.forward(a, m, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
        else:
            push_predictions, grasp_predictions, state_feat = gfwd.goal_forward(a, m, color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

        if m.use_cuda:
            loss = m.criterion(m.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
        else:
            loss = m.criterion(m.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(), requires_grad=False)
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()

        opposite_rotate_idx = (best_pix_ind[0] + m.model.num_rotations/2) % m.model.num_rotations

        if not a.grasp_goal_conditioned:
            push_predictions, grasp_predictions, state_feat = fwd.forward(a, m, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)
        else:
            push_predictions, grasp_predictions, state_feat = gfwd.goal_forward(a, m, color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)

        if m.use_cuda:
            loss = m.criterion(m.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
        else:
            loss = m.criterion(m.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()

        loss_value = loss_value/2

        print('Training loss: %f' % (loss_value))
        m.optimizer.step()
        return loss_value

    def _push_only(self, fwd: TrainerFwd, gfwd: TrainerGoalFwd, a: ArgsModel, m: TrainerModel, 
        color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap=None):

        
        # Compute labels
        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320-48), 48:(320-48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

        # Compute loss and backward pass
        m.optimizer.zero_grad()
        loss_value = 0
        # Do forward pass with specified rotation (to save gradients)
        if not a.grasp_goal_conditioned:
            push_predictions, grasp_predictions, state_feat = fwd.forward(a, m, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
        else:
            push_predictions, grasp_predictions, state_feat = gfwd.goal_forward(a, m, color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

        if m.use_cuda:
            loss = m.criterion(m.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
        else:
            loss = m.criterion(m.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        print('Training loss: %f' % (loss_value))
        m.optimizer.step()
        return loss_value

    def _push_grasp_push(self, label, label_weights,
        fwd: TrainerFwd, gfwd: TrainerGoalFwd, a: ArgsModel, m: TrainerModel, 
        color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap=None):

        # Do forward pass with specified rotation (to save gradients)
        if not a.grasp_goal_conditioned:
            push_predictions, grasp_predictions, state_feat = fwd.forward(a, m, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
        else:
            push_predictions, grasp_predictions, state_feat = gfwd.goal_forward(a, m, color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

        if m.use_cuda:
            loss = m.criterion(m.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
        else:
            loss = m.criterion(m.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        return loss_value

    def _push_grasp_grasp(self, label, label_weights,
        fwd: TrainerFwd, gfwd: TrainerGoalFwd, a: ArgsModel, m: TrainerModel, 
        color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap=None):

        
        # Do forward pass with specified rotation (to save gradients)
        if not a.grasp_goal_conditioned:
            push_predictions, grasp_predictions, state_feat = fwd.forward(a, m, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
        else:
            push_predictions, grasp_predictions, state_feat = gfwd.goal_forward(a, m, color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

        if m.use_cuda:
            loss = m.criterion(m.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
        else:
            loss = m.criterion(m.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()

        opposite_rotate_idx = (best_pix_ind[0] + m.model.num_rotations/2) % m.model.num_rotations

        if not a.grasp_goal_conditioned:
            push_predictions, grasp_predictions, state_feat = fwd.forward(a, m, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)
        else:
            push_predictions, grasp_predictions, state_feat = gfwd.goal_forward(a, m, color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)


        if m.use_cuda:
            loss = m.criterion(m.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
        else:
            loss = m.criterion(m.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()

        loss_value = loss_value/2
        return loss_value



