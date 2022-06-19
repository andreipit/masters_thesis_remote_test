from utils.arg.model import ArgsModel
from utils.trainer.model import TrainerModel

from utils.trainer.fwd import TrainerFwd
from utils.trainer.goal_fwd import TrainerGoalFwd

import numpy as np

class TrainerLabelValue(object):

    # grasp reward is rate of sucessfully-grasping
    def get_label_value(self, fwd: TrainerFwd, gfwd: TrainerGoalFwd, a: ArgsModel, m: TrainerModel, 
        primitive_action,  grasp_success, 
        grasp_reward, improved_grasp_reward, 
        change_detected, next_color_heightmap, 
        next_depth_heightmap, next_goal_mask_heightmap=None, 
        goal_catched=0, decreased_occupy_ratio=0
    ):

        if a.stage == 'grasp_only':
            return self._grasp_only(fwd, gfwd, a, m, grasp_success, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, goal_catched)
        elif self.stage == 'push_only':
            return self._push_only(fwd, gfwd, a, m, primitive_action, improved_grasp_reward, change_detected, decreased_occupy_ratio,
                grasp_success, grasp_reward, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap)
        elif self.stage == 'push_grasp':
            return self._push_grasp(fwd, gfwd, a, m, primitive_action, change_detected, grasp_success, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap)

    def _grasp_only(self, fwd: TrainerFwd, gfwd: TrainerGoalFwd, a: ArgsModel, m: TrainerModel,
        grasp_success, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap,
        goal_catched
    ):
        # Compute current reward
        current_reward = 0
        if grasp_success:
            current_reward = 1.0

        # Compute future reward
        if not grasp_success:
            future_reward = 0
        else:
            if not a.grasp_goal_conditioned:
                next_push_predictions, next_grasp_predictions, next_state_feat = fwd.forward(a, m, next_color_heightmap, next_depth_heightmap, is_volatile=True)
            else:
                next_push_predictions, next_grasp_predictions, next_state_feat = gfwd.goal_forward(a, m, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, is_volatile=True)
            future_reward = np.max(next_grasp_predictions)

        print('Current reward: %f' % (current_reward))
        print('Future reward: %f' % (future_reward))
        if goal_catched == 1:
            expected_reward = current_reward + m.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (current_reward, m.future_reward_discount, future_reward, expected_reward))
        else:
            expected_reward = m.future_reward_discount * future_reward
            print('Expected reward: %f x %f = %f' % (m.future_reward_discount, future_reward, expected_reward))
        return expected_reward, current_reward
        
    def _push_only(self, fwd: TrainerFwd, gfwd: TrainerGoalFwd, a: ArgsModel, m: TrainerModel,
        primitive_action, improved_grasp_reward, change_detected, decreased_occupy_ratio,
        grasp_success, grasp_reward, next_color_heightmap, next_depth_heightmap,
        next_goal_mask_heightmap
    ):
        # Compute current reward
        if primitive_action == 'push':
            if improved_grasp_reward > 0.1 and change_detected and decreased_occupy_ratio > 0.1:
                # change of reward after pushing
                current_reward = 0.5
            elif not change_detected or not decreased_occupy_ratio > 0.1:
                current_reward = -0.5
            else:
                current_reward = 0
            print('improved grasp reward after pushing in trainer:', improved_grasp_reward)

        elif primitive_action == 'grasp':
            if grasp_success:
                current_reward = 1.5
            else:
                current_reward = 0
                if a.alternating_training:
                    current_reward = -1.0

        # Compute future reward
        if improved_grasp_reward <= 0.1 and grasp_reward < 0.5 and not grasp_success:
            future_reward = 0
        else:
            if not a.grasp_goal_conditioned:
                next_push_predictions, next_grasp_predictions, next_state_feat = fwd.forward(a, m, next_color_heightmap, next_depth_heightmap, is_volatile=True)
            else:
                next_push_predictions, next_grasp_predictions, next_state_feat = gfwd.goal_forward(a, m, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, is_volatile=True)

            future_reward = np.max(next_push_predictions)

        print('Current reward: %f' % (current_reward))
        print('Future reward: %f' % (future_reward))
            
        if primitive_action == 'push':
            expected_reward = current_reward + m.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (current_reward, m.future_reward_discount, future_reward, expected_reward))
        else:
            expected_reward = current_reward + m.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (current_reward, m.future_reward_discount, future_reward, expected_reward))

        return expected_reward, current_reward

    
    def _push_grasp(self, fwd: TrainerFwd, gfwd: TrainerGoalFwd, a: ArgsModel, m: TrainerModel,
        primitive_action, change_detected, grasp_success, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap
    ):
        # Compute current reward
        current_reward = 0
        if primitive_action == 'push':
            if change_detected:
                current_reward = 0.5
        elif primitive_action == 'grasp':
            if grasp_success:
                current_reward = 1.0

        # Compute future reward
        if not change_detected and not grasp_success:
            future_reward = 0
        else:
            if not a.grasp_goal_conditioned:
                next_push_predictions, next_grasp_predictions, next_state_feat = fwd.forward(a, m, next_color_heightmap, next_depth_heightmap, is_volatile=True)
            else:
                next_push_predictions, next_grasp_predictions, next_state_feat = gfwd.goal_forward(a, m, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, is_volatile=True)

            future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))

        print('Current reward: %f' % (current_reward))
        print('Future reward: %f' % (future_reward))
        if primitive_action == 'push':
            expected_reward = m.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (0.0, m.future_reward_discount, future_reward, expected_reward))
        else:
            expected_reward = current_reward + m.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (current_reward, m.future_reward_discount, future_reward, expected_reward))
        return expected_reward, current_reward
