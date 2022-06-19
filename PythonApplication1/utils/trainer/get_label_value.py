class TrainerGetLabelValue(object):
    
    # grasp reward is rate of sucessfully-grasping
    def get_label_value(self, primitive_action,  grasp_success, grasp_reward, improved_grasp_reward, change_detected, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap=None, goal_catched=0, decreased_occupy_ratio=0):

        if self.stage == 'grasp_only':
            # Compute current reward
            current_reward = 0
            if grasp_success:
                current_reward = 1.0

            # Compute future reward
            if not grasp_success:
                future_reward = 0
            else:
                if not self.grasp_goal_conditioned:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                else:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.goal_forward(next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, is_volatile=True)
                future_reward = np.max(next_grasp_predictions)

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if goal_catched == 1:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f x %f = %f' % (self.future_reward_discount, future_reward, expected_reward))
            return expected_reward, current_reward
        
        elif self.stage == 'push_only':

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
                    if self.alternating_training:
                        current_reward = -1.0

            # Compute future reward
            if improved_grasp_reward <= 0.1 and grasp_reward < 0.5 and not grasp_success:
                future_reward = 0
            else:
                if not self.grasp_goal_conditioned:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                else:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.goal_forward(next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, is_volatile=True)

                future_reward = np.max(next_push_predictions)

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            
            if primitive_action == 'push':
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))

            return expected_reward, current_reward
            
        elif self.stage == 'push_grasp':
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
                if not self.grasp_goal_conditioned:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                else:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.goal_forward(next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, is_volatile=True)

                future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if primitive_action == 'push':
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (0.0, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
            return expected_reward, current_reward





