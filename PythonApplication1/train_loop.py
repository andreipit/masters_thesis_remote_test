import time
from matplotlib import pyplot as plt

# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

# helpers
from utils.train_loop.make_photo import MainloopMakePhoto
from utils.train_loop.restart import MainloopRestart
from utils.train_loop.action import MainloopAction
from utils.train_loop.runner import MainloopRunner
from utils.train_loop.backprop import MainloopBackprop
from utils.train_loop.exploration import MainloopExploration
from utils.train_loop.exp_replay import MainloopExpReplay
from utils.train_loop.snapshot import MainloopSnapshot
from utils.train_loop.threads_sync import MainloopThreadsSync

class TrainLoop(object):
    def __init__(self):
        pass

    def create_empty_helpers(self):
        # empty init START:
        self.m = MainloopModel() # empty init, just table
        self.photo = MainloopMakePhoto()
        self.restarter = MainloopRestart()
        self.action = MainloopAction()
        self.runner = MainloopRunner()
        self.backprop = MainloopBackprop()
        self.exploration = MainloopExploration()
        self.exp_replay = MainloopExpReplay()
        self.snapshot = MainloopSnapshot()
        self.threads = MainloopThreadsSync()
        # empty init END:

    def iterate(self, m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc, debug_mode = True) -> bool:
        if debug_mode: print('main 0')
        self.photo.get_rgb(m,a,r,l,t,p) # 2 photos from persp: m.color_img, m.depth_img
        #plt.imshow(tl.m.color_img); plt.show(block=True)
        #plt.imshow(tl.m.depth_img); plt.show(block=True)

        self.photo.get_heightmap(m,a,r,l,t,p)
        #plt.imshow( p.m.color_heightmap); plt.show(block=True) # ortho rgb with shadows
        #plt.imshow( p.m.depth_heightmap); plt.show(block=True) # ortho depth: 4 colors: bg, shadow(white), yel, blue
        #plt.imshow( p.m.valid_depth_heightmap); plt.show(block=True) # ortho: depth_heightmap without shadows
        #plt.imshow( p.m.goal_mask_heightmap); plt.show(block=True) # ortho: only goad obj yellow

        if debug_mode: print('main 1')
        if not self.photo.save_rgb_heightmap(m,a,r,l,t,p):
            return True #continue

        if debug_mode: print('main 2')
        if not self.restarter.restart(m,a,r,l,t,p):
            return True #continue

        if debug_mode: print('main 3')
        # set nonlocal: push_predictions, grasp_predictions, executing_action = True
        self.action.execute(m,a,r,l,t,p)

        if debug_mode: print('main 4')
        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():
            if debug_mode: print('main 5') # missing in 1.2
            change_detected = self.runner.detect_changes(m,a,r,l,t,p)
            self.runner.compute_labels(m,a,r,l,t,p, change_detected)
            self.backprop.run(m,a,r,l,t,p)
            self.exploration.run(m,a,r,l,t,p)
            if a.experience_replay and not a.is_testing:
                if debug_mode: print('main 6') # missing in 1.2
                self.exp_replay.run(m,a,r,l,t,p)
                self.exp_replay.get_samples(m,a,r,l,t,p)

                if m.sample_ind.size > 0:
                    self.exp_replay.get_highest_sample(m,a,r,l,t,p)
                    self.exp_replay.fwd_pass(m,a,r,l,t,p)
                else:
                    print('Not enough prior training samples. Skipping experience replay.')
            if debug_mode: print('main 7') # missing in 1.2
            print('----------------snapshot.run')
            self.snapshot.run(m,a,r,l,t,p)

        # Sync both action thread and training thread
        if debug_mode: print('main 8')

        while p.m.nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if debug_mode: print('main 9')
        if m.exit_called:
            return False #break

        if debug_mode: print('main 10')
        prev_color_img = self.threads.save(m,a,r,l,t,p) # Save information for next training step

        if debug_mode: print('main 11')

        return True



"""
    while True:
--------------------------------------------------
self.photo.get_rgb()
--------------------------------------------------

        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

--------------------------------------------------
self.photo.get_heightmap()
--------------------------------------------------
        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        if goal_conditioned:
            if is_testing and not random_scene_testing:
                obj_contour = robot.get_test_obj_mask(nonlocal_variables['goal_obj_idx'])
            else:
                obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
            goal_mask_heightmap = np.zeros(color_heightmap.shape[:2], np.uint8)
            goal_mask_heightmap = utils.get_goal_mask(obj_contour, goal_mask_heightmap, workspace_limits, heightmap_resolution)
            kernel = np.ones((3,3))
            nonlocal_variables['border_occupy_ratio'] = utils.get_occupy_ratio(goal_mask_heightmap, depth_heightmap)
            writer.add_scalar('border_occupy_ratio', nonlocal_variables['border_occupy_ratio'], trainer.iteration)

--------------------------------------------------
self.photo.save_rgb_heightmap()
--------------------------------------------------
        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')
                
        writer.add_image('goal_mask_heightmap', cv2.cvtColor(goal_mask_heightmap, cv2.COLOR_BGR2RGB), global_step=trainer.iteration, walltime=None, dataformats='HWC')
        logger.save_visualizations(trainer.iteration, goal_mask_heightmap, 'mask')
        cv2.imwrite('visualization.mask.png', goal_mask_heightmap)
        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold or (no_change_count[0] + no_change_count[1] > 10):
            no_change_count = [0, 0]
            if np.sum(stuff_count) < empty_threshold:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
            elif no_change_count[0] + no_change_count[1] > 10:
                print('Too many no change counts (value: %d)! Repositioning objects.' % (no_change_count[0] + no_change_count[1]))

            robot.restart_sim()
            robot.add_objects()
            if is_testing: # If at end of test run, re-load original weights (before test run)
                trainer.model.load_state_dict(torch.load(snapshot_file))

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)
            continue

--------------------------------------------------
self.restarter.restart()
--------------------------------------------------
        # Restart for push_only stage and goal-conditioned case
        if nonlocal_variables['push_step'] == max_push_episode_length + 1 or nonlocal_variables['new_episode_flag'] == 1 or nonlocal_variables['restart_scene'] == robot.num_obj / 2:
            nonlocal_variables['push_step'] = 0  # reset push step
            nonlocal_variables['new_episode_flag'] = 0
            # save episode_improved_grasp_reward
            print('episode %d begins' % nonlocal_variables['episode'])
            if nonlocal_variables['restart_scene'] == robot.num_obj / 2: # If at end of test run, re-load original weights (before test run)
                nonlocal_variables['restart_scene'] = 0
                robot.restart_sim()
                robot.add_objects()
                if is_testing: # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)
            continue

--------------------------------------------------
self.action.execute()
execute action (in another thread)
--------------------------------------------------
        trainer.push_step_log.append([nonlocal_variables['push_step']])
        logger.write_to_log('push-step', trainer.push_step_log)              

        if not exit_called:
            if stage == 'grasp_only' and grasp_explore:
                grasp_explore_actions = np.random.uniform() < grasp_explore_prob
                print('Strategy: explore (exploration probability: %f)' % (grasp_explore_prob))
                if grasp_explore_actions:
                    # Run forward pass with network to get affordances
                    push_predictions, grasp_predictions, state_feat = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True, grasp_explore_actions=True)
                    obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
                    mask = np.zeros(color_heightmap.shape[:2], np.uint8)
                    mask = utils.get_goal_mask(obj_contour, mask, workspace_limits, heightmap_resolution)
                    obj_grasp_prediction = np.multiply(grasp_predictions, mask)
                    grasp_predictions = obj_grasp_prediction / 255
                else:
                    push_predictions, grasp_predictions, state_feat = trainer.goal_forward(color_heightmap, valid_depth_heightmap, goal_mask_heightmap, is_volatile=True)

            else:
                if not grasp_goal_conditioned:
                    push_predictions, grasp_predictions, state_feat = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True)
                else:
                    push_predictions, grasp_predictions, state_feat = trainer.goal_forward(color_heightmap, valid_depth_heightmap, goal_mask_heightmap, is_volatile=True)
            
            nonlocal_variables['push_predictions'] = push_predictions
            nonlocal_variables['grasp_predictions'] = grasp_predictions

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

--------------------------------------------------
self.runner.run()
trainig runner (in current thread)
--------------------------------------------------
        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():
            --------------------------------------------------
            self.runner.detect_changes()
            Detect changes
            --------------------------------------------------
            # Detect changes
            if not goal_conditioned:
                depth_diff = abs(depth_heightmap - prev_depth_heightmap)
                change_threshold = 300
                change_value = utils.get_change_value(depth_diff)
                change_detected = change_value > change_threshold or prev_grasp_success
                print('Change detected: %r (value: %d)' % (change_detected, change_value))
            else:
                prev_mask_hull = binary_dilation(convex_hull_image(prev_goal_mask_heightmap), iterations=5)
                depth_diff = prev_mask_hull*(prev_depth_heightmap-depth_heightmap)
                change_threshold = 50
                change_value = utils.get_change_value(depth_diff)
                change_detected = change_value > change_threshold
                print('Goal change detected: %r (value: %d)' % (change_detected, change_value)) 

            if change_detected:
                if prev_primitive_action == 'push':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] = 0
            else:
                if prev_primitive_action == 'push':
                    no_change_count[0] += 1
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] += 1

            --------------------------------------------------
            self.runner.compute_labels()
            Compute training labels
            --------------------------------------------------
            # Compute training labels
            if not grasp_goal_conditioned:
                label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_grasp_success, prev_grasp_reward, prev_improved_grasp_reward, change_detected, color_heightmap, valid_depth_heightmap)
            else:
                label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_grasp_success, prev_grasp_reward, prev_improved_grasp_reward, change_detected, color_heightmap, valid_depth_heightmap, 
                goal_mask_heightmap, nonlocal_variables['goal_catched'], nonlocal_variables['decreased_occupy_ratio'])

            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)

            --------------------------------------------------
            self.backprop.run()
            # Backpropagate
            --------------------------------------------------
            # Backpropagate
            if not grasp_goal_conditioned:
                loss = trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value)
            else:
                loss = trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value, prev_goal_mask_heightmap)
            writer.add_scalar('loss', loss, trainer.iteration)

            episode_loss += loss
            if nonlocal_variables['push_step'] == max_push_episode_length or nonlocal_variables['new_episode_flag'] == 1:
                writer.add_scalar('episode loss', episode_loss, nonlocal_variables['episode'])
                episode_loss = 0
            
            --------------------------------------------------
            self.exploration.run()
            --------------------------------------------------
            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.5 * np.power(0.9998, trainer.iteration),0.1) if explore_rate_decay else 0.5
                grasp_explore_prob = max(0.8 * np.power(0.998, trainer.iteration),0.1) if explore_rate_decay else 0.8

            --------------------------------------------------
            self.exp_replay.run()
            experience replay
            --------------------------------------------------
            # Do sampling for experience replay
            if experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if grasp_goal_conditioned:
                    sample_goal_obj_idx = nonlocal_variables['goal_obj_idx']
                    print('sample_goal_obj_idx', sample_goal_obj_idx)
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                    sample_reward_value = 0 if prev_reward_value == 0.5 else 0.5
                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    sample_reward_value = 0 if prev_reward_value == 1 else 1

                --------------------------------------------------
                self.exp_replay.get_samples()
                Get samples
                --------------------------------------------------
                # Get samples of the same primitive but with different results
                if not grasp_goal_conditioned or sample_primitive_action == 'push':
                    sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[0:trainer.iteration,0] == sample_reward_value, np.asarray(trainer.executed_action_log)[0:trainer.iteration,0] == sample_primitive_action_id))
                else:
                    sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[0:trainer.iteration,0] == sample_reward_value, 
                    np.asarray(trainer.grasp_obj_log)[0:trainer.iteration,0] == sample_goal_obj_idx))
                   
                if sample_ind.size > 0:
                    --------------------------------------------------
                    self.exp_replay.get_highest_sample()
                    Get sample with highest surprise
                    --------------------------------------------------
                    print('reward_value_log:', np.asarray(trainer.reward_value_log)[sample_ind[:,0], 0])
                    # Find sample with highest surprise value
                    sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - np.asarray(trainer.label_value_log)[sample_ind[:,0]])
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000
                    
                    if grasp_goal_conditioned or goal_conditioned:
                        if is_testing and not random_scene_testing:
                            obj_contour = robot.get_test_obj_mask(nonlocal_variables['goal_obj_idx'])
                        else:
                            obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
                        sample_goal_mask_heightmap = np.zeros(color_heightmap.shape[:2], np.uint8)
                        sample_goal_mask_heightmap = utils.get_goal_mask(obj_contour, sample_goal_mask_heightmap, workspace_limits, heightmap_resolution)
                        writer.add_image('goal_mask_heightmap', cv2.cvtColor(sample_goal_mask_heightmap, cv2.COLOR_BGR2RGB), global_step=trainer.iteration, walltime=None, dataformats='HWC')

                    --------------------------------------------------
                    self.exp_replay.fwd_pass
                    forward pass
                    --------------------------------------------------
                    # Compute forward pass with sample
                    with torch.no_grad():
                        if not grasp_goal_conditioned:
                            sample_push_predictions, sample_grasp_predictions, sample_state_feat = trainer.forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)
                        else:
                            sample_push_predictions, sample_grasp_predictions, sample_state_feat = trainer.goal_forward(sample_color_heightmap, sample_depth_heightmap, sample_goal_mask_heightmap, is_volatile=True)

                    sample_grasp_success = sample_reward_value == 1
                    # Get labels for sample and backpropagate
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration,1:4]).astype(int)
                    if not grasp_goal_conditioned:  
                        trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration])
                    else:
                        trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration], sample_goal_mask_heightmap)

                    # Recompute prediction value and label for replay buffer
                    if sample_primitive_action == 'push':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
                    elif sample_primitive_action == 'grasp':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]

                else:
                    print('Not enough prior training samples. Skipping experience replay.')

            --------------------------------------------------
            self.snapshot.run()
            --------------------------------------------------
            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, stage)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.iteration, trainer.model, stage)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

--------------------------------------------------
sync threads: action + training
--------------------------------------------------

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_push_predictions = push_predictions.copy()
        prev_grasp_predictions = grasp_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        prev_grasp_reward = nonlocal_variables['grasp_reward']
        if grasp_goal_conditioned or goal_conditioned:
            prev_goal_mask_heightmap = goal_mask_heightmap.copy()
        if stage == 'push_only':
            prev_improved_grasp_reward = nonlocal_variables['improved_grasp_reward']
            prev_grasp_reward = nonlocal_variables['grasp_reward']
        else:
            prev_improved_grasp_reward = 0.0

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))

"""