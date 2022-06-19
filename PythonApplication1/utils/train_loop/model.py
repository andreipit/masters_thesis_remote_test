class MainloopModel(object):


    # MainloopMakePhoto.get_rgb()
    iteration_time_0 = None
    color_img = None
    depth_img = None

    # MainloopMakePhoto.get_heightmap()
    #color_heightmap = None
    #depth_heightmap = None
    #valid_depth_heightmap = None
    #goal_mask_heightmap = None

    # MainloopMakePhoto.save_rgb_heightmap()
    exit_called = None

    # MainloopAction.execute()
    grasp_explore_actions = None
    push_predictions = None
    grasp_predictions = None
    state_feat = None
    mask = None

    # MainloopRunner
    label_value = None
    prev_reward_value = None

    # MainloopThreadsSync.sync
    prev_depth_heightmap = None
    prev_primitive_action = None
    prev_grasp_success = None
    prev_grasp_reward = None
    prev_improved_grasp_reward = None
    change_detected = None
    prev_color_heightmap = None
    prev_valid_depth_heightmap = None
    prev_best_pix_ind = None
    prev_goal_mask_heightmap = None


    # MainloopExpReplay.run
    sample_primitive_action = None
    sample_goal_obj_idx = None
    sample_primitive_action_id = None
    sample_reward_value = None

    

    # MainloopExpReplay.get_samples
    sample_ind = None

    # MainloopExpReplay.get_highest_sample
    sample_iteration = None
    sample_color_heightmap = None
    sample_depth_heightmap = None
    sample_goal_mask_heightmap = None

    
    # MainloopExpReplay.fwd_pass
    sample_push_predictions = None
    sample_grasp_predictions = None
    sample_state_feat = None
    #useless:
    prev_push_success = None
    prev_push_predictions = None
    prev_grasp_predictions = None