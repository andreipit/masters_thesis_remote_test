

class ProcModel():
    debug_mode:bool = False

    # created at main train loop, then 
    color_heightmap = None # used in goal_conditioned case
    valid_depth_heightmap = None # used in proc_action_pos
    goal_mask_heightmap = None # used in proc_pixel_pos
    depth_heightmap = None # used in proc_pixel_pos

    # created at init:
    episode_loss = None
    no_change_count = None
    explore_prob = None
    grasp_explore_prob = None
    grasp_explore = None
    nonlocal_variables = None
    timestamp = None
    timestamp_value = None
    tensor_logging_directory = None
    writer = None

    # created at processor.process_actions():
    change_detected = None

    # created at processor.process_goal_coditioned()
    push_predictions = None
    grasp_predictions = None
    best_push_conf = None
    best_grasp_conf = None

    # created at proc.proc_goal_cond()
    exit_called = False # used later insidde 

    # created at proc_pixel_pos
    primitive_position = None
    best_rotation_angle = None
    prev_single_predictions = None
    prev_occupy_ratio = None

    
## Initialize episode loss
#episode_loss = 0

## Initialize variables for heuristic bootstrapping and exploration probability
#no_change_count = [2, 2] if not is_testing else [0, 0]
#explore_prob = 0.5 if not is_testing else 0.0
#grasp_explore_prob = 0.8 if not is_testing else 0.0
#grasp_explore = args.grasp_explore

## Quick hack for nonlocal memory between threads in Python 2
#nonlocal_variables = {'executing_action' : False,
#                      'primitive_action' : None,
#                      'best_pix_ind' : None,
#                      'push_success' : False,
#                      'grasp_success' : False,
#                      'grasp_reward' : 0,
#                      'improved_grasp_reward' : 0,
#                      'push_step' : 0, # plus one after pushing
#                      'goal_obj_idx' : 0,
#                      'goal_catched' : 0,
#                      'border_occupy_ratio' : 1,
#                      'decreased_occupy_ratio' : 0,
#                      'restart_scene' : 0,
#                      'episode' : 0, # episode number
#                      'new_episode_flag' : 0, # flag to begin a new episode
#                      'episode_grasp_reward' : 0, # grasp reward at the end of a episode
#                      'episode_ratio_of_grasp_to_push' : 0, # ratio of grasp to push at the end of a episode
#                      'episode_improved_grasp_reward' : 0,
#                      'push_predictions': np.zeros((16, 224, 224), dtype=float),
#                      'grasp_predictions' : np.zeros((16,224,224),dtype=float)} # average of improved grasp reward of a episode

## --------- Initialize nonlocal variables -----------
#nonlocal_variables['goal_obj_idx'] = args.goal_obj_idx

#if continue_logging:
#    if not is_testing:
#        nonlocal_variables['episode'] = trainer.episode_log[len(trainer.episode_log) - 1][0]
#    if stage == 'push_only':
#        # Initialize nonlocal memory
#        nonlocal_variables['push_step'] = trainer.push_step_log[trainer.iteration - 1][0]
#        nonlocal_variables['episode_improved_grasp_reward'] = trainer.episode_improved_grasp_reward_log[len(trainer.episode_improved_grasp_reward_log) - 1][0]

## ------ Tensorboard setting --------
#timestamp = time.time()
#timestamp_value = datetime.datetime.fromtimestamp(timestamp)
#tensor_logging_directory = args.tensor_logging_directory
#if continue_logging:
#    writer = SummaryWriter(os.path.join(tensor_logging_directory, logging_directory.split('/')[-1]))
#else:
#    # writer = SummaryWriter(os.path.join(tensor_logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S')))
#    writer = SummaryWriter(os.path.join(tensor_logging_directory, timestamp_value.strftime('%Y-%m-%d.%H_%M_%S')))
#print('Temp init done')