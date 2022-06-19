#import argparse


#class ArgsParser():

#    def __init__(self):
#        pass

#    @staticmethod
#    def generate_parser() -> argparse.ArgumentParser:
#        parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')
        
#        # --------------- Setup options ---------------
#        parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
#        parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
#        parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
#        parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
#        parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

#        # ------------- Algorithm options -------------
#        parser.add_argument('--stage', dest='stage', action='store', default='grasp_only',                               help='stage of training: 1.grasp_only, 2.push_only, 3.push_grasp')
#        parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
#        parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
#        parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
#        parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
#        parser.add_argument('--grasp_reward_threshold', dest='grasp_reward_threshold', type=float, action='store', default=1.8)
#        parser.add_argument('--max_push_episode_length', dest='max_push_episode_length', type=int, action='store', default=5)
#        parser.add_argument('--grasp_explore', dest='grasp_explore', action='store_true', default=False)

#        # -------------- Testing options --------------
#        parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
#        parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
#        parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
#        parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')
#        parser.add_argument('--random_scene_testing', dest='random_scene_testing', action='store_true', default=False)
    
#        # -------------- Goal-conditioned options --------------
#        parser.add_argument('--goal_obj_idx', dest='goal_obj_idx', type=int, action='store', default=0)
#        parser.add_argument('--goal_conditioned', dest='goal_conditioned', action='store_true', default=False)
#        parser.add_argument('--grasp_goal_conditioned', dest='grasp_goal_conditioned', action='store_true', default=False)

#        # ------ Pre-loading and logging options ------
#        parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
#        parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
#        parser.add_argument('--load_explore_snapshot', dest='load_explore_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
#        parser.add_argument('--explore_snapshot_file', dest='explore_snapshot_file', action='store')
#        parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
#        parser.add_argument('--logging_directory', dest='logging_directory', action='store')
#        parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')
#        parser.add_argument('--tensor_logging_directory', dest='tensor_logging_directory', action='store', default='./tensorlog')
#        parser.add_argument('--alternating_training', dest='alternating_training', action='store_true', default=False)
#        parser.add_argument('--cooperative_training', dest='cooperative_training', action='store_true', default=False)

#        return parser




