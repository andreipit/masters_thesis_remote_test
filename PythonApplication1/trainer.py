# 2.3) models wrapper -> 2 conv-nets(push/grasp) + torchvision.densenet (pretrained on imagenet, creates intermediate features)
# Freeze one. Add reward, loss, optimizer, logs, dropout/batchnorm mode

from torch._C import default_generator

from utils.arg.model import ArgsModel
from utils.trainer.model import TrainerModel

from utils.trainer.init import TrainerInit
from utils.trainer.preload import TrainerPreload
from utils.trainer.push_grasp_heuristic import TrainerPushGraspHeuristic
from utils.trainer.get_vis import TrainerGetVis
from utils.trainer.fwd import TrainerFwd
from utils.trainer.goal_fwd import TrainerGoalFwd
from utils.trainer.label_value import TrainerLabelValue
from utils.trainer.backprop import TrainerBackprop

class Trainer():

    def __init__(self):
        pass

    """
    1) convert CMD args to model table and create:
    - use_cuda
    - model, explore_model = our libraries
    - criterion = torch.nn.SmoothL1Loss
    - optimizer = Adam
    2) if a.load_snapshot: load saved weights 
    3) froze NN: stage=grasp->push, push->grasp, alternating->p or g
    4) if CUDA -> set it in NN
    5) if train -> set it in NN (just change mode of Batchnorm, Dropout and don't actually run train)
    craete train logs (lists)
    """
    def create_empty_helpers(self, a: ArgsModel):

        # just save args
        self.a = a

        # empty init:
        self.m = TrainerModel()
        self.init = TrainerInit() 
        self.preloader = TrainerPreload()
        self.heuristic = TrainerPushGraspHeuristic()
        self.vis = TrainerGetVis()
        self.fwd = TrainerFwd()
        self.gfwd = TrainerGoalFwd()
        self.label = TrainerLabelValue()
        self.back = TrainerBackprop()

#region Misc        
    def copy_args_from_log(self, transitions_directory):
        self.preloader.preload(transitions_directory, self.m) # set 'resume training' if was paused (just fill buffers-logs) # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    def get_label_value(self, primitive_action,  grasp_success, grasp_reward, improved_grasp_reward, change_detected, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap=None, goal_catched=0, decreased_occupy_ratio=0):
        return self.label.get_label_value(self.fwd, self.gfwd, self.a, self.m, primitive_action,  grasp_success, grasp_reward, improved_grasp_reward, change_detected, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, goal_catched, decreased_occupy_ratio)
#endregion

#region Move        
    def push_heuristic(self, depth_heightmap):
        return self.heuristic.push_heuristic(self.m, default_generator)
    def grasp_heuristic(self, depth_heightmap):
        return self.heuristic.grasp_heuristic(self.m, default_generator)
#endregion
        
#region Optimization
    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, grasp_explore_actions=False, specific_rotation=-1):
        return self.fwd.forward(self.a, self.m, color_heightmap, depth_heightmap, is_volatile, grasp_explore_actions, specific_rotation)
    def goal_forward(self, color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=-1):
        return self.gfwd.goal_forward(self.a, self.m, color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile, specific_rotation)
    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap=None):
        return self.back.backprop(self.fwd, self.gfwd, self.a, self.m, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap)
#endregion    

#region Visualizations
    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):
        return self.vis.get_prediction_vis(self.m, predictions, color_heightmap, best_pix_ind)
    def get_push_direction_vis(self, predictions, color_heightmap):
        return self.vis.get_push_direction_vis(self.m, predictions, color_heightmap)
    def get_best_push_direction_vis(self, best_pix_ind, color_heightmap):
        return self.vis.get_best_push_direction_vis(self.m, predictions, color_heightmap, best_pix_ind)
#endregion
