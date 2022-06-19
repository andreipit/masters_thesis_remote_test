import os

from typing import NoReturn
from logger import Logger
from proc import Proc
from robot import Robot

from train_loop import TrainLoop
from trainer import Trainer
from utils.arg.model import ArgsModel

class Monitor():
    FILEPATH = os.path.join('utils','monitor')
    PROCNAME = 'proc_model.txt'
    ROBOTNAME = 'robot_model.txt'
    TRAINERNAME = 'trainer_model.txt'

    def __init__(self):
        pass

    def start():
        pass
    
    @staticmethod
    def show(tl: TrainLoop, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc) -> NoReturn:
        Monitor.show_proc_model(p)
        Monitor.show_robot_model(r)
        Monitor.show_trainer_model(t)

    @staticmethod
    def show_proc_model(p: Proc) -> NoReturn:
        x = p.m.nonlocal_variables
        filename: str = os.path.join(Monitor.FILEPATH, Monitor.PROCNAME)
        f = open(filename, "a")
        f.seek(0); f.truncate() # clear 
        for i in x:
            f.write(str(i) + '=' + str(x[i]) + '\n')
        f.close()

    @staticmethod
    def show_robot_model(r: Robot) -> NoReturn:
        filename: str = os.path.join(Monitor.FILEPATH, Monitor.ROBOTNAME)
        f = open(filename, "a")
        f.seek(0); f.truncate() # clear

        f.write('workspace_limits' + '=' + str(r.m.workspace_limits) + '\n')
        f.write('color_space' + '=' + str(r.m.color_space) + '\n')
        f.write('num_obj' + '=' + str(r.m.num_obj) + '\n')
        f.write('stage' + '=' + str(r.m.stage) + '\n')
        f.write('goal_conditioned' + '=' + str(r.m.goal_conditioned) + '\n')
        f.write('grasp_goal_conditioned' + '=' + str(r.m.grasp_goal_conditioned) + '\n')
        f.write('obj_mesh_dir' + '=' + str(r.m.obj_mesh_dir) + '\n')
        f.write('mesh_list' + '=' + str(r.m.mesh_list) + '\n')
        f.write('obj_mesh_ind' + '=' + str(r.m.obj_mesh_ind) + '\n')
        f.write('obj_mesh_color' + '=' + str(r.m.obj_mesh_color) + '\n')
        f.write('is_testing' + '=' + str(r.m.is_testing) + '\n')
        f.write('test_preset_cases' + '=' + str(r.m.test_preset_cases) + '\n')
        f.write('test_preset_file' + '=' + str(r.m.test_preset_file) + '\n')
        
        f.write('---------created later at robot_objects.py------' + '=' + str(0) + '\n')
        f.write('test_obj_mesh_files' + '=' + str(r.m.test_obj_mesh_files) + '\n')
        f.write('test_obj_mesh_colors' + '=' + str(r.m.test_obj_mesh_colors) + '\n')
        f.write('test_obj_positions' + '=' + str(r.m.test_obj_positions) + '\n')
        f.write('test_obj_orientations' + '=' + str(r.m.test_obj_orientations) + '\n')
        f.write('object_handles' + '=' + str(r.m.object_handles) + '\n')
        
        f.write('---------# created later at robot_sim.py------' + '=' + str(0) + '\n')
        f.write('engine' + '=' + str(r.m.engine) + '\n')
        f.write('UR5_target_handle' + '=' + str(r.m.UR5_target_handle) + '\n')
        f.write('RG2_tip_handle' + '=' + str(r.m.RG2_tip_handle) + '\n')
        f.write('cam_handle' + '=' + str(r.m.cam_handle) + '\n')
        f.write('cam_pose' + '=' + str(r.m.cam_pose) + '\n')
        f.write('cam_intrinsics' + '=' + str(r.m.cam_intrinsics) + '\n')
        f.write('bg_color_img' + '=' + str(r.m.bg_color_img) + '\n')
        f.write('bg_depth_img' + '=' + str(r.m.bg_depth_img) + '\n')
        f.write('cam_depth_scale' + '=' + str(r.m.cam_depth_scale) + '\n')
        f.close()


    @staticmethod
    def show_trainer_model(t: Trainer) -> NoReturn:
        filename: str = os.path.join(Monitor.FILEPATH, Monitor.TRAINERNAME)
        f = open(filename, "a")
        f.seek(0); f.truncate() # clear
        f.write('model' + '=' + str(t.m.model) + '\n')
        f.close()


        #debug_mode:bool = False

        #model: nn.Module = None

        #stage = None
        #grasp_goal_conditioned = None
        #is_testing = None
        #alternating_training = None
        #use_cuda = None
        #explore_model = None
        #future_reward_discount = None
        #criterion = None
        #optimizer = None
        #iteration = None

        ## buffers lists:
        #executed_action_log = []
        #label_value_log = []
        #reward_value_log = []
        #predicted_value_log = []
        #use_heuristic_log = []
        #is_exploit_log = []
        #clearance_log = []
        #push_step_log = []
        #grasp_obj_log = [] # grasp object index (if push or grasp fail then index is -1)
        #episode_log = []
        #episode_improved_grasp_reward_log = []
