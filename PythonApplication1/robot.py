from utils.arg.model import ArgsModel
from utils.robot.model import RobotModel
from utils.robot.sim import RobotSim
from utils.robot.objects import RobotObjects

from utils.robot.mask import RobotMask
from utils.robot.push import RobotPush
from utils.robot.move import RobotMove
from utils.robot.gripper import RobotGripper
from utils.robot.camera import RobotCamera
from utils.robot.grasp import RobotGrasp


class Robot():
    debug: bool = False
    a: ArgsModel = None
    m: RobotModel = None
    sim: RobotSim = None
    obj: RobotObjects = None

    def __init__(self):
        pass

    def create_empty_helpers(self, a: ArgsModel):
        # save link
        self.a: ArgsModel = a

        # empty init:
        self.m: RobotModel = RobotModel() # just fill some variables, using message from cmd
        self.sim: RobotSim  = RobotSim() # connect to sim, restart, set cam pos/rot, pick 2 screenshots from cam: rgb and depth
        self.obj: RobotObjects = RobotObjects() # instantiate some random cubes in scene

        # empty init:
        self.mask = RobotMask()
        self.pusher = RobotPush()
        self.gripper = RobotGripper()
        self.mover = RobotMove()
        self.cam = RobotCamera()
        self.grasper = RobotGrasp()
                
        if self.debug:
            self.m.debug_types_values()

    def add_objects(self):
        self.obj.add_objects(self.m) #(self.sim.engine, self.m)

    # test mask    
    def get_test_obj_mask(self, obj_ind):
        return self.mask.get_test_obj_mask(obj_ind, self.sim, self.m);
    def get_test_obj_masks(self, obj_ind):
        return self.mask.get_test_obj_masks(self.sim, self.m);
        
    # just mask    
    def get_obj_mask(self, obj_ind):
        return self.mask.get_obj_mask(obj_ind, self.sim, self.m);
    def get_obj_masks(self, obj_ind):
        return self.mask.get_obj_masks(self.sim, self.m);

    # grasp/push
    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        return self.grasper.grasp(self.a, self.sim, self.m, self.cam, self.gripper, self.mover, self.obj, position, heightmap_rotation_angle, workspace_limits)
    def push(self, position, heightmap_rotation_angle, workspace_limits):
        return self.pusher.push(self.sim, self.m, self.gripper, self.mover, position, heightmap_rotation_angle, workspace_limits)

    # render
    def get_camera_data(self):
        return self.cam.get_camera_data(self.sim, self.m)

    # sim extra
    #def check_sim(self, a: ArgsModel, obj: RobotObjects, sim: RobotSim, m: RobotModel):
    def check_sim(self):
        #return self.sim.check_sim(a, obj, sim, m)
        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = self.m.engine.global_position_get(self.m.RG2_tip_handle)
        
        check_0a =  gripper_position[0] > self.a.workspace_limits[0][0] - 0.1
        check_0b = gripper_position[0] < self.a.workspace_limits[0][1] + 0.1
        check_1a =  gripper_position[1] > self.a.workspace_limits[1][0] - 0.1
        check_1b = gripper_position[1] < self.a.workspace_limits[1][1] + 0.1
        check_2a = gripper_position[2] > self.a.workspace_limits[2][0]
        check_2b =  gripper_position[2] < self.a.workspace_limits[2][1]

        sim_ok = check_0a and check_0b and check_1a and check_1b and check_2a and check_2b
        
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            #print(check_0a , check_0b , check_1a , check_1b , check_2a , check_2b)
            #print('pos0,pos1,pos2:',gripper_position[0], gripper_position[1], gripper_position[2])
            self.sim.restart_sim(self.m)
            self.add_objects()
        else:
            print('Simulation is stable')


