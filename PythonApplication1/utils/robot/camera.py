import time
import numpy as np

import utils.utils as utils
from utils.arg.model import ArgsModel
from utils.robot.model import RobotModel
from utils.robot.sim import RobotSim
from simulator import Simulator
from utils.custom_types import NDArray

class RobotCamera():

    def __init__(self):
        pass

    def get_camera_data(self, sim: RobotSim, m: RobotModel):

        # Get color image from simulation
        #sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
        sim_ret, resolution, raw_image = m.engine.camera_image_rgb_get(m.cam_handle)
        
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        #sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
        sim_ret, resolution, depth_buffer = m.engine.camera_image_depth_get(m.cam_handle)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear

        return color_img, depth_img




