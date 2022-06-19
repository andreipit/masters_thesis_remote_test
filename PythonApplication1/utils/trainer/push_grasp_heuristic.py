import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import ndimage
import matplotlib.pyplot as plt

from utils.utils import CrossEntropyLoss2d
from utils.arg.model import ArgsModel
from utils.trainer.model import TrainerModel

class TrainerPushGraspHeuristic(object):

    def push_heuristic(self, m: TrainerModel, a: ArgsModel, depth_heightmap):
        """
        Given: image with pixels == height
        Task: find best pixel on image
        """
        num_rotations = 16
        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) - rotated_heightmap > 0.02] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_push_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_push_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                push_predictions = tmp_push_predictions
            else:
                push_predictions = np.concatenate((push_predictions, tmp_push_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        return best_pix_ind


    def grasp_heuristic(self, m: TrainerModel, a: ArgsModel, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind


