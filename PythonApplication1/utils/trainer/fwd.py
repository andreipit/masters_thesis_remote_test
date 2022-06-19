#import os
#import time
import numpy as np
#import cv2
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable
#from utils import CrossEntropyLoss2d
#from models import push_grasp_net, goal_conditioned_net
from scipy import ndimage

from utils.arg.model import ArgsModel
from utils.trainer.model import TrainerModel


class TrainerFwd(object):

    # Compute forward pass through model to compute affordances/Q
    def forward(self, a: ArgsModel, m: TrainerModel, color_heightmap, depth_heightmap, is_volatile=False, grasp_explore_actions=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        # The third dimension is rgb
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        # change me
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for c in range(3):
            input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)

        # Pass input data through model
        if not grasp_explore_actions:
            output_prob, state_feat = m.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
        else:
            output_prob, state_feat = m.explore_model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
            print('Explore from graps net!')

        if a.stage == 'grasp_only':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        elif a.stage == 'push_only':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        elif a.stage == 'push_grasp':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return push_predictions, grasp_predictions, state_feat
