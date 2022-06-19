
import time
import datetime
import os

from utils.arg.model import ArgsModel
from utils.logger.model import LoggerModel

class LoggerInit():
    def __init__(self):
        pass

    def copy_args(self, m: LoggerModel, a: ArgsModel):
        """
        this method just creates folders - logs/2022-04-28.16_44_09/manySubfolders:
            - data
            - info
            - models
            - recordings
            - transitions
            - visualizations
        """

        # Create directory "logs/2020-19-31/"
        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        m.continue_logging = a.continue_logging
        if m.continue_logging:
            m.base_directory = a.logging_directory
            if m.debug_mode:
                print('Pre-loading data logging session: %s' % (m.base_directory))
        else:
            m.base_directory = os.path.join(a.logging_directory, timestamp_value.strftime('%Y-%m-%d.%H_%M_%S'))
            # m.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
            if m.debug_mode:
                print('Creating data logging session: %s' % (m.base_directory))
        
        # generate paths foreach subfolder
        m.info_directory = os.path.join(m.base_directory, 'info')
        m.color_images_directory = os.path.join(m.base_directory, 'data', 'color-images')
        m.depth_images_directory = os.path.join(m.base_directory, 'data', 'depth-images')
        m.color_heightmaps_directory = os.path.join(m.base_directory, 'data', 'color-heightmaps')
        m.depth_heightmaps_directory = os.path.join(m.base_directory, 'data', 'depth-heightmaps')
        m.grasped_object_directory = os.path.join(m.base_directory, 'data', 'grasped-object-images')
        m.grasped_object_heightmaps_directory = os.path.join(m.base_directory, 'data', 'grasped-object-heightmaps')
        m.models_directory = os.path.join(m.base_directory, 'models')
        m.visualizations_directory = os.path.join(m.base_directory, 'visualizations')
        m.recordings_directory = os.path.join(m.base_directory, 'recordings')
        m.transitions_directory = os.path.join(m.base_directory, 'transitions')

        # create subfolders from each path   
        if not os.path.exists(m.info_directory):
            os.makedirs(m.info_directory)
        if not os.path.exists(m.color_images_directory):
            os.makedirs(m.color_images_directory)
        if not os.path.exists(m.depth_images_directory):
            os.makedirs(m.depth_images_directory)
        if not os.path.exists(m.color_heightmaps_directory):
            os.makedirs(m.color_heightmaps_directory)
        if not os.path.exists(m.depth_heightmaps_directory):
            os.makedirs(m.depth_heightmaps_directory)
        if not os.path.exists(m.grasped_object_directory):
            os.makedirs(m.grasped_object_directory)
        if not os.path.exists(m.grasped_object_heightmaps_directory):
            os.makedirs(m.grasped_object_heightmaps_directory)
        if not os.path.exists(m.models_directory):
            os.makedirs(m.models_directory)
        if not os.path.exists(m.visualizations_directory):
            os.makedirs(m.visualizations_directory)
        if not os.path.exists(m.recordings_directory):
            os.makedirs(m.recordings_directory)
        if not os.path.exists(m.transitions_directory):
            os.makedirs(os.path.join(m.transitions_directory, 'data'))

