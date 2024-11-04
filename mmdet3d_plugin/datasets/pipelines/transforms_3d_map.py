# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import warnings
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines.data_augment_utils import noise_per_object_v3_
from mmdet3d.datasets.pipelines import (RandomFlip3D, GlobalRotScaleTrans)

@PIPELINES.register_module()
class RandomFlip3DMap(RandomFlip3D):
    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 update_img2lidar=False,
                 **kwargs):
        super().__init__(sync_2d=sync_2d,flip_ratio_bev_horizontal=flip_ratio_bev_horizontal,flip_ratio_bev_vertical=flip_ratio_bev_vertical,update_img2lidar=update_img2lidar,**kwargs)
    
    def update_transform(self, input_dict):
        transform = torch.zeros((input_dict['img_inputs'][1].shape[0],4,4)).float()
        transform[:,:3,:3] = input_dict['img_inputs'][1]
        transform[:,:3,-1] = input_dict['img_inputs'][2]
        transform[:, -1, -1] = 1.0

        aug_transform = torch.eye(4).float()
        if input_dict['pcd_horizontal_flip']:
            aug_transform[1,1] = -1
        if input_dict['pcd_vertical_flip']:
            aug_transform[0,0] = -1
        aug_transform = aug_transform.view(1,4,4)
        new_transform = aug_transform.matmul(transform)
        input_dict['aug_transform'] = aug_transform[0].matmul(input_dict['aug_transform'])
        input_dict['aug_transform_flip'] = aug_transform[0]
        input_dict['img_inputs'][1][...] = new_transform[:,:3,:3]
        input_dict['img_inputs'][2][...] = new_transform[:,:3,-1]

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # filp 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                # flip_horizontal = False  # fix the horizontal flip parameter
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                # flip_vertical = False  # fix the vertical flip parameter
                input_dict['pcd_vertical_flip'] = flip_vertical
        
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])

        if 'img_inputs' in input_dict:
            assert self.update_img2lidar
            self.update_transform(input_dict)
        return input_dict

@PIPELINES.register_module()
class GlobalRotScaleTransMap(GlobalRotScaleTrans):
    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False,
                 update_img2lidar=False):
        super().__init__(rot_range=rot_range,scale_ratio_range=scale_ratio_range,translation_std=translation_std,shift_height=shift_height,update_img2lidar=update_img2lidar)
    
    def update_transform(self, input_dict):
        transform = torch.zeros((input_dict['img_inputs'][1].shape[0],4,4)).float()
        transform[:,:3,:3] = input_dict['img_inputs'][1]
        transform[:,:3,-1] = input_dict['img_inputs'][2]
        transform[:, -1, -1] = 1.0

        aug_transform = torch.zeros((input_dict['img_inputs'][1].shape[0],4,4)).float()
        if 'pcd_rotation' in input_dict:
            aug_transform[:,:3,:3] = input_dict['pcd_rotation'].T * input_dict['pcd_scale_factor']
        else:
            aug_transform[:, :3, :3] = torch.eye(3).view(1,3,3) * input_dict['pcd_scale_factor']
        aug_transform[:,:3,-1] = torch.from_numpy(input_dict['pcd_trans']).reshape(1,3)
        aug_transform[:, -1, -1] = 1.0
        input_dict['aug_transform'] = aug_transform[0]

        new_transform = aug_transform.matmul(transform)
        input_dict['img_inputs'][1][...] = new_transform[:,:3,:3]
        input_dict['img_inputs'][2][...] = new_transform[:,:3,-1]