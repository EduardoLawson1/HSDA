# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import warnings
from copy import deepcopy

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose

from mmdet3d.datasets.pipelines import MultiScaleFlipAug3D

@PIPELINES.register_module()
class MultiScaleFlipAug3DMap(MultiScaleFlipAug3D):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool): Whether apply horizontal flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
        pcd_vertical_flip (bool): Whether apply vertical flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
    """
    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
        super().__init__(transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=flip,
                 flip_direction=flip_direction,
                 pcd_horizontal_flip=pcd_horizontal_flip,
                 pcd_vertical_flip=pcd_vertical_flip)
    
    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with \
                different scales and flips.
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = \
                                    pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = \
                                    pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = \
                                    pcd_vertical_flip
                                import torch
                                aug_transform = torch.zeros((4,4)).float()
                                aug_transform[:3, :3] = torch.eye(3)* _results['pcd_scale_factor']
                                aug_transform[-1, -1] = 1.0
                                _results['aug_transform'] = aug_transform
                                aug_transform = torch.eye(4).float()
                                if _results['pcd_horizontal_flip']:
                                    aug_transform[1,1] = -1
                                if _results['pcd_vertical_flip']:
                                    aug_transform[0,0] = -1
                                _results['aug_transform_flip'] = aug_transform
                                _results['aug_transform'] = aug_transform.matmul(_results['aug_transform'])

                                data = self.transforms(_results)
                                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

