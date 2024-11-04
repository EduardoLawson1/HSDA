# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torchvision
from PIL import Image
import mmcv
import numpy as np
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from nuscenes.map_expansion.map_api import NuScenesMap

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet.core import multi_apply
from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles_BEVDet

@PIPELINES.register_module()
class LoadBEVMap(object):
    def __init__(self, data_config, grid_config, dataset_root, is_train=False,
                 sequential=False, aligned=False, trans_only=True):
        self.grid_config = grid_config
        self.xbound = self.grid_config['xbound']
        self.ybound = self.grid_config['ybound']
        self.classes = self.grid_config['map_classes']
        patch_h = self.ybound[1] - self.ybound[0]
        patch_w = self.xbound[1] - self.xbound[0]
        canvas_h = int(patch_h / self.ybound[2])
        canvas_w = int(patch_w / self.xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)
    
    def get_labels(self, results):
        lidar2point = results["aug_transform"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = results["lidar2ego"]
        ego2global = results["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180
        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))
        location = results["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        
        masks = masks.astype(np.bool)
        
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        return labels
    
    def __call__(self, results):
        results["gt_masks_bev"] = self.get_labels(results)
        return results
    

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_BEV_multi(LoadMultiViewImageFromFiles_BEVDet):
    def __init__(self, data_config, grid_config, dataset_root, is_train=False,
                 sequential=False, aligned=False, trans_only=True):
        super().__init__(data_config, is_train=is_train,
                 sequential=sequential, aligned=aligned, trans_only=trans_only)
        self.grid_config = grid_config
        self.xbound = self.grid_config['xbound']
        self.ybound = self.grid_config['ybound']
        self.classes = self.grid_config['map_classes']
        patch_h = self.ybound[1] - self.ybound[0]
        patch_w = self.xbound[1] - self.xbound[0]
        canvas_h = int(patch_h / self.ybound[2])
        canvas_w = int(patch_w / self.xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def get_labels(self, results):
        lidar2ego = results["lidar2ego"]
        ego2global = results["ego2global"]
        lidar2global = ego2global @ lidar2ego 
        lidar2global = ego2global @ lidar2ego 

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180
        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))
        location = results["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        masks = masks.astype(np.bool)
        
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        return labels
    
    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        results["gt_masks_bev"] = self.get_labels(results)
        return results


    




