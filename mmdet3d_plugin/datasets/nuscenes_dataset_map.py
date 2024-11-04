# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.pipelines import Compose
import torch

from pyquaternion import Quaternion
from zmq import device

@DATASETS.register_module()
class NuScenesDatasetMap(NuScenesDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 map_classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 speed_mode='relative_dis',
                 max_interval=3,
                 min_interval=0,
                 prev_only=False,
                 next_only=False,
                 test_adj = 'prev',
                 fix_direction=False,
                 test_adj_ids=None):
        
        super().__init__(
            ann_file,
            pipeline=pipeline,
                 data_root=data_root,
                 classes=classes,
                 load_interval=load_interval,
                 with_velocity=with_velocity,
                 modality=modality,
                 box_type_3d=box_type_3d,
                 filter_empty_gt=filter_empty_gt,
                 test_mode=test_mode,
                 eval_version=eval_version,
                 use_valid_flag=use_valid_flag,
                 img_info_prototype=img_info_prototype,
                 speed_mode=speed_mode,
                 max_interval=max_interval,
                 min_interval=min_interval,
                 prev_only=prev_only,
                 next_only=next_only,
                 test_adj =test_adj,
                 fix_direction=fix_direction,
                 test_adj_ids=test_adj_ids)

        self.map_classes = map_classes

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            location=info['location'],
        )

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        input_dict["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))
            elif self.img_info_prototype == 'bevdet':
                image_paths = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                input_dict.update(dict(
                    img_info=info['cams'],
                    img_filename=image_paths
                    ))
            elif self.img_info_prototype == 'bevdet_sequential':
                if info ['prev'] is None or info['next'] is None:
                    adjacent= 'prev' if info['next'] is None else 'next'
                else:
                    if self.prev_only or self.next_only:
                        adjacent = 'prev' if self.prev_only else 'next'
                    elif self.test_mode:
                        adjacent = self.test_adj
                    else:
                        adjacent = np.random.choice(['prev', 'next'])
                if type(info[adjacent]) is list:
                    if self.test_mode:
                        if self.test_adj_ids is not None:
                            info_adj=[]
                            select_id = self.test_adj_ids
                            for id_tmp in select_id:
                                id_tmp = min(id_tmp, len(info[adjacent])-1)
                                info_adj.append(info[adjacent][id_tmp])
                        else:
                            select_id = min((self.max_interval+self.min_interval)//2,
                                            len(info[adjacent])-1)
                            info_adj = info[adjacent][select_id]
                    else:
                        if len(info[adjacent])<= self.min_interval:
                            select_id = len(info[adjacent])-1
                        else:
                            select_id = np.random.choice([adj_id for adj_id in range(
                                min(self.min_interval,len(info[adjacent])),
                                min(self.max_interval,len(info[adjacent])))])
                        info_adj = info[adjacent][select_id]
                else:
                    info_adj = info[adjacent]
                input_dict.update(dict(img_info=info['cams'],
                                       curr=info,
                                       adjacent=info_adj,
                                       adjacent_type=adjacent))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.img_info_prototype == 'bevdet_sequential':
                bbox = input_dict['ann_info']['gt_bboxes_3d'].tensor
                if 'abs' in self.speed_mode:
                    bbox[:, 7:9] = bbox[:, 7:9] + torch.from_numpy(info['velo']).view(1,2)
                if input_dict['adjacent_type'] == 'next' and not self.fix_direction:
                    bbox[:, 7:9] = -bbox[:, 7:9]
                if 'dis' in self.speed_mode:
                    time = abs(input_dict['timestamp'] - 1e-6 * input_dict['adjacent']['timestamp'])
                    bbox[:, 7:9] = bbox[:, 7:9] * time
                input_dict['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(bbox,
                                                                              box_dim=bbox.shape[-1],
                                                                              origin=(0.5, 0.5, 0.0))
        return input_dict


    # def get_data_info(self, index):
    #     """Get data info according to the given index.

    #     Args:
    #         index (int): Index of the sample data to get.

    #     Returns:
    #         dict: Data information that will be passed to the data \
    #             preprocessing pipelines. It includes the following keys:

    #             - sample_idx (str): Sample index.
    #             - pts_filename (str): Filename of point clouds.
    #             - sweeps (list[dict]): Infos of sweeps.
    #             - timestamp (float): Sample timestamp.
    #             - img_filename (str, optional): Image filename.
    #             - lidar2img (list[np.ndarray], optional): Transformations \
    #                 from lidar to different cameras.
    #             - ann_info (dict): Annotation info.
    #     """
    #     info = self.data_infos[index]
    #     # standard protocal modified from SECOND.Pytorch
    #     input_dict = dict(
    #         sample_idx=info['token'],
    #         pts_filename=info['lidar_path'],
    #         sweeps=info['sweeps'],
    #         timestamp=info['timestamp'] / 1e6,
    #         location=info['location'],
    #     )

    #     # ego to global transform
    #     ego2global = np.eye(4).astype(np.float32)
    #     ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    #     ego2global[:3, 3] = info["ego2global_translation"]
    #     input_dict["ego2global"] = ego2global

    #     # lidar to ego transform
    #     lidar2ego = np.eye(4).astype(np.float32)
    #     lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    #     lidar2ego[:3, 3] = info["lidar2ego_translation"]
    #     input_dict["lidar2ego"] = lidar2ego

    #     if self.modality['use_camera']:
    #         image_paths = []
    #         lidar2img_rts = []
    #         for cam_type, cam_info in info['cams'].items():
    #             image_paths.append(cam_info['data_path'])
    #             # obtain lidar to image transformation matrix
    #             lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    #             lidar2cam_t = cam_info[
    #                 'sensor2lidar_translation'] @ lidar2cam_r.T
    #             lidar2cam_rt = np.eye(4)
    #             lidar2cam_rt[:3, :3] = lidar2cam_r.T
    #             lidar2cam_rt[3, :3] = -lidar2cam_t
    #             intrinsic = cam_info['cam_intrinsic']
    #             viewpad = np.eye(4)
    #             viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    #             lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    #             lidar2img_rts.append(lidar2img_rt)

    #         input_dict.update(
    #             dict(
    #                 img_filename=image_paths,
    #                 lidar2img=lidar2img_rts,
    #             ))

    #     if not self.test_mode:
    #         annos = self.get_ann_info(index)
    #         input_dict['ann_info'] = annos

    #     return input_dict

    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]).to(device='cpu')
        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["bev_seg"]
            label = result["gt_masks_bev"]
            pred = pred[0]

            pred = pred.detach().reshape(num_classes, -1).to(device='cpu')
            label = label.detach().bool().reshape(num_classes, -1).to(device='cpu')

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        if 'bboxmap' in metric:
            metric = ['bbox','map']
        
        bbox_list = [dict() for _ in range(len(results))]
        for result_bbox, result in zip(bbox_list, results):
            result_bbox['pts_bbox'] = result['pts_bbox']
        
        if 'map' in metric:
            map_list = [dict() for _ in range(len(results))]
            for result_map, result in zip(map_list, results):
                result_map['bev_seg'] = result['bev_seg']
                result_map['gt_masks_bev'] = result['gt_masks_bev']

        result_files, tmp_dir = self.format_results(bbox_list, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if 'map' in metric:
            results_dict.update(self.evaluate_map(map_list))
        
        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return results_dict
