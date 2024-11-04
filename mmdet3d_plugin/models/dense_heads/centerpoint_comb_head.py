# Copyright (c) OpenMMLab. All rights reserved.
import copy
from distutils.command.config import config
from turtle import forward
import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.models import builder
from mmdet3d.models.dense_heads import CenterHead
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply

def sigmoid_focal_loss(
    inputs,
    targets,
    loss_weight,
    alpha = -1,
    gamma = 2,
    reduction = "mean",
):
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    loss = loss_weight * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

@HEADS.register_module()
class CenterCombHead(CenterHead):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 grid_config=None,
                 grid_transform=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 loss_seg_weight=1,
                 classifier='short',
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                #  in_channels_separate=256,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                #  separate=False,
                 init_cfg=None,
                 task_specific=True,
                 loss_prefix=''):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(in_channels=in_channels,
                 tasks=tasks,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 bbox_coder=bbox_coder,
                 common_heads=common_heads,
                 loss_cls=loss_cls,
                 loss_bbox=loss_bbox,
                 separate_head=separate_head,
                 share_conv_channel=share_conv_channel,
                 num_heatmap_convs=num_heatmap_convs,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 bias=bias,
                 norm_bbox=norm_bbox,
                 init_cfg=init_cfg,
                 task_specific=task_specific,
                 loss_prefix=loss_prefix)

        self.grid_config=grid_config
        self.grid_transform=grid_transform
        self.loss_seg_weight=loss_seg_weight
        # self.separate=separate
        # self.shared_conv_separate = ConvModule(
        #     in_channels_separate,
        #     share_conv_channel,
        #     kernel_size=3,
        #     padding=1,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg,
        #     bias=bias)
    
        self.map_classes = self.grid_config['map_classes']
        if classifier == 'short':
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, len(self.map_classes), 1),
            )
        elif classifier == 'medium':
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, len(self.map_classes), 1),
            )
        elif classifier == 'large':
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, len(self.map_classes), 1),
            )
        elif classifier == 'large-shrink':
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(True),
                nn.Conv2d(in_channels//2, in_channels//4, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(True),
                nn.Conv2d(in_channels//4, len(self.map_classes), 1)
                )
    
    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        seg_x = self.classifier(x)
        x = self.shared_conv(x)

        # if self.separate:
        #     seg_x = self.classifier(x[:,:320])
        #     x = self.shared_conv_separate(x[:,320:])
        # else:
        #     seg_x = self.classifier(x)
        #     x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))
        #     print('task(x) heatmap',task(x)['heatmap'].shape)
        # assert 0

        return ret_dicts, seg_x
    
    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        ret_dicts, seg_x=multi_apply(self.forward_single, feats)
        ret_dicts=ret_dicts[0]
        temp=[]
        for ret_dict in ret_dicts:
            temp.append([ret_dict])
        ret_dicts=temp

        return ret_dicts, seg_x

    @force_fp32(apply_to=('preds_dicts', 'seg_outs'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, gt_masks_bev, preds_dicts, seg_outs, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        
        # print(len(heatmaps))
        # print(heatmaps[0].shape)
        # assert 0

        # print(gt_masks_bev[:, 0].shape)
        # assert 0

        # ########## visualize heatmaps only ##########
        # import matplotlib.pyplot as plt
        # heatmap = heatmaps[1]
        # heatmap = heatmap.to('cpu')
        # for i in range(len(heatmaps[1])):
        #     print(i)
        #     print(gt_bboxes_3d[i].tensor.shape)
        #     print(gt_labels_3d[i])
        #     # plt.imsave('visualization/heatmap/heatmaps_{}.jpg'.format(i),heatmaps[0][i][0]*255)
        #     # plt.imsave('heatmaps_{}.jpg'.format(i),heatmaps[0][i][0]*255)
        #     plt.imsave('heatmaps_{}.jpg'.format(i),heatmap[i][0]*200+gt_masks_bev[i, 0].to('cpu')*50)
        # assert 0
        # ##############################################

        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            # seg_outs[0] = clip_sigmoid(seg_outs[0])
            # seg_outs[0] 8 4 128 128
            

            # import matplotlib.pyplot as plt
            # preds_dict[0]['heatmap'] = preds_dict[0]['heatmap'].to('cpu')
            # for i in range(len(preds_dict[0]['heatmap'])):
            #     print('i',i)
            #     print(preds_dict[0]['heatmap'][i][0].shape)
            #     plt.imsave('pred_comb_heatmaps_{}.jpg'.format(i),preds_dict[0]['heatmap'][i][0].detach()*125+seg_outs[0][i][0].detach().to('cpu')*125)
            #     plt.imsave('pred_heatmaps_{}.jpg'.format(i),preds_dict[0]['heatmap'][i][0].detach()*255)
            #     plt.imsave('pred_bev_{}.jpg'.format(i),seg_outs[0][i][0].detach().to('cpu')*255)
            # assert 0

            

            gt_masks_bev_comb = torch.unsqueeze(gt_masks_bev[:,0], dim=1)
            heatmaps_comb=heatmaps[task_id]+gt_masks_bev_comb # 8*1*128*128
            # num_pos_comb = heatmaps_comb.eq(1).float().sum().item()
            # print(num_pos_comb)
            
            seg_outs_comb = torch.unsqueeze(seg_outs[0][:, 0], dim=1)
            preds_comb=preds_dict[0]['heatmap']+seg_outs_comb

            loss_heatmap_comb=sigmoid_focal_loss(preds_comb, heatmaps_comb, self.loss_seg_weight)
            
            



            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                    preds_dict[0]['dim'], preds_dict[0]['rot'],
                    preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            if self.task_specific:
                name_list=['xy','z','whl','yaw','vel']
                clip_index = [0,2,3,6,8,10]
                for reg_task_id in range(len(name_list)):
                    pred_tmp = pred[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    target_box_tmp = target_box[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    bbox_weights_tmp = bbox_weights[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    loss_bbox_tmp = self.loss_bbox(
                        pred_tmp, target_box_tmp, bbox_weights_tmp, avg_factor=(num + 1e-4))
                    loss_dict[f'%stask{task_id}.loss_%s'%(self.loss_prefix,name_list[reg_task_id])] = loss_bbox_tmp
            else:
                loss_bbox = self.loss_bbox(
                    pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
                loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            loss_dict[f'%stask{task_id}.loss_heatmap'%(self.loss_prefix)] = loss_heatmap
            loss_dict[f'%stask{task_id}.loss_heatmap_comb'%(self.loss_prefix)] = loss_heatmap_comb
            
        
        ##### detected bev map masks / gt bev map & gt bboxes
        # seg_out=seg_outs[0]
        # seg_out[seg_out<0]=0
        # import matplotlib.pyplot as plt
        # import cv2
        # import os.path as path
        # import numpy as np
        # plt.figure()
        # fig, axs = plt.subplots(2, 4)
        # for i in range(len(seg_out)):
        #     for k, name in enumerate(self.map_classes):
        #         axs[0,k].imshow((seg_out[i,k].to('cpu')*50).byte())
        #         axs[0,k].invert_yaxis()
        #     for k, name in enumerate(self.map_classes):
        #         axs[1,k].imshow((gt_masks_bev[i,k].to('cpu')*50+heatmaps[0][i][0].to('cpu')*200).byte())
        #         # axs[1,k].imshow((gt_masks_bev[i,k].to('cpu')*50+temp*200).byte())
        #         axs[1,k].invert_yaxis()
        #     # plt.show()
        #     # plt.savefig(path.join('visualization/det_gt',name))
        #     plt.savefig('visualization/det_gt_test/{}_rot'.format(i))
        # assert 0
        ##########################################

        for index, name in enumerate(self.map_classes):
            loss_dict[f'task.loss_seg_{name}'] = sigmoid_focal_loss(seg_outs[0][:, index], gt_masks_bev[:, index], self.loss_seg_weight)
        
        
        
        

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        # preds_dicts = preds_dicts[0]

        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            # assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            nms_type = self.test_cfg.get('nms_type')
            if isinstance(nms_type,list):
                nms_type = nms_type[task_id]
            if nms_type == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas, task_id))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

@HEADS.register_module()
class CenterCombHeadLoss(CenterCombHead):
    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 grid_config=None,
                 grid_transform=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 loss_seg_weight=1,
                 classifier='short',
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None,
                 task_specific=True,
                 loss_prefix=''):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(in_channels=in_channels,
                 tasks=tasks,
                 grid_config=grid_config,
                 grid_transform=grid_transform,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 bbox_coder=bbox_coder,
                 common_heads=common_heads,
                 loss_cls=loss_cls,
                 loss_bbox=loss_bbox,
                 loss_seg_weight=loss_seg_weight,
                 classifier=classifier,
                 separate_head=separate_head,
                 share_conv_channel=share_conv_channel,
                 num_heatmap_convs=num_heatmap_convs,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 bias=bias,
                 norm_bbox=norm_bbox,
                 init_cfg=init_cfg,
                 task_specific=task_specific,
                 loss_prefix=loss_prefix)
    
    @force_fp32(apply_to=('preds_dicts', 'seg_outs'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, gt_masks_bev, preds_dicts, seg_outs, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)

        # ########## visualize heatmaps only ##########
        # import matplotlib.pyplot as plt
        # heatmap = heatmaps[1]
        # heatmap = heatmap.to('cpu')
        # for i in range(len(heatmaps[1])):
        #     print(i)
        #     print(gt_bboxes_3d[i].tensor.shape)
        #     print(gt_labels_3d[i])
        #     # plt.imsave('visualization/heatmap/heatmaps_{}.jpg'.format(i),heatmaps[0][i][0]*255)
        #     # plt.imsave('heatmaps_{}.jpg'.format(i),heatmaps[0][i][0]*255)
        #     plt.imsave('heatmaps_{}.jpg'.format(i),heatmap[i][0]*200+gt_masks_bev[i, 0].to('cpu')*50)
        # assert 0
        # ##############################################

        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            # preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            # seg_outs[0] = clip_sigmoid(seg_outs[0])
            # seg_outs[0] 8 4 128 128
            
            # import matplotlib.pyplot as plt
            # preds_dict[0]['heatmap'] = preds_dict[0]['heatmap'].to('cpu')
            # for i in range(len(preds_dict[0]['heatmap'])):
            #     print('i',i)
            #     print(preds_dict[0]['heatmap'][i][0].shape)
            #     plt.imsave('pred_comb_heatmaps_{}.jpg'.format(i),preds_dict[0]['heatmap'][i][0].detach()*125+seg_outs[0][i][0].detach().to('cpu')*125)
            #     plt.imsave('pred_heatmaps_{}.jpg'.format(i),preds_dict[0]['heatmap'][i][0].detach()*255)
            #     plt.imsave('pred_bev_{}.jpg'.format(i),seg_outs[0][i][0].detach().to('cpu')*255)
            # assert 0

            gt_masks_bev_comb = torch.unsqueeze(gt_masks_bev[:,0], dim=1)
            heatmaps_comb=heatmaps[task_id]+gt_masks_bev_comb # 8*1*128*128
            # num_pos_comb = heatmaps_comb.eq(1).float().sum().item()
            # print(num_pos_comb)
            
            seg_outs_comb = torch.unsqueeze(seg_outs[0][:, 0], dim=1)
            preds_comb=preds_dict[0]['heatmap']+seg_outs_comb

            loss_heatmap_comb=sigmoid_focal_loss(preds_comb, heatmaps_comb, self.loss_seg_weight)

            # num_pos = heatmaps[task_id].eq(1).float().sum().item()
            # loss_heatmap = self.loss_cls(
            #     preds_dict[0]['heatmap'],
            #     heatmaps[task_id],
            #     avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            loss_heatmap = sigmoid_focal_loss(preds_dict[0]['heatmap'], heatmaps[task_id], self.loss_seg_weight)

            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                    preds_dict[0]['dim'], preds_dict[0]['rot'],
                    preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            if self.task_specific:
                name_list=['xy','z','whl','yaw','vel']
                clip_index = [0,2,3,6,8,10]
                for reg_task_id in range(len(name_list)):
                    pred_tmp = pred[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    target_box_tmp = target_box[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    bbox_weights_tmp = bbox_weights[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    loss_bbox_tmp = self.loss_bbox(
                        pred_tmp, target_box_tmp, bbox_weights_tmp, avg_factor=(num + 1e-4))
                    loss_dict[f'%stask{task_id}.loss_%s'%(self.loss_prefix,name_list[reg_task_id])] = loss_bbox_tmp
            else:
                loss_bbox = self.loss_bbox(
                    pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
                loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            loss_dict[f'%stask{task_id}.loss_heatmap'%(self.loss_prefix)] = loss_heatmap
            loss_dict[f'%stask{task_id}.loss_heatmap_comb'%(self.loss_prefix)] = loss_heatmap_comb
            
        
        ##### detected bev map masks / gt bev map & gt bboxes
        # seg_out=seg_outs[0]
        # seg_out[seg_out<0]=0
        # import matplotlib.pyplot as plt
        # import cv2
        # import os.path as path
        # import numpy as np
        # plt.figure()
        # fig, axs = plt.subplots(2, 4)
        # for i in range(len(seg_out)):
        #     for k, name in enumerate(self.map_classes):
        #         axs[0,k].imshow((seg_out[i,k].to('cpu')*50).byte())
        #         axs[0,k].invert_yaxis()
        #     for k, name in enumerate(self.map_classes):
        #         axs[1,k].imshow((gt_masks_bev[i,k].to('cpu')*50+heatmaps[0][i][0].to('cpu')*200).byte())
        #         # axs[1,k].imshow((gt_masks_bev[i,k].to('cpu')*50+temp*200).byte())
        #         axs[1,k].invert_yaxis()
        #     # plt.show()
        #     # plt.savefig(path.join('visualization/det_gt',name))
        #     plt.savefig('visualization/det_gt_test/{}_rot'.format(i))
        # assert 0
        ##########################################

        for index, name in enumerate(self.map_classes):
            loss_dict[f'task.loss_seg_{name}'] = sigmoid_focal_loss(seg_outs[0][:, index], gt_masks_bev[:, index], self.loss_seg_weight)
        
        
        
        

        return loss_dict

@HEADS.register_module()
class CenterCatHead(CenterCombHead):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 grid_config=None,
                 grid_transform=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 loss_seg_weight=1,
                 classifier='short',
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None,
                 task_specific=True,
                 loss_prefix=''):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(in_channels=in_channels,
                 tasks=tasks,
                 grid_config=grid_config,
                 grid_transform=grid_transform,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 bbox_coder=bbox_coder,
                 common_heads=common_heads,
                 loss_cls=loss_cls,
                 loss_bbox=loss_bbox,
                 loss_seg_weight=loss_seg_weight,
                 classifier=classifier,
                 separate_head=separate_head,
                 share_conv_channel=share_conv_channel,
                 num_heatmap_convs=num_heatmap_convs,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 bias=bias,
                 norm_bbox=norm_bbox,
                 init_cfg=init_cfg,
                 task_specific=task_specific,
                 loss_prefix=loss_prefix)

    @force_fp32(apply_to=('preds_dicts', 'seg_outs'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, gt_masks_bev, preds_dicts, seg_outs, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)

        # ########## visualize heatmaps only ##########
        # import matplotlib.pyplot as plt
        # heatmap = heatmaps[1]
        # heatmap = heatmap.to('cpu')
        # for i in range(len(heatmaps[1])):
        #     print(i)
        #     print(gt_bboxes_3d[i].tensor.shape)
        #     print(gt_labels_3d[i])
        #     # plt.imsave('visualization/heatmap/heatmaps_{}.jpg'.format(i),heatmaps[0][i][0]*255)
        #     # plt.imsave('heatmaps_{}.jpg'.format(i),heatmaps[0][i][0]*255)
        #     plt.imsave('heatmaps_{}.jpg'.format(i),heatmap[i][0]*200+gt_masks_bev[i, 0].to('cpu')*50)
        # assert 0
        # ##############################################

        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            # seg_outs[0] = clip_sigmoid(seg_outs[0])
            # seg_outs[0] 8 4 128 128
            

            # import matplotlib.pyplot as plt
            # preds_dict[0]['heatmap'] = preds_dict[0]['heatmap'].to('cpu')
            # for i in range(len(preds_dict[0]['heatmap'])):
            #     print('i',i)
            #     print(preds_dict[0]['heatmap'][i][0].shape)
            #     # plt.imsave('pred_heatmaps_{}.jpg'.format(i),preds_dict[0]['heatmap'][i][0].detach()*125+seg_outs[0][i][0].detach().to('cpu')*125)
            #     plt.imsave('pred_heatmaps_{}.jpg'.format(i),preds_dict[0]['heatmap'][i][0].detach()*255)
            #     plt.imsave('seg_outs_0_{}.jpg'.format(i),seg_outs[0][i][0].detach().to('cpu')*255)
            #     plt.imsave('seg_outs_1_{}.jpg'.format(i),seg_outs[0][i][1].detach().to('cpu')*255)
            #     plt.imsave('seg_outs_2_{}.jpg'.format(i),seg_outs[0][i][2].detach().to('cpu')*255)
            #     plt.imsave('seg_outs_3_{}.jpg'.format(i),seg_outs[0][i][3].detach().to('cpu')*255)
            # assert 0

            heatmaps_comb=torch.cat((heatmaps[task_id],gt_masks_bev),dim=1) # 8*5*128*128
            
            preds_comb=torch.cat((preds_dict[0]['heatmap'],seg_outs[0]),1) # 8*5*128*128

            loss_heatmap_comb=sigmoid_focal_loss(preds_comb, heatmaps_comb, self.loss_seg_weight)

            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                    preds_dict[0]['dim'], preds_dict[0]['rot'],
                    preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            if self.task_specific:
                name_list=['xy','z','whl','yaw','vel']
                clip_index = [0,2,3,6,8,10]
                for reg_task_id in range(len(name_list)):
                    pred_tmp = pred[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    target_box_tmp = target_box[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    bbox_weights_tmp = bbox_weights[...,clip_index[reg_task_id]:clip_index[reg_task_id+1]]
                    loss_bbox_tmp = self.loss_bbox(
                        pred_tmp, target_box_tmp, bbox_weights_tmp, avg_factor=(num + 1e-4))
                    loss_dict[f'%stask{task_id}.loss_%s'%(self.loss_prefix,name_list[reg_task_id])] = loss_bbox_tmp
            else:
                loss_bbox = self.loss_bbox(
                    pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
                loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            loss_dict[f'%stask{task_id}.loss_heatmap'%(self.loss_prefix)] = loss_heatmap
            loss_dict[f'%stask{task_id}.loss_heatmap_comb'%(self.loss_prefix)] = loss_heatmap_comb
            
        ##### detected bev map masks / gt bev map & gt bboxes
        # seg_out=seg_outs[0]
        # seg_out[seg_out<0]=0
        # import matplotlib.pyplot as plt
        # import cv2
        # import os.path as path
        # import numpy as np
        # plt.figure()
        # fig, axs = plt.subplots(2, 4)
        # for i in range(len(seg_out)):
        #     for k, name in enumerate(self.map_classes):
        #         axs[0,k].imshow((seg_out[i,k].to('cpu')*50).byte())
        #         axs[0,k].invert_yaxis()
        #     for k, name in enumerate(self.map_classes):
        #         axs[1,k].imshow((gt_masks_bev[i,k].to('cpu')*50+heatmaps[0][i][0].to('cpu')*200).byte())
        #         # axs[1,k].imshow((gt_masks_bev[i,k].to('cpu')*50+temp*200).byte())
        #         axs[1,k].invert_yaxis()
        #     # plt.show()
        #     # plt.savefig(path.join('visualization/det_gt',name))
        #     plt.savefig('visualization/det_gt_test/{}_rot'.format(i))
        # assert 0
        ##########################################

        for index, name in enumerate(self.map_classes):
            loss_dict[f'task.loss_seg_{name}'] = sigmoid_focal_loss(seg_outs[0][:, index], gt_masks_bev[:, index], self.loss_seg_weight)

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        # preds_dicts = preds_dicts[0]

        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            # assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            nms_type = self.test_cfg.get('nms_type')
            if isinstance(nms_type,list):
                nms_type = nms_type[task_id]
            if nms_type == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas, task_id))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list



class BEVGridTransform(nn.Module):
    def __init__(
        self,
        input_scope,
        output_scope,
        **kwargs
    ):
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x

    
    