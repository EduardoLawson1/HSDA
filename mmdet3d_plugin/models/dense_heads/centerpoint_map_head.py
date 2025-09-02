# Copyright (c) OpenMMLab. All rights reserved.
import copy
from distutils.command.config import config
import imp
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
class CenterMapHead(CenterHead):
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

        # ########## visualize heatmaps only ##########
        # import matplotlib.pyplot as plt
        # heatmaps[0] = heatmaps[0].to('cpu')
        # for i in range(len(heatmaps[0])):
        #     print(i)
        #     print(gt_bboxes_3d[i].tensor.shape)
        #     print(gt_labels_3d[i])
        #     plt.imsave('visualization/heatmap/heatmaps_{}.jpg'.format(i),heatmaps[0][i][0]*255)
        # assert 0
        # ##############################################

        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
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
        
        ##### detected bev map masks / gt bev map & gt bboxes
        seg_out=seg_outs[0]
        seg_out[seg_out<0]=0
        import matplotlib.pyplot as plt
        import cv2
        import os.path as path
        import numpy as np
        import os
        from matplotlib.patches import Rectangle
        
        # Criar diretório se não existir
        os.makedirs('visualization/det_gt_test', exist_ok=True)
        
        # Criar visualização estilo profissional como no exemplo de referência
        for i in range(min(len(seg_out), 2)):  # Processar apenas 2 amostras
            
            # Configurar figura com layout profissional
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.2)
            
            # Título principal
            fig.suptitle(f'HSDA Model - BEV Segmentation Results (Sample {i+1})', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Linha 1: Predições do modelo
            for k, name in enumerate(self.map_classes):
                if k < 4:  # Máximo 4 classes
                    ax = fig.add_subplot(gs[0, k])
                    
                    # Converter tensor para imagem
                    pred_img = (seg_out[i,k].to('cpu') * 255).byte().numpy()
                    
                    # Aplicar colormap para melhor visualização
                    import matplotlib.cm as cm
                    colored_pred = cm.viridis(pred_img / 255.0)
                    
                    ax.imshow(colored_pred)
                    ax.set_title(f'Prediction: {name.replace("_", " ").title()}', 
                               fontsize=11, fontweight='bold')
                    ax.axis('off')
                    ax.invert_yaxis()
            
            # Linha 2: Ground Truth
            for k, name in enumerate(self.map_classes):
                if k < 4:  # Máximo 4 classes
                    ax = fig.add_subplot(gs[1, k])
                    
                    # Converter tensor para imagem
                    gt_img = (gt_masks_bev[i,k].to('cpu') * 255).byte().numpy()
                    
                    # Aplicar colormap
                    colored_gt = cm.plasma(gt_img / 255.0)
                    
                    ax.imshow(colored_gt)
                    ax.set_title(f'Ground Truth: {name.replace("_", " ").title()}', 
                               fontsize=11, fontweight='bold')
                    ax.axis('off')
                    ax.invert_yaxis()
            
            # Adicionar informações técnicas
            ax_info = fig.add_subplot(gs[0:2, 4:])
            ax_info.axis('off')
            
            info_text = f"""
MODEL INFORMATION:
• Architecture: BEVDet Multi-Camera
• Dataset: nuScenes Mini
• Task: BEV Semantic Segmentation
• Input: 6 Cameras (360° View)
• Output Resolution: {seg_out.shape[-2]}x{seg_out.shape[-1]}

SEGMENTATION CLASSES:
• Drivable Area: Road surface for vehicles
• Ped Crossing: Pedestrian crossings
• Walkway: Sidewalks and pedestrian areas  
• Divider: Lane dividers and barriers

METRICS (Sample {i+1}):
• Prediction Range: [{seg_out[i].min().item():.3f}, {seg_out[i].max().item():.3f}]
• GT Coverage: {(gt_masks_bev[i].sum() / gt_masks_bev[i].numel() * 100).item():.1f}%
• Classes Detected: {(seg_out[i].max(dim=0)[0] > 0.1).sum().item()}/{len(self.map_classes)}

HSDA AUGMENTATION:
High-frequency Shuffle Data Augmentation
improves model robustness through strategic
data permutation during training.
            """
            
            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            # Legenda das cores na parte inferior
            ax_legend = fig.add_subplot(gs[2, :])
            ax_legend.axis('off')
            
            # Criar barra de cores personalizada
            colors = ['#440154', '#31688e', '#35b779', '#fde725']  # Viridis colors
            labels = [cls.replace('_', ' ').title() for cls in self.map_classes[:4]]
            
            for idx, (color, label) in enumerate(zip(colors, labels)):
                rect = Rectangle((idx*0.2, 0.3), 0.15, 0.4, facecolor=color, alpha=0.8)
                ax_legend.add_patch(rect)
                ax_legend.text(idx*0.2 + 0.075, 0.1, label, ha='center', va='center', 
                             fontsize=9, fontweight='bold')
            
            ax_legend.set_xlim(0, 0.8)
            ax_legend.set_ylim(0, 1)
            ax_legend.text(0.4, 0.8, 'Segmentation Classes Color Map', 
                          ha='center', fontsize=12, fontweight='bold')
            
            # Salvar com nome descritivo
            output_file = f'visualization/det_gt_test/hsda_professional_sample_{i+1}.png'
            plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Visualização profissional salva: {output_file}")
        
        print(f"🎯 Visualizações no estilo do exemplo de referência criadas!")
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


class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 nn.BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out

class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = nn.BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = nn.BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = nn.BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out

@HEADS.register_module()
class CenterGraphHead(CenterMapHead):
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

        self.dualgcn = DualGCN(in_channels)

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.dualgcn(x)

        seg_x = self.classifier(x)
        x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts, seg_x



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

    
    
