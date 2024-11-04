# Copyright (c) Phigent Robotics. All rights reserved.

### the most efficient graph so far, 59.7% accuracy

_base_ = ['../_base_/datasets/nus-3d.py',
          '../_base_/schedules/cyclic_20e.py',
          '../_base_/default_runtime.py']

plugin=True
plugin_dir='mmdet3d_plugin/'

# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test':0.04,
}

# Model
grid_config={
        'xbound': [-51.2, 51.2, 0.8],
        'ybound': [-51.2, 51.2, 0.8],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],
        'map_classes': ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','divider'
        ]}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans=64

model = dict(
    type='BEVDet_Multi',
    img_backbone=dict(
        type='SwinTransformer',
        pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3,),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN_LSS',
        in_channels=384+768,
        out_channels=512,
        extra_upsample=None,
        input_feature_index=(0,1),
        scale_factor=2),
    img_view_transformer=dict(type='ViewTransformerLiftSplatShoot',
                              grid_config=grid_config,
                              data_config=data_config,
                              numC_Trans=numC_Trans),
    img_bev_encoder_backbone = dict(type='ResNetForBEVDet_graph_spatial', numC_input=numC_Trans*2),
    img_bev_encoder_neck = dict(type='FPN_LSS',
                                in_channels=(numC_Trans*8+numC_Trans*2)*2,
                                out_channels=256),
    pts_bbox_head=dict(
        type='CenterMapHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        grid_config=grid_config,
        grid_transform={
            'input_scope':[[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]],
            'output_scope':[[-50, 50, 0.5], [-50, 50, 0.5]],
            'loss':'focal',
        },
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean',loss_weight=0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0),
        loss_seg_weight=10,
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            # nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            # nms_thr=0.2,

            # Scale-NMS
            nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]]
        )))


# Data
dataset_type = 'NuScenesDatasetMap'
data_root = 'data/nuscenes/'
data_root_other = 'data/nuscenes-hsda/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTransMap',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3DMap',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(type='LoadBEVMap', is_train=True, data_config=data_config, grid_config=grid_config, dataset_root=data_root),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'img_info'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=False, data_config=data_config),
    # load lidar points for --show in test.py only
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3DMap',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='LoadBEVMap', is_train=False, data_config=data_config, grid_config=grid_config, dataset_root=data_root),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points','img_inputs', 'gt_masks_bev'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=False, data_config=data_config),
    # load lidar points for --show in test.py only
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3DMap',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='LoadBEVMap', is_train=False, data_config=data_config, grid_config=grid_config, dataset_root=data_root),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points','img_inputs', 'gt_masks_bev'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDatasetMixed',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train_bev.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            map_classes=grid_config['map_classes'],
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            img_info_prototype='bevdet'),
        other_dataset=dict(
            type=dataset_type,
            data_root=data_root_other,
            ann_file=data_root_other + 'nuscenes_infos_train_bev.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            map_classes=grid_config['map_classes'],
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_val_bev.pkl',pipeline=test_pipeline, classes=class_names, map_classes=grid_config['map_classes'],
        modality=input_modality, img_info_prototype='bevdet'),
    test=dict(type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_val_bev.pkl',pipeline=test_pipeline, classes=class_names, map_classes=grid_config['map_classes'],
        modality=input_modality, img_info_prototype='bevdet'))

# Optimizer
lr_config = dict(
    policy='cyclic',
    target_ratio=(5, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
evaluation = dict(interval=20, pipeline=eval_pipeline)
# load_from='chpt/bevdet-sttiny-pure.pth'
# load_from='work_dirs/bevdet-multi-config/epoch_1.pth'

