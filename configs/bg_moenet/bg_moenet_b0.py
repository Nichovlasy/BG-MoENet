# configs/bg_moenet/bg_moenet_b0.py

_base_ = [
   '../_base_/models/segformer_mit-b0.py',
   '../_base_/datasets/ttpla.py',
   '../_base_/default_runtime.py'
]

model = dict(
   data_preprocessor=dict(
       type='SegDataPreProcessor',
       mean=[123.675, 116.28, 103.53],
       std=[58.395, 57.12, 57.375],
       bgr_to_rgb=True,
       pad_val=0,
       seg_pad_val=255,
       size=(512, 512),
   ),
   decode_head=dict(
       type='BGMoENetHead',
       in_channels=[32, 64, 160, 256],
       in_index=[0, 1, 2, 3],
       channels=256,
       cmb_channels=256,
       c1_channels=64,
       dropout_ratio=0.1,
       num_classes=2,
       norm_cfg=dict(type='SyncBN', requires_grad=True),
       align_corners=False,

       # ===== CMB: Contextual Multi-branch Bridge =====
       cmb_dilations=(1, 6, 12, 18),
       enable_cmb_cross_level_gating=True,
       cmb_temperature=4.0,
       cmb_mix_coefficient=0.8,

       # ===== BG-PF: Boundary-Gated Progressive Fusion =====
       enable_bgpf_spatial_gating=True,
       enable_bgpf_strip_refinement=True,
       bgpf_strip_kernel=11,
       bgpf_strip_dilations=(1, 2),

       # ===== Boundary supervision =====
       boundary_loss_weight=0.12,
       boundary_foreground_index=1,   # powerline = 1
       boundary_widen_kernel=3,

       # ===== SA-MoE Fusion: Structure-Aware Mixture-of-Experts Fusion =====
       enable_sa_moe_fusion=True,
       sa_moe_kernel=11,
       sa_moe_routing_temperature=1.2,
       sa_moe_stabilization_coefficient=0.5,
       boundary_guidance_strength=0.6,
      
       loss_decode=[
           dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
           dict(type='DiceLoss', use_sigmoid=False, activate=True, reduction='mean',
                naive_dice=False, eps=1e-5, loss_weight=3.0)
       ]
   ),

   auxiliary_head=dict(
       type='FCNHead',
       in_channels=160,
       in_index=2,
       channels=64,
       num_convs=1,
       concat_input=False,
       dropout_ratio=0.1,
       num_classes=2,
       norm_cfg=dict(type='SyncBN', requires_grad=True),
       align_corners=False,
       loss_decode=[
           dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.4),
           dict(type='DiceLoss', use_sigmoid=False, activate=True, reduction='mean',
                naive_dice=False, eps=1e-5, loss_weight=1.2)
       ]
   ),

   backbone=dict(
       init_cfg=dict(type='Pretrained', checkpoint='checkpoints/mit_b0.pth')
   ),
   train_cfg=dict(),
   test_cfg=dict(mode='whole', crop_size=(512, 512))
)

optim_wrapper = dict(
   type='OptimWrapper',
   optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8),
   paramwise_cfg=dict(custom_keys={
       'absolute_pos_embed': dict(decay_mult=0.),
       'relative_position_bias_table': dict(decay_mult=0.),
       'norm': dict(decay_mult=0.),
       'head': dict(lr_mult=6.0)
   })
)

param_scheduler = [
   dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
   dict(type='CosineAnnealingLR', T_max=79000, eta_min=2e-6, by_epoch=False, begin=1000, end=80000)
]

train_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=4, persistent_workers=True)
test_dataloader = val_dataloader

val_evaluator = [
   dict(type='IoUMetric',
        iou_metrics=['mIoU', 'mDice', 'mFscore'],
        class_names=['background', 'powerline'],
        nan_to_num=0, ignore_index=255)
]
test_evaluator = val_evaluator

default_hooks = dict(
   timer=dict(type='IterTimerHook'),
   logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
   param_scheduler=dict(type='ParamSchedulerHook'),
   checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000,
                   max_keep_ckpts=3, save_best=['mIoU', 'mFscore'], rule='greater'),
   sampler_seed=dict(type='DistSamplerSeedHook'),
   visualization=dict(type='SegVisualizationHook', draw=True, interval=1000)
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

randomness = dict(seed=42, deterministic=False)