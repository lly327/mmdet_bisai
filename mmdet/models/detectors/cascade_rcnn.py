# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


@MODELS.register_module()
class CascadeRCNN(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 pretrained = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            pretrained=pretrained)
        if pretrained:
            pretrained = torch.load(pretrained, map_location='cpu')['state_dict']
            self.load_state_dict(pretrained, strict=False)
            print('loaded pretrained successfully')
            # with open('1.txt','w') as f:
            #     f.write(str(pretrained.keys()))
            # with open('2.txt','w') as f:
            #     f.write(str(self.state_dict().keys()))
            # exit(0)
