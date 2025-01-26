# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

# from util.misc import NestedTensor, is_main_process


from .position_encoding import build_position_encoding

import os, sys
PATH = os.getcwd()
sys.path.append(PATH)

import importlib

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool):
        super().__init__()

        self.backbone = backbone

        if not train_backbone:
            for name, parameter in backbone.named_parameters():
                parameter.requires_grad = False

    def forward(self, tensor):
        features, pos_embed = self.backbone(tensor)

        return features, pos_embed


class Backbone(BackboneBase):
    """3DETR backbone"""
    def __init__(self, name: str,
                 args,
                 train_backbone: bool):
        spec_3detr = importlib.util.spec_from_file_location("act_3detr", "3detr/models/act_3detr.py")
        module_3detr = importlib.util.module_from_spec(spec_3detr)
        spec_3detr.loader.exec_module(module_3detr)
        backbone = module_3detr.build_3detr(args)

        # Load pretrained weights
        checkpoint = torch.load(args.ckpt_pth)
        backbone.load_state_dict(checkpoint["model"])

        self.d_model = backbone.num_queries

        super().__init__(backbone, train_backbone)



def build_backbone(args):
    train_backbone = args.lr_backbone > 0

    model = Backbone(args.backbone, args, train_backbone)
    return model
