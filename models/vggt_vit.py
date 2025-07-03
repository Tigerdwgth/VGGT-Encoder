
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from diffusion_policy_3d.model.vision.vggt.layers import PatchEmbed
from diffusion_policy_3d.model.vision.vggt.layers.block import Block
from diffusion_policy_3d.model.vision.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from diffusion_policy_3d.model.vision.vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

# def VGGT_ViT()