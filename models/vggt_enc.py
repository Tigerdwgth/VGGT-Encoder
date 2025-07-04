# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from coremltools.optimize.torch.quantization import (
    PostTrainingQuantizerConfig,
    PostTrainingQuantizer,
)

from diffusion_policy_3d.model.vision.vggt.models.aggregator import Aggregator
from diffusion_policy_3d.model.vision. vggt.heads.camera_head import CameraHead
from diffusion_policy_3d.model.vision.vggt.heads.dpt_head import DPTHead
from diffusion_policy_3d.model.vision.vggt.heads.track_head import TrackHead


class VGGT_ENC(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24):
        super().__init__()
        self.depth=depth
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,depth=depth)
     
    def load_weight_freeze(self):
        cache_dir='/home/jdh/nvme2/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9/model.safetensors'
        from safetensors.torch import load_file
        # 加载全部权重
        weights = load_file(cache_dir, device='cpu')  # 使用 CPU 设备加载权重
        agg_weights = {k: v for k, v in weights.items() if k.startswith("aggregator.")}
        self.aggregator.load_state_dict(agg_weights, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        # unfreeze self.aggregator.patch_embed 
        # for param in self.aggregator.patch_embed.parameters():
        #     param.requires_grad = True 
    def ptq_quantize(self,weight_dtype="int4"):
        # fp8_e5m2 int4 int8
        config=PostTrainingQuantizerConfig.from_dict(
            {
                "global_config": {
                    "weight_dtype": weight_dtype,  
                },
            }
        )
        ptq = PostTrainingQuantizer(self.aggregator, config)
        self.aggregator = ptq.compress()
        
        #freeze the parameters again
        for param in self.parameters():
            param.requires_grad = False
    
    def upload_weight(self):
        import os
        from safetensors.torch import save_file
        from huggingface_hub import HfApi

        # 定义保存路径
        fold_path = os.path.expanduser("~/nvme2/vggtenc")
        os.makedirs(fold_path, exist_ok=True)
        model_path = os.path.join(fold_path, "model.safetensors")

        # # 保存模型权重为 safetensors 格式
        save_file(self.state_dict(), model_path)

        # 上传到 Hugging Face
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.upload_folder(
            folder_path=fold_path,
            repo_id="T1g3rGE/VGGT_Enc",
            repo_type="model",
        )

    def quantize_to_int4(self):
        #quantize the model weight to int4
        import bitsandbytes as bnb
        from bitsandbytes.nn import Quantizer  # Example replacement

        for name, param in self.named_parameters():
            quantized_param = Quantizer(param.data, quant_type='int4')  # Adjust this based on the library docs
            setattr(self, name, quantized_param)

        print("Model weights have been quantized to int4.")
        
    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        # import pdb; pdb.set_trace()
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        
        features = aggregated_tokens_list[-1]  #b,w*h,dim # Get the last aggregated tokens for the current iteration
        # features = features[:,:,patch_start_idx:]  # Remove camera tokens
        # camera_tokens = aggregated_tokens_list[0][:, :, :patch_start_idx]  # Get camera tokens from the first iteration
        return features
       
if __name__=="__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # 初始化模型
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    modelpath='/home/jdh/nvme2/vggtenc'
    config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


    import time
    from torch.nn.functional import cosine_similarity
    model = VGGT_ENC().to("cuda")  
    model.load_weight_freeze()
    
    # 输入数据
    images = torch.randn(1, 3, 518, 518, device="cuda")  # Example input on GPU

    # 测试量化前的耗时和输出
    model.eval()
    
    with torch.no_grad():
        start_time = time.time()
        features_before = model(images)
        end_time = time.time()
    time_before = end_time - start_time
    print(f"量化前耗时: {time_before:.4f}秒")
    print(f"量化前输出特征形状: {features_before.shape}")



       
    # 测试量化后的耗时和输出
   
    with torch.no_grad():
        with torch.amp.autocast('cuda',dtype=dtype):
            start_time = time.time()
            features_after = model(images)
            end_time = time.time()
    time_after = end_time - start_time
    print(f"BF16后耗时: {time_after:.4f}秒")
    print(f"BF16后输出特征形状: {features_after.shape}")

    # 计算输出结果的余弦相似度
    cossim = cosine_similarity(features_before.flatten(), features_after.flatten(), dim=0)
    print(f"BF16前后输出结果的余弦相似度: {cossim.item():.4f}")
    
    model.to('cpu')
    model.ptq_quantize()
    model.to('cuda')
    with torch.no_grad():
        start_time = time.time()
        features_after_quantized = model(images)
        end_time = time.time()
    time_after_quantized = end_time - start_time
    print(f"量化后耗时: {time_after_quantized:.4f}秒")
    print(f"量化后输出特征形状: {features_after_quantized.shape}")
    # 计算量化后输出结果的余弦相似度
    cossim_quantized = cosine_similarity(features_before.flatten(), features_after_quantized.flatten(), dim=0)
    print(f"量化前后输出结果的余弦相似度: {cossim_quantized.item():.4f}")                   

    # 上传量化后的模型权重到 Hugging Face
    # model.upload_weight()

    # # 从 Hugging Face 加载量化后的模型
    # new_model = VGGT_ENC.from_pretrained("T1g3rGE/VGGT_Enc")
    # print("从 Hugging Face 加载量化后的模型成功")