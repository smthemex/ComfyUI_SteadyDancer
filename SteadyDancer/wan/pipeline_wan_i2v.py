# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tqdm import tqdm
import html
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import PIL
import math
import numpy as np
import regex as re
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
import torch.cuda.amp as amp
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import gc
import random,sys

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> import numpy as np
        >>> from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_image
        >>> from transformers import CLIPVisionModel

        >>> # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        >>> image_encoder = CLIPVisionModel.from_pretrained(
        ...     model_id, subfolder="image_encoder", torch_dtype=torch.float32
        ... )
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanImageToVideoPipeline.from_pretrained(
        ...     model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )
        >>> max_area = 480 * 832
        >>> aspect_ratio = image.height / image.width
        >>> mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        >>> height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        >>> width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        >>> image = image.resize((width, height))
        >>> prompt = (
        ...     "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
        ...     "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        ... )
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     image=image,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=height,
        ...     width=width,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""

I2V14B_VAE_CONFIG={
    "latents_mean": [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921
    ],
    "latents_std": [
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.916
    ],
    "z_dim": 16
    }


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")




class WanImageToVideoPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for image-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer_2 ([`WanTransformer3DModel`], *optional*):
            Conditional Transformer to denoise the input latents during the low-noise stage. In two-stage denoising,
            `transformer` handles high-noise stages and `transformer_2` handles low-noise stages. If not provided, only
            `transformer` is used.
        boundary_ratio (`float`, *optional*, defaults to `None`):
            Ratio of total timesteps to use as the boundary for switching between transformers in two-stage denoising.
            The actual boundary timestep is calculated as `boundary_ratio * num_train_timesteps`. When provided,
            `transformer` handles timesteps >= boundary_timestep and `transformer_2` handles timesteps <
            boundary_timestep. If `None`, only `transformer` is used for the entire denoising process.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2", "image_encoder", "image_processor"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer: WanTransformer3DModel = None,
        transformer_2: WanTransformer3DModel = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal =  4
        self.vae_scale_factor_spatial =  8
        # self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        # self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor

    def load_lora_weights_(self, lora_path: str, adapter_name: str = "default", weight_name: str = None):
        """
        为WAN非diffusers版本的DiT模型加载LoRA权重
        
        Args:
            lora_path (str): LoRA权重路径
            adapter_name (str): 适配器名称
            weight_name (str, optional): 权重文件名
        """
        try:
            # 直接为transformer/DiT模型加载LoRA权重
            if hasattr(self, 'transformer') and self.transformer is not None:
                # 使用自定义的LoRA加载方法
                state_dict = self._load_lora_weights_to_dit(
                    self.transformer, 
                    lora_path, 
                    weight_name=weight_name,
                    adapter_name=adapter_name
                )
                print(f"Successfully loaded LoRA weights to DiT transformer from {lora_path}")
                return state_dict
            else:
                raise ValueError("No transformer (DiT) model found in the pipeline")
                
        except Exception as e:
            print(f"Error loading LoRA weights to DiT: {e}")
            

    def _load_lora_weights_to_dit(self, dit_model, lora_path: str, weight_name: str = None, adapter_name: str = "default"):
        """
        专门为DiT模型加载LoRA权重的内部方法
        
        Args:
            dit_model: DiT模型实例
            lora_path: LoRA权重路径
            weight_name: 权重文件名
            adapter_name: 适配器名称
        """
        import os
        import json
        from safetensors.torch import load_file
        
        # 处理不同的路径情况
        if os.path.isfile(lora_path):
            # 如果是文件路径
            if lora_path.endswith('.safetensors'):
                state_dict = load_file(lora_path)
            else:
                state_dict = torch.load(lora_path, map_location='cpu')
        elif os.path.isdir(lora_path):
            # 如果是目录路径，查找权重文件
            weight_files = []
            for file in os.listdir(lora_path):
                if file.endswith(('.pt', '.pth', '.bin', '.safetensors')):
                    weight_files.append(file)
            
            # 根据weight_name选择文件或使用默认规则
            target_file = None
            if weight_name:
                for file in weight_files:
                    if weight_name in file:
                        target_file = file
                        break
            else:
                # 优先选择包含'lora'的文件
                for file in weight_files:
                    if 'lora' in file.lower():
                        target_file = file
                        break
                # 如果没有找到，选择第一个文件
                if not target_file and weight_files:
                    target_file = weight_files[0]
            
            if target_file:
                full_path = os.path.join(lora_path, target_file)
                if full_path.endswith('.safetensors'):
                    state_dict = load_file(full_path)
                else:
                    state_dict = torch.load(full_path, map_location='cpu')
            else:
                raise FileNotFoundError(f"No valid weight files found in {lora_path}")
        else:
            # 假设是HuggingFace模型标识符
            try:
                from huggingface_hub import hf_hub_download
                # 下载权重文件
                if weight_name:
                    file_path = hf_hub_download(repo_id=lora_path, filename=weight_name)
                else:
                    # 尝试下载常见的LoRA权重文件名
                    common_names = ['pytorch_lora_weights.bin', 'lora_weights.bin', 'diffusion_pytorch_model.bin']
                    file_path = None
                    for name in common_names:
                        try:
                            file_path = hf_hub_download(repo_id=lora_path, filename=name)
                            break
                        except:
                            continue
                    
                    if not file_path:
                        raise FileNotFoundError(f"Could not find LoRA weights in {lora_path}")
                
                if file_path.endswith('.safetensors'):
                    state_dict = load_file(file_path)
                else:
                    state_dict = torch.load(file_path, map_location='cpu')
            except Exception as e:
                raise FileNotFoundError(f"Cannot find or download LoRA weights from {lora_path}: {e}")
        
        # 应用LoRA权重到DiT模型
        self._apply_lora_weights_to_dit(dit_model, state_dict, adapter_name)
        
        return state_dict

    def _apply_lora_weights_to_dit(self, dit_model, state_dict: Dict[str, torch.Tensor], adapter_name: str = "default"):
        """
        将LoRA权重应用到DiT模型
        
        Args:
            dit_model: DiT模型实例
            state_dict: LoRA权重字典
            adapter_name: 适配器名称
        """
        # 创建适配器存储（如果不存在）
        if not hasattr(self, '_lora_adapters'):
            self._lora_adapters = {}
        
        if adapter_name not in self._lora_adapters:
            self._lora_adapters[adapter_name] = {}
        
        # 遍历状态字典并应用权重
        for key, value in state_dict.items():
            # 解析键名以确定目标模块
            # 通常LoRA权重会有特定的命名约定，例如:
            # "transformer.blocks.0.attn.qkv.lora_A.weight"
            # "transformer.blocks.0.attn.qkv.lora_B.weight"
            
            try:
                # 获取目标模块
                module_keys = key.split('.')[:-1]  # 移除最后的weight/bias部分
                target_module = dit_model
                for module_key in module_keys:
                    if hasattr(target_module, module_key):
                        target_module = getattr(target_module, module_key)
                    elif module_key.isdigit() and isinstance(target_module, (list, tuple)):
                        target_module = target_module[int(module_key)]
                    else:
                        # 如果找不到模块，跳过这个权重
                        #print(f"Warning: Could not find module for key {key}")
                        break
                else:
                    # 成功找到目标模块
                    param_name = key.split('.')[-1]  # 获取参数名称 (weight/bias)
                    
                    # 保存LoRA权重以便后续使用
                    self._lora_adapters[adapter_name][key] = value.clone()
                    
                    # 注意：不直接注入权重到模型中，等待set_lora_adapter调用
                    # 如果需要直接注入权重到模型中（适用于某些实现），取消注释以下代码
                    # if hasattr(target_module, param_name):
                    #     existing_param = getattr(target_module, param_name)
                    #     if existing_param.shape == value.shape:
                    #         setattr(target_module, param_name, torch.nn.Parameter(value))
                    #         #print(f"Injected LoRA weight for {key}")
                    #     else:
                    #         pass
                    #         #print(f"Shape mismatch for {key}: expected {existing_param.shape}, got {value.shape}")
                            
            except Exception as e:
                print(f"Warning: Failed to apply LoRA weight {key}: {e}")

    def set_lora_adapter(self, adapter_name: str = "default", weight: float = 1.0):
        """
        激活特定的LoRA适配器
        
        Args:
            adapter_name: 要激活的适配器名称
            weight: 适配器权重
        """
        if not hasattr(self, '_lora_adapters') or adapter_name not in self._lora_adapters:
            print(f"Adapter {adapter_name} not found")
            return
        
        # 应用LoRA权重到模型
        # 这里简化实现：直接替换参数
        # 注意：这不是标准的LoRA实现，标准的LoRA应该添加低秩适配器
        adapter_weights = self._lora_adapters[adapter_name]
        
        # 遍历所有保存的权重并应用到模型
        for key, value in adapter_weights.items():
            try:
                # 解析键名以确定目标模块
                module_keys = key.split('.')[:-1]  # 移除最后的weight/bias部分
                target_module = self.transformer  # 注意：这里使用transformer而不是model
                for module_key in module_keys:
                    if hasattr(target_module, module_key):
                        target_module = getattr(target_module, module_key)
                    elif module_key.isdigit() and isinstance(target_module, (list, tuple)):
                        target_module = target_module[int(module_key)]
                    else:
                        break
                else:
                    # 成功找到目标模块
                    param_name = key.split('.')[-1]  # 获取参数名称 (weight/bias)
                    
                    if hasattr(target_module, param_name):
                        existing_param = getattr(target_module, param_name)
                        if existing_param.shape == value.shape:
                            # 应用LoRA权重（这里简化：直接替换）
                            # 标准的LoRA应该是：new_weight = original_weight + weight * lora_weight
                            lora_weight = value * weight
                            new_weight = existing_param.data + lora_weight.to(existing_param.device)
                            setattr(target_module, param_name, torch.nn.Parameter(new_weight))
            except Exception as e:
                print(f"Warning: Failed to apply LoRA weight {key}: {e}")
        
        print(f"Set LoRA adapter to {adapter_name} with weight {weight}")

    def unload_lora_weights(self):
        """
        卸载LoRA权重
        """
        if hasattr(self, '_lora_adapters'):
            self._lora_adapters.clear()
            print("Unloaded all LoRA weights")      

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: PipelineImageInput,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

        if self.config.boundary_ratio is not None and image_embeds is not None:
            raise ValueError("Cannot forward `image_embeds` when the pipeline's `boundary_ratio` is not configured.")

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        cond_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        if image is not None:
            
            image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

            if self.config.expand_timesteps:
                video_condition = image

            elif last_image is None:
                video_condition = torch.cat(
                    [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
                )
            else:
                last_image = last_image.unsqueeze(2)
                video_condition = torch.cat(
                    [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                    dim=2,
                )
            video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )

            if isinstance(generator, list):
                latent_condition = [
                    retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
                ]
                latent_condition = torch.cat(latent_condition)
            else:
                latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
                latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            latent_condition = latent_condition.to(dtype)
            latent_condition = (latent_condition - latents_mean) * latents_std

        else:
           
            latents_mean = (
                torch.tensor(I2V14B_VAE_CONFIG["latents_mean"])
                .view(1,I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1)
                .to(cond_latents.device, cond_latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(I2V14B_VAE_CONFIG["latents_std"]).view(1, I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1).to(
                cond_latents.device, cond_latents.dtype
            )
            latent_condition = (cond_latents - latents_mean) * latents_std

        if self.config.expand_timesteps:
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
            )
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_embeds_c: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "latent",
        return_dict: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        condition_pos=None,
        condition_null=None,
        sample_solver="unipc",
        sample_shift=5.0,
        offload_model=True,
        condition_guide_scale=1.0,
        ref_x=None,
        ref_c=None,
        st_cond_cfg=0.1,
        end_cond_cfg=0.5,
        wrapper=True,
        seed: int = -1,
        
        
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
                image embeddings are generated from the `image` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        self.st_cond_cfg, self.end_cond_cfg = st_cond_cfg, end_cond_cfg
        if image is not None:
            height,width = image.shape[2]*8,image.shape[3]*8
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     negative_prompt,
        #     image,
        #     height,
        #     width,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     image_embeds,
        #     callback_on_step_end_tensor_inputs,
        #     guidance_scale_2,
        # )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        if prompt is not None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )


        # Encode image embedding
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype

        prompt_embeds=prompt_embeds.to(device,self.transformer.dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds=negative_prompt_embeds.to(device,self.transformer.dtype)

        # only wan 2.1 i2v transformer accepts image_embeds
        # if self.transformer is not None and self.transformer.config.image_dim is not None:
        #     if image_embeds is None:
        #         if last_image is None:
        #             image_embeds = self.encode_image(image, device)
        #         else:
        #             image_embeds = self.encode_image([image, last_image], device)
            
        #     image_embeds = image_embeds.repeat(batch_size, 1, 1)
        #     image_embeds = image_embeds.to(transformer_dtype)

        image_embeds=image_embeds.to(device,self.transformer.dtype)
        image_embeds_c=image_embeds_c.to(device,self.transformer.dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = int(I2V14B_VAE_CONFIG["z_dim"])

        # image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        # if last_image is not None:
        #     last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
        #         device, dtype=torch.float32
        #     )

        # latents_outputs = self.prepare_latents(
        #     None,
        #     batch_size * num_videos_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     num_frames,
        #     torch.float32,
        #     device,
        #     generator,
        #     latents,
        #     last_image,
        #     cond_latents=image,
        # )
   
        # latents, condition = latents_outputs

        #noise=latents[0] #TODO
        # noise = torch.randn(
        #     16, (num_frames - 1) // 4 + 1,
        #     lat_h,
        #     lat_w,
        #     dtype=torch.float32,
        #     generator=generator,
        #     device=self.device)
        # condition=condition[0]

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        vae_stride = (4, 8, 8)
        patch_size = (1, 2, 2)    
        max_area=width*height
        aspect_ratio = height / width
        sp_size=1
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
            patch_size[1] * patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
            patch_size[2] * patch_size[2])
        max_seq_len = ((num_frames - 1) // vae_stride[0] + 1) * lat_h * lat_w // (
            patch_size[1] * patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / sp_size)) * sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (num_frames - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, num_frames, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        condition = torch.concat([msk, image])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.transformer, 'no_sync', noop_no_sync)


        # evaluation mode
        with amp.autocast(dtype=torch.bfloat16), torch.no_grad(), no_sync():

            #if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                num_inference_steps, device=self.device, shift=sample_shift)
            timesteps = sample_scheduler.timesteps
            # elif sample_solver == 'dpm++':
            #     sample_scheduler = FlowDPMSolverMultistepScheduler(
            #         num_train_timesteps=self.num_train_timesteps,
            #         shift=1,
            #         use_dynamic_shifting=False)
            #     sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            #     timesteps, _ = retrieve_timesteps(
            #         sample_scheduler,
            #         device=self.device,
            #         sigmas=sampling_sigmas)
            # else:
            #     raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [prompt_embeds[0]], #TODO
                'clip_fea_c': image_embeds_c,
                'clip_fea_x': image_embeds,
                'seq_len': max_seq_len,
                'y': [condition],
                'condition': condition_pos,
                'ref_c': ref_c,
                'ref_x': ref_x,
            }

            arg_null_context = {
                'context': negative_prompt_embeds,        # null context
                'clip_fea_c': image_embeds_c,
                'clip_fea_x': image_embeds,
                'seq_len': max_seq_len,
                'y': [condition],
                'condition': condition_pos,
                'ref_c': ref_c,
                'ref_x': ref_x,
            }

            arg_null_condition = {
                'context': [prompt_embeds[0]],
                'clip_fea_c': image_embeds_c,
                'clip_fea_x': image_embeds,
                'seq_len': max_seq_len,
                'y': [condition],
                'condition': condition_null,   # null condition
                'ref_c': ref_c,
                'ref_x': ref_x,
            }

            if offload_model:
                torch.cuda.empty_cache()
            do_classifier_free_guidance=True
            #self.transformer.to(self.device)
            for idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)
                
                noise_pred_cond = self.transformer(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if do_classifier_free_guidance:
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond_context = self.transformer(
                        latent_model_input, t=timestep, **arg_null_context)[0].to(
                            torch.device('cpu') if offload_model else self.device)
                    if offload_model:
                        torch.cuda.empty_cache()

                    if idx / len(timesteps) > self.st_cond_cfg and idx / len(timesteps) < self.end_cond_cfg:
                        noise_pred_uncond_condition = self.transformer(
                            latent_model_input, t=timestep, **arg_null_condition)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()

                        cond_context = noise_pred_cond - noise_pred_uncond_context
                        cond_condition = noise_pred_cond - noise_pred_uncond_condition                    
                        noise_pred = noise_pred_uncond_context + guidance_scale * cond_context + condition_guide_scale * cond_condition
                    else:
                        cond_context = noise_pred_cond - noise_pred_uncond_context
                        noise_pred = noise_pred_uncond_context + guidance_scale * cond_context
                else:
                    noise_pred = noise_pred_cond
                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=generator)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.transformer.cpu()
                torch.cuda.empty_cache()

            if  not wrapper:
                videos = self.vae.decode(x0)
            else:         
                videos = x0
        del latents, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        
        return videos[0]

