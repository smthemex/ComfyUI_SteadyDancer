# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import comfy.utils
warnings.filterwarnings('ignore')
from diffusers import  GGUFQuantizationConfig
import random

import torch
import torch.distributed as dist
from PIL import Image
from .wan import WanI2VDancer
from . import wan
from .wan.modules.vae import WanVAE
from .wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
#from .wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from .wan.utils.utils import cache_image, cache_video, str2bool
from .wan.configs.wan_i2v_14B import i2v_14B
from contextlib import contextmanager
from .wan.pipeline_wan_i2v import I2V14B_VAE_CONFIG
@contextmanager
def temp_patch_module_attr(module_name: str, attr_name: str, new_obj):
    mod = sys.modules.get(module_name)
    if mod is None:
        yield
        return
    had = hasattr(mod, attr_name)
    orig = getattr(mod, attr_name, None)
    setattr(mod, attr_name, new_obj)
    try:
        yield
    finally:
        if had:
            setattr(mod, attr_name, orig)
        else:
            try:
                delattr(mod, attr_name)
            except Exception:
                pass


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--cond_pos_folder",
        type=str,
        default=None,
        help="The positive condition folder that contains all types of inputs")
    parser.add_argument(
        "--cond_neg_folder",
        type=str,
        default=None,
        help="The negative condition folder that contains all types of inputs")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--condition_guide_scale",
        type=float,
        default=1.5,
        help="Classifier free guidance scale, specific to the condition.")
    parser.add_argument(
        "--st_cond_cfg",
        type=float,
        default=0.1,
        help="Begin cfg with cond_neg_folder.")
    parser.add_argument(
        "--end_cond_cfg",
        type=float,
        default=0.4,
        help="End cfg with cond_neg_folder.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def load_models(args,is_wrapper=True):
    cfg = i2v_14B
    logging.info("Creating WanI2V pipeline.")
    if is_wrapper:
        from .wan.transformer_wan import WanTransformer3DModelSD
        from .wan.pipeline_wan_i2v import WanImageToVideoPipeline
        with temp_patch_module_attr("diffusers", "WanTransformer3DModel", WanTransformer3DModelSD):
            if args.gguf_path is not None:
                transformer = WanTransformer3DModelSD.from_single_file(
                    args.gguf_path,
                    config=args.config_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16,
                    )
            elif args.dit_path is not None:
                transformer = WanTransformer3DModelSD.from_single_file(
                    args.dit_path,
                    config=args.config_path,
                    torch_dtype=torch.bfloat16,)   
            else:
                raise ValueError("Please specify either gguf_ckpt or dit_path.")  
        wan_i2v = WanImageToVideoPipeline.from_pretrained(args.repo,transformer=transformer,vae=None, text_encoder=None,image_encoder=None,torch_dtype=torch.bfloat16)
    else:
        wan_i2v = wan.WanI2VDancer(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            st_cond_cfg=args.st_cond_cfg, end_cond_cfg=args.end_cond_cfg,
        )

    return wan_i2v


def pre_data_process(vae,image,pose_video,pose_video_negative,num_frames,width,height,device):
    orginal_image = image.clone().detach()
    if isinstance(vae, WanVAE):
        model_device = next(vae.model.parameters()).device
        if model_device != device:
            vae.model.to(device)
        image=image.sub_(0.5).div_(0.5).squeeze(0).permute(2, 0, 1).to(vae.device) #bhwc-->chw
        latent = vae.encode([
                    torch.concat([
                        torch.nn.functional.interpolate(
                            image[None].cpu(), size=(height, width), mode='bicubic').transpose(
                                0, 1),
                        torch.zeros(3, num_frames - 1, height, width)
                    ],
                                dim=1).to(vae.device)
                ])[0]   # img[3, H, W]

        # ref img_x
        img_x=orginal_image.sub_(0.5).div_(0.5).squeeze(0).permute(2, 0, 1).to(vae.device) #bhwc-->chw
        ref_x = vae.encode([
                    torch.nn.functional.interpolate(
                        img_x[None].cpu(), size=(height, width), mode='bicubic').transpose(
                        0, 1).to(vae.device)
                ])[0]
        msk_ref = torch.ones(4, 1, ref_x.shape[2], ref_x.shape[3], device=ref_x.device)
        ref_x = torch.concat([ref_x, msk_ref, ref_x])

        # ref img_c
        img_c=pose_video[:1, :, :, :3].sub_(0.5).div_(0.5).squeeze(0).permute(2, 0, 1).to(vae.device) #bhwc-->chw
        ref_c = vae.encode([
                    torch.nn.functional.interpolate(
                        img_c[None].cpu(), size=(height, width), mode='bicubic').transpose(
                        0, 1).to(vae.device)
                ])[0]
        msk_c = torch.zeros(4, 1, ref_c.shape[2], ref_c.shape[3], device=ref_c.device)
        ref_c = torch.concat([ref_c, msk_c, ref_c])
      
        conditions =  vae.encode([pose_video[:, :, :, :3].sub_(0.5).div_(0.5).permute(3, 0, 1,2).to(vae.device)])[0]
        condition_null = vae.encode([pose_video_negative[:, :, :, :3].sub_(0.5).div_(0.5).permute(3, 0, 1,2).to(vae.device)])[0]
        vae.model.to("cpu")
    else: 
        orginal_image = image.clone().detach()
        start_image = comfy.utils.common_upscale(image[:num_frames].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        image = torch.ones((num_frames, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
        image[:start_image.shape[0]] = start_image
        latent = vae.encode(image[:, :, :, :3])
        latents_mean = (
                torch.tensor(I2V14B_VAE_CONFIG["latents_mean"])
                .view(1,I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
        latents_std = 1.0 / torch.tensor(I2V14B_VAE_CONFIG["latents_std"]).view(1, I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1).to(
            latent.device, latent.dtype
        )
        latent = ((latent - latents_mean) * latents_std)[0]

        # ref img_x
        ref_x=vae.encode(orginal_image[:1, :, :, :3])
        latents_mean = (
                torch.tensor(I2V14B_VAE_CONFIG["latents_mean"])
                .view(1,I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1)
                .to(ref_x.device, ref_x.dtype)
            )
        latents_std = 1.0 / torch.tensor(I2V14B_VAE_CONFIG["latents_std"]).view(1, I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1).to(
            ref_x.device, ref_x.dtype
        )
        ref_x = ((ref_x - latents_mean) * latents_std)[0]

        msk_ref = torch.ones(4, 1, ref_x.shape[2], ref_x.shape[3], device=ref_x.device)
        ref_x = torch.concat([ref_x, msk_ref, ref_x])

        # ref img_c
        ref_c=vae.encode(pose_video[:1, :, :, :3])
        latents_mean = (
                torch.tensor(I2V14B_VAE_CONFIG["latents_mean"])
                .view(1,I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1)
                .to(ref_c.device, ref_c.dtype)
            )
        latents_std = 1.0 / torch.tensor(I2V14B_VAE_CONFIG["latents_std"]).view(1, I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1).to(
            ref_c.device, ref_c.dtype
        )
        ref_c = ((ref_c - latents_mean) * latents_std)[0]
        msk_c = torch.zeros(4, 1, ref_c.shape[2], ref_c.shape[3], device=ref_c.device)
        ref_c = torch.concat([ref_c, msk_c, ref_c])

        conditions =  vae.encode(pose_video[:, :, :, :3])
        condition_null = vae.encode(pose_video_negative[:, :, :, :3])
        latents_mean = (
                torch.tensor(I2V14B_VAE_CONFIG["latents_mean"])
                .view(1,I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1)
                .to(conditions.device, conditions.dtype)
            )
        latents_std = 1.0 / torch.tensor(I2V14B_VAE_CONFIG["latents_std"]).view(1, I2V14B_VAE_CONFIG["z_dim"], 1, 1, 1).to(
            conditions.device, conditions.dtype
        )
        conditions = ((conditions - latents_mean) * latents_std)[0]
        condition_null = ((condition_null - latents_mean) * latents_std)[0]

    latent = latent.to(device)
    ref_x = ref_x.to(device)
    ref_c = ref_c.to(device)
    conditions = conditions.to(device)
    condition_null = condition_null.to(device)
    cond={ "latent": latent, #torch.Size([1, 16, 7, 104, 60]) 
          "ref_x": ref_x, #torch.Size([36, 1, 104, 60])
          "ref_c":ref_c, #torch.Size([36, 1, 104, 60])
          "condition_pos": conditions, #torch.Size([16, 7, 104, 60])
          "condition_null": condition_null,} #torch.Size([16, 7, 104, 60])
    
    return cond


def inference(wan_i2v,cond,steps,frame_num,sample_shift,sample_guide_scale,condition_guide_scale,seed,max_area,offload_model):
    logging.info("Generating video ...")
    if isinstance(wan_i2v, WanI2VDancer):  
        video = wan_i2v.generate(
            None, #text or vl distll text
            cond["latent"], #rgb pil image
            img_x=cond["image_x"],  #rgb pil image
            img_c=cond["image_c"],#pose pil image index 0
            condition=cond["condition_pos"], # list of pos images
            condition_null=cond["condition_null"], # list of neg images
            max_area=max_area[0]*max_area[1],
            frame_num=frame_num,
            shift=sample_shift,
            sampling_steps=steps,
            guide_scale=sample_guide_scale,
            condition_guide_scale=condition_guide_scale,
            seed=seed,
            offload_model=offload_model,
            cond_input=cond,
            )
    else:
        video = wan_i2v(
            cond["latent"],
            num_frames=frame_num,
            num_inference_steps= steps,
            guidance_scale = sample_guide_scale,
            generator= torch.Generator(device="cpu").manual_seed( seed), #TODO
            prompt_embeds = cond["prompt_embeds"],
            negative_prompt_embeds= cond["negative_prompt_embeds"],
            image_embeds = cond["image_embeds"],
            image_embeds_c = cond["image_embeds_c"],
            condition_pos=cond["condition_pos"], # list of pos images
            condition_null=cond["condition_null"], # list of neg images
            condition_guide_scale=condition_guide_scale,
            sample_shift=sample_shift,
            ref_x=cond["ref_x"],
            ref_c=cond["ref_c"],
            seed=seed,

        )
    return video

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    elif "i2v" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")

        logging.info(f"Input cond_pos_folder: {args.cond_pos_folder}")
        logging.info(f"Input cond_neg_folder: {args.cond_neg_folder}")

        condition_pos_paths = [os.path.join(args.cond_pos_folder, "", f"{i:04d}.jpg") for i in range(args.frame_num)]
        condition_pos = list(
            [Image.open(f).convert("RGB").resize(img.size, Image.Resampling.BICUBIC) for f in condition_pos_paths])
        image_cond_pos = condition_pos_paths[0]

        condition_neg_paths = [os.path.join(args.cond_neg_folder, "", f"{i:04d}.jpg") for i in range(args.frame_num)]
        condition_neg = list(
            [Image.open(f).convert("RGB").resize(img.size, Image.Resampling.BICUBIC) for f in condition_neg_paths])
        # image_cond_neg = condition_neg_paths[0]

        logging.info(f"Input img_x: {args.image}")
        logging.info(f"Input img_c: {image_cond_pos}")

        img_x = Image.open(args.image).convert("RGB")
        img_c = Image.open(image_cond_pos).convert("RGB")
        img_c = img_c.resize(img.size, Image.Resampling.BICUBIC)

        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2VDancer(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            st_cond_cfg=args.st_cond_cfg, end_cond_cfg=args.end_cond_cfg,
        )

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            img_x=img_x,
            img_c=img_c,
            condition=condition_pos,
            condition_null=condition_neg,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            condition_guide_scale=args.condition_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    else:
        raise ValueError(f"Unkown task type: {args.task}")

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")


# if __name__ == "__main__":
#     args = _parse_args()
#     generate(args)
