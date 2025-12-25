 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from omegaconf import OmegaConf
from diffusers.hooks import apply_group_offloading
from .SteadyDancer.generate_dancer import load_models,pre_data_process,inference
import comfy.utils
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
import comfy.model_management as mm
from .SteadyDancer.wan import WanI2VDancer
from .SteadyDancer.wan.modules.vae import WanVAE
from .SteadyDancer.wan.pipeline_wan_i2v import I2V14B_VAE_CONFIG


MAX_SEED = np.iinfo(np.int32).max

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")

node_cr_path = os.path.dirname(os.path.abspath(__file__))

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir

class SteadyDancer_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):     
        return io.Schema(
            node_id="SteadyDancer_SM_Model",
            display_name="SteadyDancer_SM_Model",
            category="SteadyDancer_SM",
            inputs=[
                io.Combo.Input("dit",options= ["none"] +folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] +folder_paths.get_filename_list("gguf") ),   
            ],
            outputs=[
                io.Custom("SteadyDancer_SM_Model").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf) -> io.NodeOutput:
        origin_dict={}
        args=OmegaConf.create(origin_dict)
        args.repo=os.path.join(node_cr_path, "SteadyDancer/wan_repo")
        args.config_path=os.path.join(node_cr_path, "SteadyDancer/wan_repo/transformer")
        args.dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        args.ckpt_dir=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        args.gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        model=load_models(args)
        return io.NodeOutput(model)
    

class SteadyDancer_SM_VAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="SteadyDancer_SM_VAE",
            display_name="SteadyDancer_SM_VAE",
            category="SteadyDancer_SM",
            inputs=[
                io.Combo.Input("vae",options= ["none"] +folder_paths.get_filename_list("vae") ), 
                io.Latent.Input("latents",optional=True),     
            ],
            outputs=[
                io.Vae.Output(),
                io.Image.Output(),
                ],
            )
    @classmethod
    def execute(cls, vae, latents=None) -> io.NodeOutput:
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        model = WanVAE(vae_pth=vae_path,device=device)
        if latents is not None:
            import torchvision
            model_device = next(model.model.parameters()).device
            if model_device != device:
                model.model.to(device)
            tensor=model.decode( [i.to(model_device) for i in latents["samples"]])[0][None]
            try:
                print(tensor.shape)
            except:
                pass
            # preprocess
            value_range=(-1, 1)
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=8, normalize=True, value_range=value_range)
                for u in tensor.unbind(2)
            ],dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()
            try:
                print(tensor.shape)
            except:
                pass
            model.model.to("cpu")
        else:
            tensor=None
        return io.NodeOutput(model,tensor)
    
class SteadyDancer_SM_Cond(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SteadyDancer_SM_Cond",
            display_name="SteadyDancer_SM_Cond",
            category="SteadyDancer_SM",
            inputs=[
                io.Vae.Input("vae"),
                io.ClipVision.Input("clip_vision"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Image.Input("image",),
                io.Image.Input("pose_video",),
                io.Image.Input("pose_neg_lvideo",),
                io.Combo.Input("wh_size",options= ['720*1280', '1280*720', '480*832', '832*480', '576*1024', '1024*576', '1024*800']),  
                io.Int.Input("num_frames", default=81, min=25, max=10240,step=4,display_mode=io.NumberDisplay.number),
                ],
            outputs=[
                io.Conditioning.Output("cond"),
                     ],
        )

    @classmethod
    def execute(cls, vae,clip_vision,positive,negative,image,pose_video,pose_neg_lvideo,wh_size,num_frames) -> io.NodeOutput: 

        width,height = map(int, wh_size.split('*'))
        if image.shape[1]!=height or image.shape[2]!=width:
            image=comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if pose_video.shape[1]!=height or pose_video.shape[2]!=width:
            pose_video=comfy.utils.common_upscale(pose_video.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if pose_neg_lvideo.shape[1]!=height or pose_neg_lvideo.shape[2]!=width:
            pose_neg_lvideo=comfy.utils.common_upscale(pose_neg_lvideo.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if pose_video.shape[0]!=num_frames:
            pose_video=pose_video[:num_frames]
        if pose_neg_lvideo.shape[0]!=num_frames:
            pose_neg_lvideo=pose_neg_lvideo[:num_frames]
        num_frames=min(num_frames,pose_video.shape[0],pose_neg_lvideo.shape[0])
        print(f"########## Final num_frames is : {num_frames} #########")
        # pre-process
        image_embeds=clip_vision.encode_image(image)["penultimate_hidden_states"]
        image_embeds_c=clip_vision.encode_image(pose_video[:1,:,:,:3])["penultimate_hidden_states"]
        if isinstance(vae,WanVAE):
            # Offload model to CPU before custom vae encoding
            cf_models=mm.loaded_models()
            for model in cf_models:   
                model.unpatch_model(device_to=torch.device("cpu"))
            mm.soft_empty_cache()      
            torch.cuda.empty_cache()
        cond=pre_data_process(vae,image,pose_video,pose_neg_lvideo,num_frames,width,height,device)
        cond["image_embeds"]=image_embeds
        cond["image_embeds_c"]=image_embeds_c
        cond["prompt_embeds"]=positive[0][0]
        cond["negative_prompt_embeds"]=negative[0][0]
        cond["size"]=(height,width)
        cond["num_frames"]=num_frames

        # Offload model to CPU after encoding
        cf_models=mm.loaded_models()
        for model in cf_models:   
            model.unpatch_model(device_to=torch.device("cpu"))
        mm.soft_empty_cache()      
        torch.cuda.empty_cache()

        return io.NodeOutput (cond)


class SteadyDancer_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SteadyDancer_SM_KSampler",
            display_name="SteadyDancer_SM_KSampler",
            category="SteadyDancer_SM",
            inputs=[
                io.Custom("SteadyDancer_SM_Model").Input("model"),
                io.Conditioning.Input("cond"),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras") ),
                io.Int.Input("steps", default=1, min=1, max=10000,display_mode=io.NumberDisplay.number),
                io.Float.Input("sample_shift", default=5.0, min=0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("sample_guide_scale", default=1.0, min=0.0, max=100.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("condition_guide_scale", default=1.5, min=0.0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Boolean.Input("offload_model", default=True),
                io.Int.Input("block_num", default=1, min=1, max=100,display_mode=io.NumberDisplay.number),
                ],
            outputs=[
                io.Latent.Output(display_name="latents"),
            ],
        ) 
    @classmethod
    def execute(cls, model,cond,lora,steps,sample_shift,sample_guide_scale,condition_guide_scale,seed,offload_model,block_num ) -> io.NodeOutput: 
        lora_path=folder_paths.get_full_path("loras", lora) if lora != "none" else None
        if lora_path is not None and not isinstance(model, WanI2VDancer):
            model.load_lora_weights_(lora_path)
            model.set_lora_adapter(adapter_name="default", weight=1.0)
        max_area=cond["size"]
        # Upscale
        if max_area[0]==832 or  max_area[0]==480:
            sample_shift = 3.0
        frame_num=cond["num_frames"]
        if not isinstance(model, WanI2VDancer):
            apply_group_offloading(model.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=block_num)
        videos=inference( model,cond,steps,frame_num,sample_shift,sample_guide_scale,condition_guide_scale,seed,max_area,offload_model)
        videos=videos.unsqueeze(0)
        #print(f"########## Final video shape is : {videos.shape} #########")
        latents_mean = (
        torch.tensor(I2V14B_VAE_CONFIG["latents_mean"]).view(1, 16, 1, 1, 1).to(videos.device, videos.dtype))
        latents_std = 1.0 / torch.tensor(I2V14B_VAE_CONFIG['latents_std']).view(1, 16, 1, 1, 1).to(videos.device, videos.dtype)
        videos = videos / latents_std + latents_mean           
        output={}
        output["samples"]=videos
        return io.NodeOutput(output)


from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/SteadyDancer_SM_Extension")
async def get_hello(request):
    return web.json_response("SteadyDancer_SM_Extension")

class SteadyDancer_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SteadyDancer_SM_Model,
            SteadyDancer_SM_VAE,
            SteadyDancer_SM_Cond,
            SteadyDancer_SM_KSampler,
        ]


async def comfy_entrypoint() -> SteadyDancer_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return SteadyDancer_SM_Extension()
