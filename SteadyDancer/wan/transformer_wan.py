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

import math
from typing import Any, Dict, Optional, Tuple, Union
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from .modules.attention import flash_attention
from .modules.small_archs import FactorConv3d, PoseRefNetNoBNV3
from .modules.mobilenetv2_dcd import DYModule
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

# @amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    # return torch.stack(output).float()
    return torch.stack(output)

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)

class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.to(torch.bfloat16)
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # Check if modulation dimension matches e dimension
        modulation_dim = self.modulation.shape[2]
        e_dim = e.shape[2]
        #print(f"modulation_dim: {self.modulation.shape}, e_dim: {e.shape},attn")
        if modulation_dim != e_dim:
            # Adjust modulation to match e dimension
            if modulation_dim < e_dim:
                # Pad modulation with zeros
                pad_size = e_dim - modulation_dim
                modulation = torch.nn.functional.pad(self.modulation, (0, pad_size, 0, 0, 0, 0))
                #print(f"Adjusted modulation dimension from {modulation_dim} to {e_dim} by padding")
            else:
                # Truncate modulation
                modulation = self.modulation[:, :, :e_dim]
                #print(f"Adjusted modulation dimension from {modulation_dim} to {e_dim} by truncating")
        else:
            modulation = self.modulation
        
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        with amp.autocast("cuda",dtype=torch.bfloat16, ):
            e = (modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        # with amp.autocast(dtype=x.dtype):
        with amp.autocast("cuda",dtype=torch.bfloat16,):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=x.dtype):
            with amp.autocast("cuda",dtype=torch.bfloat16,):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # Check if modulation dimension matches e dimension
        modulation_dim = self.modulation.shape[2]
        e_dim = e.shape[1]
        #print(f"modulation_dim: {self.modulation.shape}, e_dim: {e.shape}")
        #modulation_dim: torch.Size([1, 6, 5440]), e_dim: torch.Size([1, 6, 5120])
        if modulation_dim != e_dim:
            # Adjust modulation to match e dimension
            if modulation_dim < e_dim:
                # Pad modulation with zeros
                pad_size = e_dim - modulation_dim
                modulation = torch.nn.functional.pad(self.modulation, (0, pad_size, 0, 0, 0, 0))
                #print(f"Head: Adjusted modulation dimension from {modulation_dim} to {e_dim} by padding")
            else:
                # Truncate modulation
                modulation = self.modulation[:, :, :e_dim]
                #print(f"Head: Adjusted modulation dimension from {modulation_dim} to {e_dim} by truncating")
        else:
            modulation = self.modulation
        
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x

class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanTransformer3DModelSD(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """
    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    # _supports_gradient_checkpointing = True
    # _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    # _no_split_modules = ["WanTransformerBlock"]
    # _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    # _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    # _repeated_blocks = ["WanTransformerBlock"]
    # _cp_plan = {
    #     "rope": {
    #         0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
    #         1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
    #     },
    #     "blocks.0": {
    #         "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
    #     },
    #     "blocks.*": {
    #         "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
    #     },
    #     "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    #     "": {
    #         "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
    #     },
    # }

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        in_dim_c=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()


        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.in_dim_c = in_dim_c
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        ############### Condition-Reconciliation Mechanism ###############
        self.patch_embedding = nn.Conv3d(           # ref_x
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_fuse = nn.Conv3d(      # x, fused pose, aligned pose
            in_dim + self.in_dim_c + self.in_dim_c, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_ref_c = nn.Conv3d(    # ref_c
            self.in_dim_c, dim, kernel_size=patch_size, stride=patch_size)

        ############### Synergistic Pose Modulation Modules ###############
        # Spatial Structure Adaptive Extractor
        self.condition_embedding_spatial = DYModule(inp=self.in_dim_c, oup=self.in_dim_c)
        # Temporal Motion Coherence Module
        self.condition_embedding_temporal = nn.Sequential(
            FactorConv3d(in_channels=self.in_dim_c, out_channels=self.in_dim_c, kernel_size=(3, 3, 3), stride=1),
            nn.SiLU(),
            FactorConv3d(in_channels=self.in_dim_c, out_channels=self.in_dim_c, kernel_size=(3, 3, 3), stride=1),
            nn.SiLU(),
            FactorConv3d(in_channels=self.in_dim_c, out_channels=self.in_dim_c, kernel_size=(3, 3, 3), stride=1),
            nn.SiLU()
        )
        # Frame-wise Attention Alignment Unit
        self.condition_embedding_align = PoseRefNetNoBNV3(in_channels_x=16, #
                                                in_channels_c=16,
                                                hidden_dim=128,
                                                num_heads=8)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)


        self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        condition=None,
        ref_x=None,
        ref_c=None,
        clip_fea_x=None,
        clip_fea_c=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        assert clip_fea_x is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x_noise_clone = torch.stack(x)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Temporal Motion Coherence Module.
        condition_temporal = [self.condition_embedding_temporal(c.unsqueeze(0)) for c in [condition]]
        
        # Spatial Structure Adaptive Extractor.
        with amp.autocast("cuda",dtype=torch.bfloat16, ):
            condition = condition[None]
            bs, _, time_steps, _, _ = condition.shape
            condition_reshape = rearrange(condition, 'b c t h w -> (b t) c h w')
            condition_spatial = self.condition_embedding_spatial(condition_reshape)
            condition_spatial = rearrange(condition_spatial, '(b t) c h w -> b c t h w', t=time_steps, b=bs)

        # Hierarchical Aggregation (1): condition, temporal condition, spatial condition
        condition_fused = condition + condition_temporal[0] + condition_spatial

        # Frame-wise Attention Alignment Unit.
        with amp.autocast("cuda",dtype=torch.bfloat16,):
            condition_aligned = self.condition_embedding_align(condition_fused, x_noise_clone)

        real_seq = x[0].shape[1]
        
        # Condition Fusion/Injection, Hierarchical Aggregation (2): x, fused condition, aligned condition
        x = [self.patch_embedding_fuse(torch.cat([u[None], c[None], a[None]], 1)) for u, c, a in
             zip(x, condition_fused, condition_aligned)]
        
        # Condition Augmentation: x_cond, ref_x, ref_c
        ref_x = [ref_x]
        ref_c = [ref_c]
        ref_x = [self.patch_embedding(r.unsqueeze(0)) for r in ref_x]
        ref_c = [self.patch_embedding_ref_c(r[:16].unsqueeze(0)) for r in ref_c]
        x = [torch.cat([r, u, v], dim=2) for r, u, v in zip(x, ref_x, ref_c)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast("cuda",dtype=torch.bfloat16,):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea_x is not None:
            context_clip_x = self.img_emb(clip_fea_x)  # bs x 257 x dim
        if clip_fea_c is not None:
            context_clip_c = self.img_emb(clip_fea_c)  # bs x 257 x dim
        if clip_fea_x is not None:
            context_clip = context_clip_x if context_clip_c is None else context_clip_x + context_clip_c    # Condition Augmentation
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        # torch.Size([1])
        # torch.Size([1, 3])
        # torch.Size([1, 769, 5120])
        # torch.Size([1024, 64])
        # print(seq_lens.shape)
        # print(grid_sizes.shape)
        # print(context.shape if context is not None else "None")
        # print(self.freqs.shape)
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        # return [u.float() for u in x]
        return [u[:, :real_seq, ...] for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
    
        #return Transformer2DModelOutput(sample=output)
    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
