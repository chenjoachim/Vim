# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from rope import *
import random

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


__all__ = [
    "vim_tiny_patch16_224",
    "vim_small_patch16_224",
    "vim_base_patch16_224",
    "vim_tiny_patch16_384",
    "vim_small_patch16_384",
    "vim_base_patch16_384",
]


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), (
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual.clone()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(
        Mamba,
        d_state=d_state,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_divide_out=if_divide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class MaskPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.predictor = nn.Linear(dim, 2)

    def forward(self, x, mask=None):
        # Return raw logits — caller is responsible for softmax or Gumbel-softmax.
        # Gumbel-softmax (training) and softmax+topK (inference) both require logits.
        return self.predictor(x)


class VisionMamba(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=192,
        d_state=16,
        channels=3,
        num_classes=1000,
        ssm_cfg=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        ft_seq_len=None,
        pt_hw_seq_len=14,
        if_bidirectional=False,
        final_pool_type="none",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        flip_img_sequences_ratio=-1.0,
        if_bimamba=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        init_layer_scale=None,
        use_double_cls_token=False,
        use_middle_cls_token=True,
        token_ratio=0.7,
        enable_dyvm=False,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # DyVM
        self.enable_dyvm = enable_dyvm
        self.token_ratio = token_ratio
        if enable_dyvm:
            self.bin_masks = MaskPredictor(embed_dim)

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_tokens, self.embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim, pt_seq_len=pt_hw_seq_len, ft_seq_len=hw_seq_len
            )
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # TODO: release this comment
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=0.02)
                trunc_normal_(self.cls_token_tail, std=0.02)
            else:
                trunc_normal_(self.cls_token, std=0.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def reorder(self, x, token_position, mask=None):
        """Rearrange tokens per DyVM Eq. 15-17.

        Args:
            x: [B, L, D] token features after mask multiplication
            token_position: int, cls token index in the original sequence
            mask: [B, L] binary mask (1=retained, 0=pruned), detached.
                  If None, falls back to norm>0 heuristic (fragile — avoid).

        Returns:
            rearranged_tokens: [B, L, D]
            cls_positions: list[int] of length B — cls index per batch item
            orig_indices: None (placeholder)
        """
        B, L, D = x.shape
        results = []
        cls_positions = []  # per-batch: different batch items may retain different counts

        for b in range(B):
            if mask is not None:
                # Use explicit binary mask; exclude cls from the patch-token set
                keep_bool = mask[b].bool().clone()
                keep_bool[token_position] = False
                keep_idx = torch.where(keep_bool)[0]
            else:
                # Fallback: detect retained tokens by non-zero, finite norm
                norms = x[b].norm(dim=1)
                keep_bool = torch.isfinite(norms) & (norms > 0)
                keep_bool[token_position] = False
                keep_idx = torch.where(keep_bool)[0]

            keep_tokens = x[b, keep_idx, :]
            cls_token = x[b, token_position, :].unsqueeze(0)

            n_kept = len(keep_idx)
            n_zeros = L - n_kept - 1
            zeros = torch.zeros(n_zeros, D, dtype=x.dtype, device=x.device)

            half = n_kept // 2
            cls_positions.append(half)  # cls lands at this index in the rearranged row

            rearranged = torch.cat(
                [
                    keep_tokens[:half],
                    cls_token,
                    keep_tokens[half:],
                    zeros,
                ],
                dim=0,
            )
            results.append(rearranged)

        rearranged_tokens = torch.stack(results, dim=0)
        print(
            f"DEBUG: reorder cls_positions min={min(cls_positions)}, max={max(cls_positions)}, "
            f"unique_counts={len(set(cls_positions))}"
        )
        return rearranged_tokens, cls_positions, None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed",
            "cls_token",
            "dist_token",
            "cls_token_head",
            "cls_token_tail",
        }

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(
        self,
        x,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
        return_aux=False,
        skip_dyvm=False,
        return_all_tokens=False,
    ):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token

        # Runs cheaply; fires the first time a weight goes NaN so we know exactly
        # which parameter was corrupted and at what step.
        _nan_params = [
            name for name, p in self.named_parameters() if not torch.isfinite(p).all()
        ]
        if _nan_params:
            print(
                f"DEBUG PARAM NaN: {_nan_params[:5]}"
                f"{'...' if len(_nan_params) > 5 else ''} "
                f"(total {len(_nan_params)} NaN params)"
            )
        # ────────────────────────────────────────────────────────────────────

        x = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat(
                        (x[:, :token_position, :], cls_token, x[:, token_position:, :]),
                        dim=1,
                    )
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat(
                        (x[:, :token_position, :], cls_token, x[:, token_position:, :]),
                        dim=1,
                    )
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(
                        B, -1, -1
                    )  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]
                print(f"**shape after CLS: {x.shape}")

        # Initialise to None; set inside the masking block below if DyVM runs.
        dyvm_masks = None
        orig_indices = None
        # Post-reorder binary mask [B, L]: 1 for retained+cls positions, 0 for pruned zeros.
        # Applied to hidden_states after Mamba to stop gradient flowing through zero-padded positions.
        post_reorder_mask = None

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

            if self.enable_dyvm and not skip_dyvm:
                print("DEBUG: Inside pruning block")
                B_m, L_m, _ = x.shape

                # Early NaN detection — if x is already NaN the predictor output will be NaN
                # x_has_nan = torch.isnan(x).any().item()
                # print(f"DEBUG: x has NaN before bin_masks: {x_has_nan}")
                # if x_has_nan:
                # print(f"DEBUG: NaN fraction in x: {torch.isnan(x).float().mean().item():.4f}")

                raw_logits = self.bin_masks(x)  # [B, L, 2] — raw logits (no softmax)
                # logits_nan = torch.isnan(raw_logits).any().item()
                # clamp prevents overflow → inf → NaN in gumbel/softmax
                # nan_to_num handles NaN that leaked from collapsed weights (clamp alone can't fix NaN)
                raw_logits = torch.nan_to_num(raw_logits, nan=0.0).clamp(-10.0, 10.0)
                # print(f"DEBUG: raw_logits shape: {raw_logits.shape}, "
                # f"had_nan_before_fix={logits_nan}, "
                # f"min={raw_logits.min().item():.3f}, max={raw_logits.max().item():.3f}")
                # print(f"DEBUG: Training mode: {self.training}")

                # during training
                if self.training:
                    # print("DEBUG: Using Gumbel-Softmax (training)")
                    # Paper Eq. 12: Π = Softmax(P(H, M_{s-1})) — apply softmax first
                    pi = F.softmax(raw_logits, dim=-1)  # [B, L, 2] — Π per Eq. 12
                    # Paper Eq. 13: M_hat = Gumbel-Softmax(Π)
                    # We pass the log-probabilities as logits to gumbel_softmax
                    # (gumbel_softmax internally adds Gumbel noise and re-softmaxes)
                    probs = F.gumbel_softmax(
                        pi.log().clamp(-20, 0), tau=1.0, hard=True, dim=-1
                    )
                    # print(f"DEBUG: probs has_nan={torch.isnan(probs).any().item()}")

                    retain_prob = probs[
                        :, :, 0
                    ]  # [B, L] — {0,1} forward, differentiable backward

                    # Force cls token to be retained — done WITHOUT in-place modification
                    # (in-place on a non-leaf tensor corrupts autograd)
                    cls_keep = torch.zeros(
                        B_m, L_m, device=x.device, dtype=retain_prob.dtype
                    )
                    cls_keep[:, token_position] = 1.0
                    masks = (
                        retain_prob * (1.0 - cls_keep) + cls_keep
                    )  # cls position → 1, others unchanged

                    # Hard binary mask (detached) for the reorder structural decision
                    masks_hard = masks.detach().round()

                    x = masks.unsqueeze(-1) * x
                    n_kept_per_batch = masks_hard.sum(
                        dim=1
                    ).long()  # [B] — retained + cls per sample
                    # print(f"DEBUG: Masks shape: {masks.shape}, "
                    # f"num kept: {n_kept_per_batch.float().mean().item():.1f}, "
                    # f"has_nan: {torch.isnan(masks).any().item()}")
                    dyvm_masks = masks
                    x, token_position, orig_indices = self.reorder(
                        x, token_position, mask=masks_hard
                    )
                    # token_position is now a list[int] of per-batch cls positions
                    # print(f"DEBUG: After reorder - x shape: {x.shape}, token_position (list): {token_position[:4]}...")

                    # Build post-reorder mask: 1 for retained positions (0..n_kept-1), 0 for pruned zeros.
                    # This is applied to hidden_states after Mamba to stop gradient from flowing
                    # backward through the zero-padded pruned positions (which cause NaN gradients
                    # via the bidirectional SSM backward scan accumulating A^2304).
                    post_reorder_mask = torch.zeros(
                        B_m, L_m, device=x.device, dtype=x.dtype
                    )
                    for _b in range(B_m):
                        post_reorder_mask[_b, : n_kept_per_batch[_b]] = 1.0
                    # print(f"DEBUG: post_reorder_mask mean retained={post_reorder_mask.sum(dim=1).mean().item():.1f}")

                # inference path
                else:
                    # print("DEBUG: Using top-K (inference)")
                    retain_probs = F.softmax(raw_logits, dim=-1)[
                        :, :, 0
                    ]  # [B, L] Π[:,i,0]
                    K = max(
                        1, int(self.token_ratio * (L_m - 1))
                    )  # non-cls tokens to keep
                    # print(f"DEBUG: retain_probs min={retain_probs.min().item():.3f}, "
                    # f"max={retain_probs.max().item():.3f}, "
                    # f"mean={retain_probs.mean().item():.3f}, "
                    # f"has_nan={torch.isnan(retain_probs).any().item()}")
                    # print(f"DEBUG: retain_probs[0, :10]={retain_probs[0, :10]}")
                    # print(f"DEBUG: retain_probs[0, token_position]={retain_probs[0, token_position].item():.3f}")
                    # print(f"DEBUG: K={K}, token_ratio={self.token_ratio}")
                    # Force cls token to always be in top-K by inflating its score.
                    retain_probs_sorted = retain_probs.clone()
                    retain_probs_sorted[:, token_position] = float("inf")
                    topk_indices = retain_probs_sorted.topk(
                        K + 1, dim=1
                    ).indices  # K + cls

                    # print(f"DEBUG: topk_indices[0]={topk_indices[0][:10]}... (first 10)")
                    # print(f"DEBUG: Is token_position in topk? {token_position in topk_indices[0].tolist()}")

                    masks = torch.zeros(B_m, L_m, device=x.device, dtype=x.dtype)
                    masks.scatter_(1, topk_indices, 1.0)
                    # print(f"DEBUG: Masks shape: {masks.shape}, num kept: {masks.sum(dim=1).mean().item():.1f}")
                    x = masks.unsqueeze(-1) * x
                    dyvm_masks = masks
                    x, token_position, orig_indices = self.reorder(
                        x, token_position, mask=masks
                    )
                    # All batch items have the same K in inference → cls_positions are all K//2
                    # print(f"DEBUG: After reorder - x shape: {x.shape}, token_position (list): {token_position[:4]}...")
                    token_position = (
                        K // 2
                    )  # scalar: consistent across batch in inference

        if if_random_token_rank:
            # DyVM reorder produces per-batch cls positions (list of length B).
            # Random token rank shuffle is incompatible with per-batch positions —
            # skip it when DyVM has already rearranged the sequence.
            if isinstance(token_position, list):
                print(
                    "DEBUG: skipping if_random_token_rank, DyVM reorder already active"
                )
            else:
                # 生成随机 shuffle 索引
                shuffle_indices = torch.randperm(M)
                print("original value: ", x[0, token_position, 0])
                print("original token_position: ", token_position)

                # 执行 shuffle
                x = x[:, shuffle_indices, :]

                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[
                    0
                ].item()
                print("new value: ", x[0, token_position, 0])
                print("new token_position: ", token_position)

        if_flip_img_sequences = False
        if (
            self.flip_img_sequences_ratio > 0
            and (self.flip_img_sequences_ratio - random.random()) > 1e-5
        ):
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1]),
                    inference_params=inference_params,
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

                print(f"**hidden state shape: {hidden_states.shape}")
                print(f"**residual shape: {residual.shape}")

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # ── Post-reorder output mask ─────────────────────────────────────────
        if post_reorder_mask is not None:
            hidden_states = hidden_states * post_reorder_mask.unsqueeze(-1)
            # print(f"DEBUG: applied post_reorder_mask, hidden_states has_nan="f"{torch.isnan(hidden_states).any().item()}")
        # ─────────────────────────────────────────────────────────────────────

        # ---- Save all token features BEFORE extraction (needed for Ldis_token) ----
        all_token_hidden = hidden_states  # [B, L, D]

        # Allow teacher to retrieve full token tensor without cls extraction.
        if return_all_tokens:
            return all_token_hidden

        # ---- Extract cls token or apply pooling ----
        if self.if_cls_token:
            if self.use_double_cls_token:
                x_out = (
                    hidden_states[:, token_position[0], :]
                    + hidden_states[:, token_position[1], :]
                ) / 2
            else:
                if isinstance(token_position, list):
                    # Per-batch cls positions from DyVM training reorder.
                    # Different batch items may have retained different numbers of tokens
                    # so their cls lands at different indices — must index per-sample.
                    pos_t = torch.tensor(
                        token_position, device=hidden_states.device, dtype=torch.long
                    )
                    x_out = hidden_states[
                        torch.arange(B, device=hidden_states.device), pos_t, :
                    ]
                    print(
                        f"DEBUG: cls extraction — per-batch positions {token_position[:4]}..."
                    )
                else:
                    x_out = hidden_states[:, token_position, :]
        elif self.final_pool_type == "none":
            x_out = hidden_states[:, -1, :]
        elif self.final_pool_type == "mean":
            x_out = hidden_states.mean(dim=1)
        elif self.final_pool_type in ("max", "all"):
            x_out = hidden_states  # handled by forward()
        else:
            raise NotImplementedError

        if return_aux:
            return (
                x_out,
                {
                    "masks": [dyvm_masks],  # list of [B, L] per pruning stage
                    "token_feats": all_token_hidden,  # [B, L, D] post-Mamba
                    "orig_indices": orig_indices,  # [B, L] original patch positions (-1=padding)
                },
            )
        return x_out

    @torch.no_grad()
    def forward_all_tokens(self, x):
        """Return all token hidden states [B, L, D] without DyVM pruning.
        Used by the teacher model to produce aligned token features for Ldis_token.
        """
        return self.forward_features(x, skip_dyvm=True, return_all_tokens=True)

    def forward(
        self,
        x,
        return_features=False,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
        return_aux=False,
    ):
        result = self.forward_features(
            x,
            inference_params,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank,
            return_aux=return_aux,
        )

        if return_aux:
            x, aux = result
        else:
            x = result

        if return_features:
            return (x, aux) if return_aux else x

        x = self.head(x)
        if self.final_pool_type == "max":
            x = x.max(dim=1)[0]

        if return_aux:
            return x, aux
        return x


@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        stride=8,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        stride=8,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        embed_dim=768,
        d_state=16,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
