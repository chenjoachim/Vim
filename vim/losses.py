# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss and DyVM joint training loss.
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class DyVMLoss(torch.nn.Module):
    """
    DyVM joint training loss (Equation 27 in the paper):

        Ljoint = λ_cls * Lcls
               + λ_token * Ltoken          (Eq. 23 — token pruning ratio supervision)
               + λ_block * Lblock          (Eq. 24 — block selection ratio; zero until blocks added)
               + λ_dis_out * Ldis_out      (Eq. 25 — KL divergence with teacher logits)
               + λ_dis_token * Ldis_token  (Eq. 26 — MSE on retained token features)

    Usage:
        model output must be (logits, aux_dict) when return_aux=True.
        aux_dict keys:
            'masks'       — list of [B, L] binary masks, one per pruning stage
            'token_feats' — [B, L, D] post-Mamba token hidden states (student, reordered)
            'orig_indices'— [B, L] mapping reordered position → original patch position
                            (-1 for zero-padded positions)

    Notes:
        Lblock is currently always 0 because block-level selection is not yet implemented
        in the model.  Set lambda_block=0 (default) to skip it entirely.

        Ldis_token requires the teacher to be a VisionMamba instance (it calls
        teacher.forward_all_tokens).  If the teacher is any other architecture,
        Ldis_token is silently skipped.
    """

    def __init__(
        self,
        base_criterion: torch.nn.Module,
        teacher_model: torch.nn.Module,
        token_ratio: float = 0.7,
        block_ratio: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_token: float = 2.0,
        lambda_block: float = 0.0,
        lambda_dis_out: float = 0.5,
        lambda_dis_token: float = 0.5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.token_ratio = token_ratio
        self.block_ratio = block_ratio
        self.lambda_cls = lambda_cls
        self.lambda_token = lambda_token
        self.lambda_block = lambda_block
        self.lambda_dis_out = lambda_dis_out
        self.lambda_dis_token = lambda_dis_token
        self.temperature = temperature

    def forward(self, inputs, outputs, labels):
        """
        inputs  : [B, C, H, W]  — original images (fed to teacher for distillation)
        outputs : (logits, aux_dict)  when model called with return_aux=True,
                  or just logits Tensor (eval / non-DyVM path)
        labels  : [B] or soft targets
        """
        # ── Unpack outputs ──────────────────────────────────────────────────
        if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
            logits, aux = outputs
        else:
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            aux = None

        total_loss = torch.tensor(0.0, device=logits.device)

        # ── Lcls  (Eq. 22) ─────────────────────────────────────────────────
        loss_cls = self.base_criterion(logits, labels)
        total_loss = total_loss + self.lambda_cls * loss_cls

        # ── Ltoken  (Eq. 23) ───────────────────────────────────────────────
        # For each pruning stage s, penalise deviation from target keep ratio ρ_s.
        # With a single stage S=1 and a uniform ratio ρ:
        #   Ltoken = (1/BS) Σ_b Σ_s (ρ_s − mean_i M_{b,s,i})²
        if aux is not None and self.lambda_token > 0:
            masks_list = [m for m in aux.get("masks", []) if m is not None]
            if masks_list:
                loss_token = torch.tensor(0.0, device=logits.device)
                for mask_s in masks_list:                       # mask_s: [B, L]
                    actual_ratio = mask_s.float().mean(dim=1)   # [B]
                    target = torch.full_like(actual_ratio, self.token_ratio)
                    loss_token = loss_token + F.mse_loss(actual_ratio, target)
                loss_token = loss_token / len(masks_list)
                total_loss = total_loss + self.lambda_token * loss_token

        # ── Lblock (Eq. 24) ────────────────────────────────────────────────
        # Requires block-gate tensors aux['block_masks'] = list of [B, N, 2] Q values.
        # Not yet implemented in the model → always 0 unless you add block selection.
        if aux is not None and self.lambda_block > 0:
            block_masks = aux.get("block_masks")   # list of [B, 2] per layer
            if block_masks is not None:
                N = len(block_masks)
                # mean over batch and layers of (Q0 + Q1) / 2
                avg_active = sum(
                    (q[:, 0] + q[:, 1]) / 2.0 for q in block_masks
                ) / N                               # [B]
                avg_active = avg_active.mean()
                loss_block = (self.block_ratio - avg_active) ** 2
                total_loss = total_loss + self.lambda_block * loss_block

        # ── Teacher distillation ────────────────────────────────────────────
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_out = self.teacher_model(inputs)
                teacher_logits = teacher_out if isinstance(teacher_out, torch.Tensor) \
                                 else teacher_out[0]

            # Ldis_out  (Eq. 25) — KL divergence on output distributions
            if self.lambda_dis_out > 0:
                T = self.temperature
                loss_dis_out = F.kl_div(
                    F.log_softmax(logits / T, dim=1),
                    F.log_softmax(teacher_logits / T, dim=1),
                    reduction="batchmean",
                    log_target=True,
                ) * (T * T)
                total_loss = total_loss + self.lambda_dis_out * loss_dis_out

            # Ldis_token  (Eq. 26) — MSE on retained tokens vs teacher tokens
            # Only possible when teacher is a VisionMamba (exposes forward_all_tokens).
            if (
                self.lambda_dis_token > 0
                and aux is not None
                and aux.get("token_feats") is not None
                and aux.get("orig_indices") is not None
                and hasattr(self.teacher_model, "forward_all_tokens")
            ):
                with torch.no_grad():
                    # Teacher: full [B, L, D] token features, original patch order, no pruning
                    teacher_tokens = self.teacher_model.forward_all_tokens(inputs)

                student_tokens = aux["token_feats"]   # [B, L, D]  — reordered
                orig_indices   = aux["orig_indices"]  # [B, L]  (-1 = padding)

                B, L, _ = student_tokens.shape
                valid_mask = orig_indices >= 0        # [B, L]

                # Gather teacher features at each student token's original position.
                # Clamp -1 → 0 for safe indexing; the valid_mask zeros those out.
                safe_idx = orig_indices.clamp(min=0)  # [B, L]
                batch_idx = (
                    torch.arange(B, device=student_tokens.device)
                    .unsqueeze(1)
                    .expand(B, L)
                )
                gathered_teacher = teacher_tokens[batch_idx, safe_idx, :]  # [B, L, D]

                # Weighted MSE: only count valid (non-padding) positions
                diff_sq  = (student_tokens - gathered_teacher).pow(2).sum(dim=-1)  # [B, L]
                n_valid  = valid_mask.float().sum().clamp(min=1.0)
                loss_dis_token = (valid_mask.float() * diff_sq).sum() / n_valid
                total_loss = total_loss + self.lambda_dis_token * loss_dis_token

        return total_loss
