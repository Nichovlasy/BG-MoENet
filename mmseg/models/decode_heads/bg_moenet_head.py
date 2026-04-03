from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


class _CMBContextBranches(nn.ModuleList):
    """Context branches used in CMB.

    Y = {Y_pool, Y_loc, Y_0, ..., Y_{D-1}}

    Notes:
        - dilation == 1 branch is implemented as 1x1 conv
        - local DW 3x3 branch is explicitly separated as Y_loc
    """

    def __init__(self,
                 dilations: Tuple[int, ...],
                 in_channels: int,
                 out_channels: int,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.dilations = tuple(dilations)

        for dilation in self.dilations:
            if dilation == 1:
                self.append(
                    ConvModule(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                self.append(
                    DepthwiseSeparableConvModule(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        dilation=dilation,
                        padding=dilation,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

    def forward(self, x: torch.Tensor):
        return [branch(x) for branch in self]


class _CrossLevelConditionalGate(nn.Module):
    """Cross-level conditional gate in CMB.

    alpha = Softmax(Gamma([GAP(psi4(E4)), GAP(psi3(E3))]) / tau_c)
    Y_gate = phi_g(sum(alpha_m * Y_m))
    """

    def __init__(self,
                 c4: int,
                 c3: int,
                 out_channels: int,
                 num_branches: int,
                 temperature: float = 1.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.temperature = float(temperature)

        self.deep_projection = nn.Sequential(
            ConvModule(
                c4,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.AdaptiveAvgPool2d(1),
        )
        self.skip_projection = nn.Sequential(
            ConvModule(
                c3,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.AdaptiveAvgPool2d(1),
        )

        hidden_channels = max(64, out_channels // 2)
        self.gating_network = nn.Sequential(
            ConvModule(
                2 * out_channels,
                hidden_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.Conv2d(hidden_channels, num_branches, kernel_size=1),
        )

        self.gated_fusion_refine = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self,
                branch_responses: List[torch.Tensor],
                deep_feature: torch.Tensor,
                skip_feature: torch.Tensor) -> torch.Tensor:
        assert len(branch_responses) > 0

        n, _, _, _ = branch_responses[0].shape
        conditional_descriptor = torch.cat(
            [self.deep_projection(deep_feature), self.skip_projection(skip_feature)],
            dim=1)

        branch_logits = self.gating_network(conditional_descriptor).flatten(1)
        branch_weights = torch.softmax(
            branch_logits / self.temperature, dim=1).view(n, -1, 1, 1, 1)

        stacked_branches = torch.stack(branch_responses, dim=1)
        gated_sum = (branch_weights * stacked_branches).sum(dim=1)
        return self.gated_fusion_refine(gated_sum)


class ContextualMultiBranchBridge(nn.Module):
    """CMB: Contextual Multi-branch Bridge."""

    def __init__(self,
                 c4: int,
                 c3: int,
                 out_channels: int,
                 dilations: Tuple[int, ...] = (1, 6, 12, 18),
                 temperature: float = 3.5,
                 mix_coefficient: float = 0.6,
                 enable_cross_level_gating: bool = True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.enable_cross_level_gating = bool(enable_cross_level_gating)
        self.mix_coefficient = float(mix_coefficient)

        self.global_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                c4,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

        self.local_branch = DepthwiseSeparableConvModule(
            c4,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.context_branches = _CMBContextBranches(
            dilations=dilations,
            in_channels=c4,
            out_channels=out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.num_branches = 2 + len(dilations)

        self.concat_fusion_refine = ConvModule(
            self.num_branches * out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.residual_projection = ConvModule(
            c4,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if self.enable_cross_level_gating:
            self.cross_level_gate = _CrossLevelConditionalGate(
                c4=c4,
                c3=c3,
                out_channels=out_channels,
                num_branches=self.num_branches,
                temperature=temperature,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, e4: torch.Tensor, e3: torch.Tensor) -> torch.Tensor:
        y_pool = resize(
            self.global_pool_branch(e4),
            size=e4.shape[2:],
            mode='bilinear',
            align_corners=False)

        y_loc = self.local_branch(e4)
        y_context = self.context_branches(e4)

        branch_responses = [y_pool, y_loc] + y_context
        y_cat = self.concat_fusion_refine(torch.cat(branch_responses, dim=1))

        if self.enable_cross_level_gating:
            y_gate = self.cross_level_gate(branch_responses, deep_feature=e4, skip_feature=e3)
            fused_bridge = (
                (1.0 - self.mix_coefficient) * y_gate
                + self.mix_coefficient * y_cat
            )
        else:
            fused_bridge = y_cat

        f32 = fused_bridge + self.residual_projection(e4)
        return f32


class BoundaryGatedProgressiveFusion(nn.Module):
    """BG-PF stage at a specific resolution.

    W^(s) = sigma(g([D^(s), C^(s)]))
    C_e^(s) = (1 - W^(s)) * C^(s)
    D_e^(s) = W^(s) * D^(s)
    Y^(s) = phi([C_e^(s), D_e^(s)])
    """

    def __init__(self,
                 context_channels: int,
                 detail_channels: int,
                 out_channels: int,
                 enable_spatial_gating: bool = True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.enable_spatial_gating = bool(enable_spatial_gating)

        if self.enable_spatial_gating:
            hidden_channels = max(32, (context_channels + detail_channels) // 4)
            self.spatial_gate = nn.Sequential(
                ConvModule(
                    context_channels + detail_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )

        self.fusion_refine = DepthwiseSeparableConvModule(
            context_channels + detail_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self,
                context_feature: torch.Tensor,
                detail_feature: torch.Tensor) -> torch.Tensor:
        if self.enable_spatial_gating:
            spatial_weight = torch.sigmoid(
                self.spatial_gate(torch.cat([detail_feature, context_feature], dim=1)))
            context_enhanced = (1.0 - spatial_weight) * context_feature
            detail_enhanced = spatial_weight * detail_feature
            fused_feature = torch.cat([context_enhanced, detail_enhanced], dim=1)
        else:
            fused_feature = torch.cat([context_feature, detail_feature], dim=1)

        return self.fusion_refine(fused_feature)


class MultiDilationStripRefinement(nn.Module):
    """Directional refinement used in BG-PF at OS=8."""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 11,
                 dilations: Iterable[int] = (1, 2),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        assert kernel_size % 2 == 1, 'bgpf_strip_kernel must be odd.'

        self.branches = nn.ModuleList()
        for dilation in tuple(dilations):
            padding = (kernel_size // 2) * dilation
            horizontal_branch = ConvModule(
                channels,
                channels,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
                dilation=(1, dilation),
                groups=channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            vertical_branch = ConvModule(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                dilation=(dilation, 1),
                groups=channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.branches.append(nn.ModuleList([horizontal_branch, vertical_branch]))

        self.fusion = ConvModule(
            2 * channels * len(self.branches),
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        directional_responses = []
        for horizontal_branch, vertical_branch in self.branches:
            directional_responses.append(horizontal_branch(x))
            directional_responses.append(vertical_branch(x))

        z = self.fusion(torch.cat(directional_responses, dim=1))
        yb_8 = x + z
        return yb_8


class StructureAwareMixtureOfExpertsFusion(nn.Module):
    """SA-MoE Fusion at OS=4."""

    def __init__(self,
                 channels: int,
                 expert_kernel: int = 11,
                 routing_temperature: float = 1.2,
                 stabilization_coefficient: float = 0.5,
                 boundary_guidance_strength: float = 0.6,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        if expert_kernel % 2 == 0:
            expert_kernel += 1

        self.routing_temperature = float(routing_temperature)
        self.stabilization_coefficient = float(stabilization_coefficient)
        self.boundary_guidance_strength = float(boundary_guidance_strength)

        padding = expert_kernel // 2

        # {E_h, E_v, E_loc, E_dil}
        self.horizontal_expert = DepthwiseSeparableConvModule(
            channels,
            channels,
            kernel_size=(1, expert_kernel),
            padding=(0, padding),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.vertical_expert = DepthwiseSeparableConvModule(
            channels,
            channels,
            kernel_size=(expert_kernel, 1),
            padding=(padding, 0),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.local_expert = DepthwiseSeparableConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.dilated_expert = DepthwiseSeparableConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.experts = nn.ModuleList([
            self.horizontal_expert,
            self.vertical_expert,
            self.local_expert,
            self.dilated_expert,
        ])

        hidden_channels = max(64, channels // 4)
        self.routing_network = nn.Sequential(
            ConvModule(
                channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.Conv2d(hidden_channels, len(self.experts), kernel_size=1),
        )

        self.boundary_embedding = ConvModule(
            1,
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.refinement_conv = DepthwiseSeparableConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self,
                fused_high_resolution_feature: torch.Tensor,
                boundary_logits: Optional[torch.Tensor],
                interpolate_mode: str = 'bilinear',
                align_corners: bool = False) -> torch.Tensor:
        if boundary_logits is None:
            boundary_response = fused_high_resolution_feature.new_zeros(
                fused_high_resolution_feature.size(0),
                1,
                fused_high_resolution_feature.size(2),
                fused_high_resolution_feature.size(3))
        else:
            boundary_response = resize(
                boundary_logits,
                size=fused_high_resolution_feature.shape[-2:],
                mode=interpolate_mode,
                align_corners=align_corners)
            boundary_response = torch.sigmoid(boundary_response)

        # G^(4) = F^(4) + lambda_g * phi_e(up sigma(B))
        boundary_guided_feature = (
            fused_high_resolution_feature
            + self.boundary_guidance_strength * self.boundary_embedding(boundary_response)
        )

        expert_responses = [expert(fused_high_resolution_feature) for expert in self.experts]
        stacked_expert_responses = torch.stack(expert_responses, dim=1)

        routing_logits = self.routing_network(boundary_guided_feature)
        routing_weights = torch.softmax(
            routing_logits / self.routing_temperature, dim=1).unsqueeze(2)

        z_moe = (routing_weights * stacked_expert_responses).sum(dim=1)
        z_mean = stacked_expert_responses.mean(dim=1)

        refined_feature = fused_high_resolution_feature + (
            (1.0 - self.stabilization_coefficient) * z_moe
            + self.stabilization_coefficient * z_mean
        )
        fb_4 = self.refinement_conv(refined_feature)
        return fb_4


@MODELS.register_module()
class BGMoENetHead(BaseDecodeHead):
    """Decoder head of BG-MoENet.

    Expected transformed inputs:
        [E1(OS=4), E2(OS=8), E3(OS=16), E4(OS=32)]
    """

    def __init__(self,
                 cmb_dilations: Tuple[int, ...] = (1, 6, 12, 18),
                 cmb_channels: int = 256,
                 c1_channels: int = 64,
                 interpolate_mode: str = 'bilinear',
                 enable_cmb_cross_level_gating: bool = True,
                 enable_bgpf_spatial_gating: bool = True,
                 enable_bgpf_strip_refinement: bool = True,
                 bgpf_strip_kernel: int = 11,
                 bgpf_strip_dilations: Tuple[int, ...] = (1, 2),
                 cmb_temperature: float = 3.5,
                 cmb_mix_coefficient: float = 0.6,
                 boundary_loss_weight: float = 0.12,
                 boundary_foreground_index: int = 1,
                 boundary_widen_kernel: int = 3,
                 enable_sa_moe_fusion: bool = True,
                 sa_moe_kernel: int = 11,
                 sa_moe_routing_temperature: float = 1.2,
                 sa_moe_stabilization_coefficient: float = 0.5,
                 boundary_guidance_strength: float = 0.6,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        assert len(self.in_channels) == len(self.in_index) == 4, (
            'BGMoENetHead expects 4 input feature maps: E1, E2, E3, E4.')

        self.interpolate_mode = interpolate_mode
        self.cmb_channels = int(cmb_channels)
        self.c1_channels = int(c1_channels)

        self.enable_bgpf_strip_refinement = bool(enable_bgpf_strip_refinement)
        self.enable_sa_moe_fusion = bool(enable_sa_moe_fusion)

        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_foreground_index = int(boundary_foreground_index)
        self.boundary_widen_kernel = int(boundary_widen_kernel)

        self._cached_boundary_logits: Optional[torch.Tensor] = None

        c1, c2, c3, c4 = self.in_channels

        # CMB at OS=32
        self.contextual_multi_branch_bridge = ContextualMultiBranchBridge(
            c4=c4,
            c3=c3,
            out_channels=self.cmb_channels,
            dilations=cmb_dilations,
            temperature=cmb_temperature,
            mix_coefficient=cmb_mix_coefficient,
            enable_cross_level_gating=enable_cmb_cross_level_gating,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Encoder-side projections
        self.encoder_projection_os16 = ConvModule(
            c3,
            self.channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.encoder_projection_os8 = ConvModule(
            c2,
            self.channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.encoder_projection_os4 = ConvModule(
            c1,
            self.c1_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # BG-PF16 / BG-PF8
        self.bgpf16 = BoundaryGatedProgressiveFusion(
            context_channels=self.cmb_channels,
            detail_channels=self.channels,
            out_channels=self.channels,
            enable_spatial_gating=enable_bgpf_spatial_gating,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.bgpf8 = BoundaryGatedProgressiveFusion(
            context_channels=self.channels,
            detail_channels=self.channels,
            out_channels=self.channels,
            enable_spatial_gating=enable_bgpf_spatial_gating,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.enable_bgpf_strip_refinement:
            self.multi_dilation_strip_refinement = MultiDilationStripRefinement(
                channels=self.channels,
                kernel_size=bgpf_strip_kernel,
                dilations=bgpf_strip_dilations,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        self.boundary_prediction_head = nn.Conv2d(self.channels, 1, kernel_size=1)

        # Fuse(up F8, E1) -> F^(4)
        self.high_resolution_fusion = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + self.c1_channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
        )

        if self.enable_sa_moe_fusion:
            self.structure_aware_moe_fusion = StructureAwareMixtureOfExpertsFusion(
                channels=self.channels,
                expert_kernel=sa_moe_kernel,
                routing_temperature=sa_moe_routing_temperature,
                stabilization_coefficient=sa_moe_stabilization_coefficient,
                boundary_guidance_strength=boundary_guidance_strength,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        self._cached_boundary_logits = None

        e1, e2, e3, e4 = self._transform_inputs(inputs)

        # F32 = CMB(E4, E3)
        f32 = self.contextual_multi_branch_bridge(e4, e3)

        # F16 = BG-PF16(up F32, E3)
        f32_up = resize(
            f32,
            size=e3.shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners)
        d16 = self.encoder_projection_os16(e3)
        f16 = self.bgpf16(context_feature=f32_up, detail_feature=d16)

        # {F8, B} = BG-PF8(up F16, E2)
        f16_up = resize(
            f16,
            size=e2.shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners)
        d8 = self.encoder_projection_os8(e2)
        f8 = self.bgpf8(context_feature=f16_up, detail_feature=d8)

        if self.enable_bgpf_strip_refinement:
            f8 = self.multi_dilation_strip_refinement(f8)

        boundary_logits = self.boundary_prediction_head(f8)
        self._cached_boundary_logits = boundary_logits

        # F^(4) = Fuse(up F8, E1)
        f8_up = resize(
            f8,
            size=e1.shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners)
        f4 = self.high_resolution_fusion(
            torch.cat([f8_up, self.encoder_projection_os4(e1)], dim=1))

        # Fe4 / Fb^(4) = SA-MoE(F^(4), up B)
        if self.enable_sa_moe_fusion:
            fe4 = self.structure_aware_moe_fusion(
                fused_high_resolution_feature=f4,
                boundary_logits=boundary_logits,
                interpolate_mode=self.interpolate_mode,
                align_corners=self.align_corners)
        else:
            fe4 = f4

        return self.cls_seg(fe4)

    @torch.no_grad()
    def _stack_gt_semantic_labels(self, batch_data_samples: List) -> torch.Tensor:
        """Stack gt_sem_seg into [N, 1, H, W]."""
        gt_list = []
        for sample in batch_data_samples:
            if hasattr(sample, 'gt_sem_seg'):
                gt = sample.gt_sem_seg.data
            else:
                gt = sample['gt_sem_seg'].data
            gt_list.append(gt)
        return torch.stack(gt_list, dim=0).squeeze(1)[:, None, ...]

    @torch.no_grad()
    def _make_boundary_target(self,
                              gt: torch.Tensor,
                              out_hw: Tuple[int, int]) -> torch.Tensor:
        """Generate boundary supervision B* online from segmentation labels."""
        ignore_index = self.ignore_index if self.ignore_index is not None else 255
        h_out, w_out = out_hw

        gt_downsampled = F.interpolate(
            gt.float(), size=(h_out, w_out), mode='nearest').long()

        valid_mask = (gt_downsampled != ignore_index)
        foreground_mask = (gt_downsampled == self.boundary_foreground_index).float()

        # Morphological gradient
        dilated = F.max_pool2d(foreground_mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-foreground_mask, kernel_size=3, stride=1, padding=1)
        boundary_target = (dilated - eroded) > 0

        if self.boundary_widen_kernel is not None and self.boundary_widen_kernel > 1:
            kernel = int(self.boundary_widen_kernel)
            padding = kernel // 2
            boundary_target = F.max_pool2d(
                boundary_target.float(),
                kernel_size=kernel,
                stride=1,
                padding=padding) > 0

        ignore_region = (~valid_mask).float()
        ignore_neighborhood = F.max_pool2d(
            ignore_region, kernel_size=3, stride=1, padding=1) > 0

        boundary_target = boundary_target & valid_mask & (~ignore_neighborhood)
        return boundary_target.float()

    def loss(self, inputs: Sequence[torch.Tensor], batch_data_samples, train_cfg=None):
        seg_logits = self.forward(inputs)
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        boundary_logits = self._cached_boundary_logits
        if boundary_logits is not None and self.boundary_loss_weight > 0:
            with torch.no_grad():
                gt = self._stack_gt_semantic_labels(batch_data_samples).to(boundary_logits.device)
                boundary_target = self._make_boundary_target(
                    gt, out_hw=boundary_logits.shape[-2:])

                ignore_index = self.ignore_index if self.ignore_index is not None else 255
                valid_mask = (
                    F.interpolate(
                        (gt != ignore_index).float(),
                        size=boundary_logits.shape[-2:],
                        mode='nearest') > 0
                ).float()

            bce = F.binary_cross_entropy_with_logits(
                boundary_logits, boundary_target, reduction='none')
            bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            losses['loss_boundary'] = bce * self.boundary_loss_weight

        return losses