# opengait/modeling/models/pruegait.py
# -*- coding: utf-8 -*-

import os
import math
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ..base_model import BaseModel
import csv
from pathlib import Path

# =========================================================================
# Region 0: Utils
# =========================================================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_.",
            stacklevel=2
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def forward(self, x):
        if self.drop_prob == 0. or (not self.training):
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def torso_unit_norm(pose: torch.Tensor) -> torch.Tensor:
    """
    pose: [N, T, 17, 3]
    """
    if pose.numel() == 0:
        return pose
    if pose.sum() == 0:
        return pose
    torso_idx = [5, 6, 11, 12]
    center = pose[:, :, torso_idx, :].mean(dim=2, keepdim=True)
    pose = pose - center
    scale = torch.norm(pose, p=2, dim=-1, keepdim=True).max(dim=2, keepdim=True)[0]
    pose = pose / (scale + 1e-6)
    return pose


def _rank0():
    if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()):
        return True
    return torch.distributed.get_rank() == 0


# =========================================================================
# Region 1: Visual Stream
# =========================================================================

class HPM(nn.Module):
    def __init__(self, bin_list=(1, 2, 4, 8, 16)):
        super().__init__()
        self.bin_list = list(bin_list)

    def forward(self, x):
        # x: [N, C, H, W]
        n, c, H, W = x.size()
        feats = []
        for b in self.bin_list:
            # 1. 垂直方向全局池化，水平方向分 b 段
            # [N, C, H, W] -> [N, C, b, 1]
            z = F.adaptive_max_pool2d(x, (b, 1))
            # 2. 去掉宽度维 -> [N, C, b]
            z = z.squeeze(-1)
            feats.append(z)
        # 3. 在 Part 维度拼接
        # 结果形状: [N, C, sum(bin_list)] = [N, C, 31]
        return torch.cat(feats, dim=2)


class VisualBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)
        # [FIX 1] 修改 Conv1 为单通道输入，避免 forward 时 repeat(1,3,1,1)
        # 这样输入可以是 [N, 1, H, W]，更加高效
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # 将 RGB 权重的均值初始化给单通道，保留预训练的纹理提取能力
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        backbone.conv1=new_conv
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

        # Reduce final stride (keep more spatial detail)
        for m in self.features[-1].modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (1, 1)
            if isinstance(m, nn.Sequential):
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Conv2d) and sub_m.stride == (2, 2):
                        sub_m.stride = (1, 1)

    def forward(self, x):
        # x: [N, 1, H, W] or [N, 3, H, W]
        # 如果前面数据处理已经是单通道，这里直接进；如果是3通道也兼容
        if x.shape[1] == 3:
            # 如果不想改 conv1，可以用 x = x[:, 0:1, :, :] 取单通道
            # 但既然改了 conv1，这里就主要为了防呆
            x = x.mean(dim=1, keepdim=True)
        return self.features(x)


# =========================================================================
# Region 2: Structural Stream (DSTformer + ViewLoRA)
# =========================================================================

class ViewLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank=4, num_views=11):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.num_views = num_views

        self.lora_A = nn.Parameter(torch.randn(num_views, self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(num_views, rank, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        # flag for Attention to detect
        self.view_modulator = True

    def forward(self, x, view_idx=None):
        out = self.base_layer(x)
        if view_idx is None:
            return out

        # x: [BF, N, in_features]
        BF = x.shape[0]
        B = view_idx.shape[0]

        if BF != B:
            # expand view idx from [B] -> [BF]
            curr_view_idx = view_idx.unsqueeze(1).expand(-1, BF // B).reshape(-1)
        else:
            curr_view_idx = view_idx

        curr_view_idx = curr_view_idx.clamp(0, self.num_views - 1)

        # LoRA: x @ A @ B
        lora = torch.bmm(torch.bmm(x, self.lora_A[curr_view_idx]), self.lora_B[curr_view_idx])
        return out + lora


def inject_view_lora(model: nn.Module, rank=4, num_views=11):
    """
    Replace specific Linear layers with ViewLoRALinear for view conditioning.
    """
    device = next(model.parameters()).device
    module_dict = dict(model.named_modules())

    for name, module in list(model.named_modules()):
        if any(k in name for k in [ "mlp_s.fc", "mlp_t.fc"]):
            parent_name = ".".join(name.split(".")[:-1])
            target_name = name.split(".")[-1]
            parent = module_dict.get(parent_name, None)
            if parent is None:
                continue
            old = getattr(parent, target_name)
            if isinstance(old, ViewLoRALinear):
                continue
            if not isinstance(old, nn.Linear):
                continue
            setattr(parent, target_name, ViewLoRALinear(old, rank=rank, num_views=num_views).to(device))
    return model


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, view_idx=None):
        if hasattr(self.fc1, "view_modulator"):
            x = self.fc1(x, view_idx=view_idx)
        else:
            x = self.fc1(x)

        x = self.act(x)
        x = self.drop(x)

        if hasattr(self.fc2, "view_modulator"):
            x = self.fc2(x, view_idx=view_idx)
        else:
            x = self.fc2(x)

        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0., proj_drop=0., st_mode="vanilla"
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mode = st_mode

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seqlen=1, view_idx=None):
        # x: [B, N, C]
        B, N, C = x.shape

        if hasattr(self.qkv, "view_modulator"):
            qkv = self.qkv(x, view_idx=view_idx)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads) \
                 .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.mode == "temporal":
            x_out = self.forward_temporal(q, k, v, seqlen=seqlen)
        else:
            x_out = self.forward_spatial(q, k, v)

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def forward_spatial(self, q, k, v):
        # q,k,v: [B, heads, N, C]
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).contiguous().reshape(B, N, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v, seqlen=8):
        # B here is BF in DSTformer
        B, _, N, C = q.shape
        # BF must be divisible by seqlen
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (self.attn_drop(attn) @ vt).permute(0, 3, 2, 1, 4).contiguous().reshape(B, N, C * self.num_heads)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0., st_mode="stage_st"):
        super().__init__()
        self.norm1_s = nn.LayerNorm(dim)
        self.norm1_t = nn.LayerNorm(dim)

        self.attn_s = Attention(dim, num_heads=num_heads, st_mode="spatial")
        self.attn_t = Attention(dim, num_heads=num_heads, st_mode="temporal")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_s = nn.LayerNorm(dim)
        self.norm2_t = nn.LayerNorm(dim)

        self.mlp_s = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.mlp_t = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

        self.st_mode = st_mode

    def forward(self, x, seqlen=1, view_idx=None):
        if self.st_mode == "stage_st":
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen=seqlen, view_idx=view_idx))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x), view_idx=view_idx))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen=seqlen, view_idx=view_idx))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x), view_idx=view_idx))
        else:
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen=seqlen, view_idx=view_idx))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x), view_idx=view_idx))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen=seqlen, view_idx=view_idx))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x), view_idx=view_idx))
        return x


class DSTformer(nn.Module):
    def __init__(self, dim_feat=256, dim_rep=512, depth=5, num_heads=8, num_joints=17, maxlen=243):
        super().__init__()
        self.joints_embed = nn.Linear(3, dim_feat)
        self.pos_drop = nn.Dropout(p=0.)

        self.blocks_st = nn.ModuleList([Block(dim=dim_feat, num_heads=num_heads, st_mode="stage_st") for _ in range(depth)])
        self.blocks_ts = nn.ModuleList([Block(dim=dim_feat, num_heads=num_heads, st_mode="stage_ts") for _ in range(depth)])

        self.norm = nn.LayerNorm(dim_feat)
        self.pre_logits = nn.Sequential(OrderedDict([
            ("fc", nn.Linear(dim_feat, dim_rep)),
            ("act", nn.Tanh())
        ]))

        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for _ in range(depth)])
        self.dim_feat = dim_feat

    def get_representation(self, x, view_idx=None):
        # x: [B, F, J, 3]
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)  # [BF, J, 3]
        BF = x.shape[0]

        x = self.joints_embed(x) + self.pos_embed  # [BF, J, dim]
        x = x.reshape(-1, F, J, self.dim_feat) + self.temp_embed[:, :F, :, :]
        x = x.view(B * F, J, self.dim_feat)  #  [BF, J, dim]
        x = self.pos_drop(x)

        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, seqlen=F, view_idx=view_idx)
            x_ts = blk_ts(x, seqlen=F, view_idx=view_idx)
            alpha = self.ts_attn[idx](torch.cat([x_st, x_ts], dim=-1)).softmax(dim=-1)
            x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]

        x = self.norm(x).reshape(B, F, J, -1)
        x = self.pre_logits(x)
        return x


class StructuralBackbone(nn.Module):
    def __init__(self, embed_dim=256, dim_rep=512):
        super().__init__()
        self.encoder = DSTformer(dim_feat=embed_dim, dim_rep=dim_rep)

    def forward(self, x, view_idx=None):
        # x: [B, T, 51]
        B, T, _ = x.shape
        x = x.reshape(B, T, 17, 3)
        x = self.encoder.get_representation(x, view_idx=view_idx)  # [B,T,17,dim_rep]
        return x.mean(dim=2)  # [B,T,dim_rep]


# =========================================================================
# Region 3: Fusion + Heads
# =========================================================================

class ViewAwarePPM(nn.Module):
    def __init__(self, feature_dim=512, num_views=11):
        super().__init__()
        self.view_embedding = nn.Embedding(num_views, 128)
        self.view_gate = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x, view_idx):
        # x: [B, T, D]
        v_mask = self.view_gate(self.view_embedding(view_idx))  # [B, D]
        return (x * v_mask.unsqueeze(1)).mean(dim=1)           # [B, D]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, common_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, common_dim),
            nn.BatchNorm1d(common_dim),
            nn.ReLU(inplace=True),
            nn.Linear(common_dim, common_dim)
        )

    def forward(self, x):
        if x.dim() == 2:  # [N,C]
            return self.net(x)
        n, p, c = x.shape  # [N,P,C]
        x_flat = x.reshape(-1, c)
        x_flat = self.net(x_flat)
        return x_flat.reshape(n, p, -1)


class FAMFusion(nn.Module):
    def __init__(self, common_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim * 2),
            nn.BatchNorm1d(common_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(common_dim * 2, common_dim)
        )

    def forward(self, f_vis, f_struct):
        # f_vis:    [N, P, C]
        # f_struct: [N, P, C] (已广播)

        # 1. 拼接 -> [N, P, 2C]
        x = torch.cat((f_vis, f_struct), dim=-1)

        # 2. 展平 -> [N*P, 2C]
        n, p, c = x.shape
        x_flat = x.reshape(-1, c)

        # 3. 进网络
        x_flat = self.net(x_flat)

        # 4. 还原 -> [N, P, C]
        return x_flat.reshape(n, p, -1)


# =========================================================================
# Region 4: Main Model
# =========================================================================

class PAGFSLModel(BaseModel):
    """
    Dual-stream (silhouette + skeleton) model for OpenGait.
    Designed to be robust for first-run debugging:
      - no hard-coded 64x64 in visual_summary
      - reshape instead of view to avoid non-contiguous issues
      - labs forced to long [B]
      - optional one-time debug prints
    """

    def build_network(self, model_cfg):
        self.common_dim = model_cfg.get("Common_Dim", 512)
        self.hpm_bins = model_cfg.get("bin_list", [1, 2, 4, 8, 16])
        self.num_classes = model_cfg.get("num_classes", 74)

        self.debug_flow = bool(model_cfg.get("debug_flow", True))
        self.pretrained_backbone = bool(model_cfg.get("pretrained_backbone", False))

        # Visual
        self.vis_backbone = VisualBackbone(pretrained=self.pretrained_backbone)
        self.hpm = HPM(bin_list=self.hpm_bins)

        # Structural
        self.struct_backbone = StructuralBackbone(dim_rep=self.common_dim)
        self.struct_backbone = inject_view_lora(self.struct_backbone, rank=4, num_views=11)

        # Heads
        self.ppm = ViewAwarePPM(feature_dim=self.common_dim, num_views=11)
        self.proj_vis = ProjectionHead(input_dim=512, common_dim=self.common_dim)
        self.proj_struct = ProjectionHead(input_dim=self.common_dim, common_dim=self.common_dim)
        self.fam = FAMFusion(self.common_dim)

        # Classifier
        if self.num_classes!=74:
            raise ValueError("num_classes must be 74,but now be",self.num_classes)
        self.classifier = nn.Linear(self.common_dim, self.num_classes)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.classifier.bias, 0.)

        # View map
        self.view_map = {v: i for i, v in enumerate(
            ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
        )}

        self._dbg_printed = False
        if _rank0():
            for i, (n, p) in enumerate(self.named_parameters()):
                if i in {82, 83, 86, 87, 90, 91, 94, 95}:  # 先挑几个看
                    print(i, n, p.shape, p.requires_grad)

        self.monitor_enable = bool(model_cfg.get("monitor_enable", True))
        self.monitor_every = int(model_cfg.get("monitor_every", 200))  # 每多少 iter 记录一次
        self.monitor_dir = Path(model_cfg.get("monitor_dir", "output/CASIABFusion/PAGFSLModel/PAGFSL_Fusion/pic"))
        self.monitor_margin = float(model_cfg.get("monitor_triplet_margin", 0.2))

        self._iter_cnt = 0
        self._monitor_inited = False
        # seed = int(model_cfg.get("seed", 0))
        # init_seeds(seed)
    def _debug_once(self, **tensors):
        if (not self.debug_flow) or self._dbg_printed or (not _rank0()):
            return
        self._dbg_printed = True
        print("\n========== [PAGFSLModel DEBUG] ==========")
        for k, v in tensors.items():
            if v is None:
                print(f"{k}: None")
            elif isinstance(v, torch.Tensor):
                msg = f"{k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
                if v.numel() and v.dtype.is_floating_point:
                    msg += f" min={v.min().item():.4f} max={v.max().item():.4f}"
                print(msg)
            else:
                print(f"{k}: {type(v)} {v}")
        print("========================================\n")

    def _batch_hard_triplet_stats(self, emb: torch.Tensor, labels: torch.Tensor, margin=0.2):
        """
        emb: [B, D] (L2 normalized or not)
        labels: [B]
        return: triplet_loss, pos_mean, neg_mean 若本卡全是孤儿则返回 None
        """
        B = emb.size(0)
        if B <= 1:
            return None
        # cosine distance (稳定且适合 gait embedding)
        emb = F.normalize(emb.float(), p=2, dim=-1)
        sim = emb @ emb.t()  # [B,B]
        dist = 1.0 - sim  # cosine dist in [0,2]

        labels = labels.view(-1, 1)  # [B,1]
        eye = torch.eye(B, device=dist.device, dtype=torch.bool)
        same = (labels == labels.t()) & (~eye)
        diff = (labels != labels.t()) & (~eye)



        # 3) valid anchor: 本卡必须同时有正样本和负样本
        has_pos = same.any(dim=1)  #  bool [B]
        # 对负样本：diff 包括对角线，因此要把对角线去掉
        has_neg = diff.any(dim=1)
        valid = has_pos & has_neg
        if valid.sum().item() == 0:
            return None
        # hardest positive: 最大距离的同类
        pos_dist = dist.masked_fill(~same, float("-inf"))
        neg_dist = dist.masked_fill(~diff, float("inf"))

        hardest_pos = pos_dist.max(dim=1).values[valid]
        hardest_neg = neg_dist.min(dim=1).values[valid]
        trip = F.relu(margin + hardest_pos - hardest_neg).mean()
        return trip, hardest_pos.mean(), hardest_neg.mean()

    def _monitor_init_if_needed(self):
        if (not self.monitor_enable) or self._monitor_inited:
            return
        if not _rank0():
            self._monitor_inited = True
            return
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.monitor_dir / "train_monitor.csv"
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                m = csv.writer(f)
                m.writerow(["iter", "ce_loss", "tri_loss", "pos_dist", "neg_dist",
                            "emb_norm", "fvis_norm", "fstruct_norm"])
        self._monitor_inited = True

    @torch.no_grad()
    def _monitor_plot(self):
        # 只在 rank0 画图
        if not _rank0():
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs, ce, tri, pos, neg = [], [], [], [], []
            with open(self._csv_path, "r") as f:
                r = csv.DictReader(f)
                for row in r:
                    xs.append(int(row["iter"]))
                    ce.append(float(row["ce_loss"]))
                    tri.append(float(row["tri_loss"]))
                    vpos = float(row["pos_dist"])
                    vneg = float(row["neg_dist"])
                    if not (math.isfinite(vpos) and math.isfinite(vneg)):
                        continue
                    pos.append(vpos);
                    neg.append(vneg)

            if len(xs) < 2:
                return

            # loss 图
            plt.figure()
            plt.plot(xs, ce, label="CE")
            plt.plot(xs, tri, label="Triplet")
            plt.legend()
            plt.xlabel("iter")
            plt.ylabel("loss")
            plt.tight_layout()
            plt.savefig(self.monitor_dir / "loss_curve.png")
            plt.close()

            # dist 图
            plt.figure()
            plt.plot(xs, pos, label="pos_dist")
            plt.plot(xs, neg, label="neg_dist")
            plt.legend()
            plt.xlabel("iter")
            plt.ylabel("cos_dist")
            plt.tight_layout()
            plt.savefig(self.monitor_dir / "dist_curve.png")
            plt.close()

        except Exception as e:
            # 画图失败不影响训练
            pass

    def forward(self, inputs):
        # OpenGait inputs: [ipts, labs, typs, vies, seqL, ...]
        ipts, labs, _, views, _ = inputs[:5]
        device = labs.device
        # labels stable
        labs = labs.reshape(-1).long()

        # ipts should be [sils, skels] when data_in_use=[true,true]
        if not isinstance(ipts, (list, tuple)) or len(ipts) < 1:
            raise RuntimeError(f"ipts invalid: type={type(ipts)} len={len(ipts) if hasattr(ipts,'__len__') else 'NA'}")

        # -------- infer sils / skels robustly --------
        sils, skels = None, None

        def _is_sils(x: torch.Tensor) -> bool:
            # sils: [N,T,H,W] or [N,1,T,H,W] (H/W 不一定相等，比如 64x44)
            if x.dim() == 5:
                # [N,C,T,H,W]
                H, W = x.shape[-2], x.shape[-1]
                return (H >= 32 and W >= 32)
            if x.dim() == 4:
                # [N,T,H,W]
                # 排除 skeleton 的 [N,T,17,3]
                if x.shape[-2] == 17 and x.shape[-1] == 3:
                    return False
                H, W = x.shape[-2], x.shape[-1]
                return (H >= 32 and W >= 32)
            return False

        def _is_skels(x: torch.Tensor) -> bool:
            # skels 常见: [N,T,17,3] 或 [N,T,51]
            if x.dim() == 4 and x.shape[-2] == 17 and x.shape[-1] == 3:
                return True
            if x.dim() == 3 and x.shape[-1] in (51, 34):
                return True
            return False

        for x in ipts:
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            # 注意：这里不 .to(device)，先判断类型/shape
            if sils is None and _is_sils(x):
                sils = x
            elif skels is None and _is_skels(x):
                skels = x

        if sils is None:
            raise RuntimeError(f"Cannot infer sils from ipts shapes: {[tuple(torch.as_tensor(z).shape) for z in ipts]}")

        # ---- sils: ensure tensor, float, [N,1,T,H,W] ----
        if not torch.is_tensor(sils):
            sils = torch.as_tensor(sils)
        sils = sils.to(device)

        if sils.dim() == 4: # [N,T,H,W]
            sils = sils.unsqueeze(1)  # [N,1,T,H,W]
        elif sils.dim() == 5:  # [N,1,T,H,W] or [N,C,T,H,W]
            pass
        else:
            raise RuntimeError(f"sils dim must be 4 or 5, got {sils.dim()} with shape {tuple(sils.shape)}")

        # normalize value range if needed
        if sils.dtype != torch.float32 and sils.dtype != torch.float16:
            sils = sils.float()
        if sils.numel() and sils.max() > 1.5:
            sils = sils / 255.0

        n, c, t,H, W = sils.shape

        # ---- skels: ensure tensor, float32, [N,T,51] then torso norm ----
        x_struct = None
        if skels is not None:
            if not torch.is_tensor(skels):
                skels = torch.as_tensor(skels)
            x_struct = skels.to(device).float()

            # allow [N,T,17,3] or [N,T,51]
            if x_struct.dim() == 4:
                x_struct = x_struct.reshape(x_struct.size(0), x_struct.size(1), -1)
            if x_struct.dim() != 3:
                raise RuntimeError(f"skels dim must be 3 or 4, got {x_struct.dim()} shape={tuple(x_struct.shape)}")
            if x_struct.shape[-1] != 51:
                raise ValueError(f"Skeleton data shape error! Expected last dim=51 (17*3), but got {x_struct.shape}. "
                                 f"Check your .pkl files or datasets/merge_sil_sk_modality.py")
            # torso norm
            x_struct = torso_unit_norm(
                x_struct.reshape(x_struct.size(0), x_struct.size(1), 17, 3)
            ).reshape(x_struct.size(0), x_struct.size(1), -1)
        # ================= [DEBUG START]检查视角 =================
        # 利用你已有的 _dbg_printed 标志，确保只打印一次，且只在主进程打印
        if self.debug_flow and (not self._dbg_printed) and _rank0():
            print("\n" + "=" * 40)
            print("[DEBUG CHECK] Views Data Inspection")
            print(f"Raw 'views' type: {type(views)}")
            print(f"Raw 'views' content (first batch): {views}")

            if len(views) > 0:
                v_sample = views[0]
                print(f"Sample element type: {type(v_sample)}")
                print(f"Sample element value: {v_sample}")


                if isinstance(v_sample, str):
                    v_str_test = v_sample
                    print(f" Logic Branch: String detected. Key will be '{v_str_test}'")
                else:
                    v_int_test = int(v_sample.item()) if hasattr(v_sample, "item") else int(v_sample)
                    v_str_test = f"{v_int_test:03d}"
                    print(f" Logic Branch: Number detected. Formatted key will be '{v_str_test}'")

                mapped_idx = self.view_map.get(v_str_test, "MISSING")
                if mapped_idx == "MISSING":
                    print(f" [WARNING] Key '{v_str_test}' NOT FOUND in view_map! It will fallback to index 5.")
                else:
                    print(f" [SUCCESS] Key '{v_str_test}' maps to index {mapped_idx}.")
            print("=" * 40 + "\n")
        # ================= [DEBUG END] =================
        # ---- view_idx ----
        view_idx = []
        for v in views:
            # handle int/torch scalar/str like "000"
            if isinstance(v, str):
                v_str = v
            else:
                v_int = int(v.item()) if hasattr(v, "item") else int(v)
                v_str = f"{v_int:03d}"
            view_idx.append(self.view_map.get(v_str, 5))
        view_idx = torch.tensor(view_idx, device=device).long()

        self._debug_once(labs=labs, sils=sils, skels=x_struct, view_idx=view_idx)

        # ---- Visual stream ----
        x_vis = sils.reshape(n * t, c, H, W)#
        f_map = self.vis_backbone(x_vis)  # 原来的 ResNet / CNN
        f_hpm = self.hpm(f_map)  # [N*T, C, P]
        f_hpm = f_hpm.permute(0, 2, 1)  # [N*T, P, C]
        P = f_hpm.size(1)
        C = f_hpm.size(2)
        f_hpm = f_hpm.reshape(n, t, P, C)
        f_vis = f_hpm.max(dim=1)[0]  # [B, P, C]
        f_vis = self.proj_vis(f_vis)  # [B, P, D]

        if not hasattr(self, "_vis_debugged"):
            self._vis_debugged = True
            print("sils:", sils.shape, sils.dtype, sils.min().item(), sils.max().item())
            print("x_vis:", x_vis.shape)
            print("f_map:", f_map.shape)
            tmp = self.hpm(f_map)
            print("hpm:", tmp.shape)
        # ---- Structural stream ----
        if x_struct is not None and x_struct.abs().sum() > 0:
            f_struct_seq = self.struct_backbone(x_struct, view_idx=view_idx)  # [N,T,D]
            f_struct = self.ppm(f_struct_seq, view_idx=view_idx)  # [B,D]
            f_struct = self.proj_struct(f_struct)  # [B,D]
            f_struct = f_struct.unsqueeze(1).expand(-1, f_vis.size(1), -1)  # [B,P,D]

        else:
            f_struct = torch.zeros_like(f_vis)

        # ---- Fusion ----
        final_feat = self.fam(f_vis, f_struct)  # [B,P,D]

        emb_part = F.normalize(final_feat, p=2, dim=-1)  # [B,P,D]
        emb_global = F.normalize(emb_part.mean(dim=1), p=2, dim=-1) # [B,D]
        logits = self.classifier(emb_global)  # [B,74]
        emb_eval = emb_part.permute(0, 2, 1).contiguous().float()  # [B,D,P] fp32
        logits_part = self.classifier(emb_part)  # [B,P,C]
        logits_part = logits_part.permute(0, 2, 1).contiguous()  # [B,C,P]
        self._debug_once(f_vis=f_vis, f_struct=f_struct, emb_part=emb_part, emb_global=emb_global, logits=logits)

        # ===== monitor (train only) =====
        if self.training:
            self._iter_cnt += 1  # 只在训练 step 递增
            iter_now = self._iter_cnt

            if self.monitor_enable:
                self._monitor_init_if_needed()

                if _rank0() and (self._iter_cnt % self.monitor_every == 0):
                    # 监控版 CE（用你当前 logits + labs）
                    # 逐 part CE 后求均值
                    ce_loss = sum(F.cross_entropy(logits_part[:, :, p], labs) for p in
                                  range(logits_part.size(2))) / logits_part.size(2)

                    # 监控版 triplet（用 emb_global）
                    stats = self._batch_hard_triplet_stats(emb_global, labs, margin=self.monitor_margin)
                    if stats is None:
                        tri_loss = torch.tensor(0.0, device=device)
                        pos_d = torch.tensor(float("nan"), device=device)
                        neg_d = torch.tensor(float("nan"), device=device)
                    else:
                        tri_loss, pos_d, neg_d = stats

                    emb_norm = emb_global.norm(p=2, dim=-1).mean()
                    fvis_norm = f_vis.norm(p=2, dim=-1).mean() if f_vis.dim() == 2 else f_vis.norm(p=2, dim=-1).mean()
                    fstruct_norm = f_struct.norm(p=2, dim=-1).mean() if f_struct.dim() == 2 else f_struct.norm(p=2,
                                                                                                               dim=-1).mean()

                    with open(self._csv_path, "a", newline="") as f:
                        m = csv.writer(f)
                        m.writerow([iter_now,
                                    float(ce_loss.item()),
                                    float(tri_loss.item()),
                                    float(pos_d.item()),
                                    float(neg_d.item()),
                                    float(emb_norm.item()),
                                    float(fvis_norm.item()),
                                    float(fstruct_norm.item())])

                    # 每次写完顺手更新 png（你也可以改成每 5 次写一次图）
                    self._monitor_plot()  
        # [DEBUG PROBE] 强力诊断：每 500 步检查一次传入 Loss 的数据是否正常
        if self.training and _rank0() and (self._iter_cnt % 500 == 0):
            print("\n" + "=" * 20 + " [Triplet Loss Input Check] " + "=" * 20)
            print(f"Iter: {self._iter_cnt}")

            # 1. 检查 Embedding 形状
            # OpenGait TripletLoss 通常期待 [N, C, P] 或 [N, P, C]
            # 你的 emb_eval 是 [B, D, P] (e.g., 32, 512, 31)
            print(f"Emb Shape: {emb_eval.shape} (Expect [B, 512, 31])")

            # 2. 检查数值健康度 (是否含 NaN/Inf)
            if torch.isnan(emb_eval).any() or torch.isinf(emb_eval).any():
                print("!!! [ALERT] NaN/Inf found in embeddings !!!")
            else:
                print(
                    f"Emb Range: min={emb_eval.min():.4f}, max={emb_eval.max():.4f}, mean={emb_eval.mean():.4f}")

            # 3. 检查 Label 分布 (这是 triplet_loss_num 暴跌的头号嫌疑人)
            # 这里的 labs 是当前这张卡(Rank 0)拿到的局部数据
            print(f"Labs Shape: {labs.shape}")
            unique_ids, counts = torch.unique(labs, return_counts=True)
            print(f"Unique IDs on Rank0: {len(unique_ids)}")
            print(f"Min samples per ID: {counts.min().item()} (Should be >= 2 for Triplet!)")
            print(f"Max samples per ID: {counts.max().item()}")

            # 如果 Min samples < 2，说明 DDP 切分后出现了“孤儿”，TripletLoss 无法计算正样本对
            if counts.min().item() < 2:
                print("!!! [CRITICAL] Orphan IDs detected! This causes Loss=0 and loss_num drop. !!!")

            print("=" * 60 + "\n")

        return {
            "training_feat": {
                # 注意：如果你仍然遇到 ctypes.ArgumentError，99% 是 loss 的参数名不匹配。
                # 最常见的是这些 key；不匹配时你需要改成 feat/label 或 inputs/targets（看你本地 loss 实现）。
                "triplet": {"embeddings": emb_eval,"labels": labs},
                "softmax": {"logits": logits_part, "labels": labs}
            },
            "visual_summary": {
                "image/sils": sils.reshape(-1, 1, H, W)[:4]
            },
            "inference_feat": {
                "embeddings": emb_eval
            }
        }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()