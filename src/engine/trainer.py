"""
MRI 预训练用的 DINO Trainer（性能优化版）。

基于 SPECTRE 的 `pretrain_dino.py`（MIT License）重构而来：
  - 将训练循环抽离到 Trainer 类中
  - 支持单数据源（IXIDataset / MRIDataset）和多数据源（MultiSourceMRIDataset）
  - 支持区域均衡采样（WeightedRandomSampler）
  - 保留完整的 DINO 训练流程：余弦 LR/WD/momentum、EMA、梯度裁剪、
    cancel_last_layer_gradients、checkpoint 保存与恢复

性能优化：
  - 混合精度训练（fp16/bf16），显存减半、速度翻倍
  - torch.compile 自动优化模型图（PyTorch 2.0+）
  - MONAI CacheDataset 缓存确定性预处理结果，消除重复 IO
  - cudnn benchmark 自动选择最快卷积算法
  - DataLoader prefetch_factor 预加载下一批数据
"""
import os
import time
from itertools import chain

import torch
import torch.nn as nn
from accelerate import Accelerator, DataLoaderConfiguration
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import (
    DINOTransform,
    SafeDINOTransform,
    IXIDataset,
    MRIDataset,
    MultiSourceMRIDataset,
)
from src.data.collate import collate_dino
from src.models import DINO, DINOLoss, GramLoss
from src.models import vit_base_patch16_96, vit_base_rope_patch16_96, vit_small_patch16_96
from src.utils import (
    cosine_schedule,
    cosine_warmup_schedule,
    load_state,
    save_state,
    update_momentum,
)
from src.utils.misc import fix_random_seeds
from src.utils.param_groups import get_param_groups_with_decay


# 可用 ViT 架构注册表
_ARCHITECTURES = {
    "vit_small_patch16_96": vit_small_patch16_96,
    "vit_base_patch16_96": vit_base_patch16_96,
    "vit_base_rope_patch16_96": vit_base_rope_patch16_96,
}


class Trainer:
    """DINO 预训练器（性能优化版）。

    负责完整训练流程，包括模型构建、数据加载、优化器初始化、
    训练迭代、checkpoint 管理与日志记录。

    性能优化项：
      - mixed_precision: "fp16" / "bf16" / "no"（配置文件控制）
      - use_compile: 是否启用 torch.compile（PyTorch 2.0+）
      - cache_rate: MONAI CacheDataset 的缓存比例（0.0-1.0）
      - cudnn benchmark: 自动选择最快卷积实现
      - prefetch_factor: DataLoader 预取批次数
    """

    def __init__(self, cfg):
        # 固定配置与随机种子
        self.cfg = cfg
        fix_random_seeds(cfg.train.seed)

        # --- 性能优化：cudnn benchmark ---
        torch.backends.cudnn.benchmark = True

        # --- 混合精度配置 ---
        mixed_precision = cfg.train.get("mixed_precision", "fp16")

        dataloader_config = DataLoaderConfiguration(
            non_blocking=cfg.train.pin_memory,
        )

        # 使用 Hugging Face Accelerate 管理训练
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train.grad_accum_steps,
            log_with="wandb" if cfg.train.log_wandb else None,
            dataloader_config=dataloader_config,
            mixed_precision=mixed_precision,
        )

        if mixed_precision != "no":
            self.accelerator.print(
                f"Mixed precision: {mixed_precision} "
                f"(~2x speed, ~0.5x memory)"
            )

        if cfg.train.log_wandb:
            self.accelerator.init_trackers(
                project_name="dinov3-mri",
                config={k: v for d in cfg.values() for k, v in d.items()},
                init_kwargs={"wandb": {"dir": os.path.join(cfg.train.output_dir, "logs")}},
            )

        self.print = self.accelerator.print

    def _build_transform(self):
        """构建 DINO 多裁剪变换流水线。"""
        cfg = self.cfg

        # 判断是否启用前景裁剪
        use_foreground_crop = cfg.model.get("use_foreground_crop", False)

        # roi_size 可以为 None（使用前景裁剪时不需要固定 ROI）
        raw_roi = cfg.model.get("roi_size", None)
        roi_size = tuple(raw_roi) if raw_roi is not None else None

        transform = DINOTransform(
            num_base_patches=cfg.model.num_base_patches,
            global_views_size=tuple(cfg.model.global_views_size),
            local_views_size=tuple(cfg.model.local_views_size),
            local_views_scale=tuple(cfg.model.local_views_scale),
            num_local_views=cfg.model.num_local_views,
            roi_size=roi_size,
            spacing=tuple(cfg.model.spacing),
            use_foreground_crop=use_foreground_crop,
        )
        # 容错包装：损坏文件返回 None，不中断训练
        return SafeDINOTransform(transform)

    def build_dataloader(self):
        """构建数据加载器。

        支持三种配置方式：
          1. train.data_dirs（dict/list）：多数据源，使用 MultiSourceMRIDataset
          2. train.data_dir（str）：单数据源
          3. 同时存在时 data_dirs 优先

        性能优化：
          - cache_rate > 0 时使用 MONAI CacheDataset 缓存确定性预处理
          - prefetch_factor 控制 DataLoader 预加载批次数
        """
        cfg = self.cfg
        transform = self._build_transform()

        # 判断数据源类型
        data_dirs = cfg.train.get("data_dirs", None)
        sampler = None

        # 是否使用缓存数据集
        cache_rate = cfg.train.get("cache_rate", 0.0)
        cache_num_workers = cfg.train.get("cache_num_workers", 4)

        if data_dirs is not None:
            # --- 多数据源模式 ---
            if hasattr(data_dirs, "items"):
                data_dirs_dict = dict(data_dirs)
            elif isinstance(data_dirs, (list, tuple)):
                data_dirs_dict = {
                    os.path.basename(d.rstrip("/")): d for d in data_dirs
                }
            else:
                raise ValueError(
                    f"train.data_dirs 格式错误: {type(data_dirs)}，"
                    f"期望 dict 或 list"
                )

            multi_dataset = MultiSourceMRIDataset(
                data_dirs=data_dirs_dict,
                transform=transform,
                fraction=cfg.train.data_fraction,
                cache_rate=cache_rate,
                cache_num_workers=cache_num_workers,
            )
            self.print(multi_dataset.summary())
            dataset = multi_dataset.dataset

            # 区域均衡采样
            balance = cfg.train.get("balance_regions", False)
            if balance:
                region_weights = cfg.train.get("region_weights", None)
                if region_weights is not None:
                    region_weights = dict(region_weights)
                sampler = multi_dataset.get_balanced_sampler(
                    weights=region_weights,
                )
                self.print(
                    f"启用区域均衡采样，权重: "
                    f"{region_weights or '按数据量倒数均衡'}"
                )
        else:
            # --- 单数据源模式 ---
            data_dir = cfg.train.data_dir
            sites = cfg.train.get("sites", None)

            if cache_rate > 0:
                from monai.data import CacheDataset
                from src.data.mri_dataset import _discover_nifti_files, _discover_ixi_files

                if sites is not None:
                    data_list = _discover_ixi_files(data_dir, sites=list(sites),
                                                     fraction=cfg.train.data_fraction)
                else:
                    data_list = _discover_nifti_files(data_dir,
                                                       fraction=cfg.train.data_fraction)

                dataset = CacheDataset(
                    data=data_list,
                    transform=transform,
                    cache_rate=cache_rate,
                    num_workers=cache_num_workers,
                )
                self.print(
                    f"CacheDataset: cache_rate={cache_rate}, "
                    f"workers={cache_num_workers}"
                )
            else:
                if sites is not None:
                    dataset = IXIDataset(
                        data_dir=data_dir,
                        transform=transform,
                        sites=list(sites),
                        fraction=cfg.train.data_fraction,
                    )
                else:
                    dataset = MRIDataset(
                        data_dir=data_dir,
                        transform=transform,
                        fraction=cfg.train.data_fraction,
                    )

        self.print(f"Dataset: {len(dataset)} volumes")

        # DataLoader 预取配置
        prefetch = cfg.train.get("prefetch_factor", 2)
        num_workers = cfg.train.num_workers

        data_loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=cfg.train.pin_memory,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=cfg.train.drop_last,
            persistent_workers=cfg.train.persistent_workers and num_workers > 0,
            collate_fn=collate_dino,
            prefetch_factor=prefetch if num_workers > 0 else None,
        )
        return data_loader

    def build_model(self):
        cfg = self.cfg
        arch_name = cfg.model.architecture
        if arch_name not in _ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture: {arch_name}. "
                f"Available: {list(_ARCHITECTURES.keys())}"
            )

        # DINOv3 特性：register tokens + RoPE 坐标增强
        backbone_kwargs = dict(num_classes=0, dynamic_img_size=True)

        reg_tokens = cfg.model.get("reg_tokens", 0)
        if reg_tokens > 0:
            backbone_kwargs["reg_tokens"] = reg_tokens
            self.print(f"Using {reg_tokens} register tokens (DINOv3)")

        # RoPE 坐标增强（DINOv3 风格）
        rope_kwargs = {}
        if cfg.model.get("rope_shift_coords") is not None:
            rope_kwargs["shift_coords"] = cfg.model.rope_shift_coords
        if cfg.model.get("rope_jitter_coords") is not None:
            rope_kwargs["jitter_coords"] = cfg.model.rope_jitter_coords
        if cfg.model.get("rope_rescale_coords") is not None:
            rope_kwargs["rescale_coords"] = cfg.model.rope_rescale_coords
        if rope_kwargs:
            backbone_kwargs["pos_embed"] = "rope"
            backbone_kwargs["rope_kwargs"] = rope_kwargs
            self.print(f"RoPE augmentations: {rope_kwargs}")

        backbone = _ARCHITECTURES[arch_name](**backbone_kwargs)
        embed_dim = backbone.embed_dim
        self.print(f"Backbone: {arch_name}, embed_dim={embed_dim}")

        model = DINO(
            backbone,
            input_dim=embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            bottleneck_dim=cfg.model.bottleneck_dim,
            output_dim=cfg.model.output_dim,
            freeze_last_layer=cfg.model.freeze_last_layer,
        )

        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.print(f"Params: {n_total:,} total, {n_train:,} trainable")

        # --- 性能优化：torch.compile（PyTorch 2.0+） ---
        use_compile = cfg.train.get("use_compile", False)
        if use_compile:
            if hasattr(torch, "compile"):
                self.print("Applying torch.compile to student backbone...")
                model.backbone_student = torch.compile(model.backbone_student)
                self.print("torch.compile: enabled (first few steps will be slow)")
            else:
                self.print("torch.compile: not available (need PyTorch >= 2.0)")

        return model

    def build_optimizer(self, model):
        cfg = self.cfg
        param_groups = get_param_groups_with_decay(
            model,
            llrd_factor=cfg.optim.llrd_factor,
            patch_embed_lr_mult=cfg.optim.patch_embed_lr_mult,
            projection_head_wd_mult=cfg.optim.projection_head_wd_mult,
        )
        optimizer = AdamW(
            param_groups,
            lr=cfg.optim.lr,
            betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
        )
        return optimizer

    def _cleanup_old_checkpoints(self, output_dir, max_keep=3):
        """滚动清理旧的带编号 checkpoint，只保留最新的 max_keep 个。"""
        import glob as _glob
        pattern = os.path.join(output_dir, "checkpoint_epoch=*.pt")
        ckpts = sorted(_glob.glob(pattern))

        if len(ckpts) <= max_keep:
            return

        to_delete = ckpts[:-max_keep]
        for old_ckpt in to_delete:
            try:
                os.remove(old_ckpt)
                self.print(f"  Deleted old checkpoint: {os.path.basename(old_ckpt)}")
            except OSError:
                pass

    def _check_disk_space(self, path, min_free_gb=50):
        """检查磁盘剩余空间，低于阈值时打印警告。"""
        try:
            stat = os.statvfs(path)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            if free_gb < min_free_gb:
                self.print(
                    f"  WARNING: Disk space low! "
                    f"{free_gb:.1f} GB free < {min_free_gb} GB threshold. "
                    f"Consider reducing saveckp_freq or max_keep_ckpts."
                )
        except Exception:
            pass

    def train(self):
        cfg = self.cfg
        os.makedirs(cfg.train.output_dir, exist_ok=True)

        # 构建训练组件
        data_loader = self.build_dataloader()
        model = self.build_model()
        criterion = DINOLoss(
            output_dim=cfg.model.output_dim,
            warmup_teacher_temp=cfg.model.warmup_teacher_temp,
            teacher_temp=cfg.model.teacher_temp,
            warmup_teacher_temp_epochs=cfg.model.warmup_teacher_temp_epochs,
            student_temp=cfg.model.student_temp,
            center_momentum=cfg.model.center_momentum,
        )
        optimizer = self.build_optimizer(model)

        # 准备分布式训练相关封装
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model, data_loader, criterion, optimizer = self.accelerator.prepare(
            model, data_loader, criterion, optimizer,
        )
        unwrapped = self.accelerator.unwrap_model(model)
        gram_teacher = None

        # DINOv3：Gram Anchoring 损失
        gram_cfg = cfg.get("gram", {})
        use_gram = gram_cfg.get("enabled", False)
        if use_gram:
            gram_criterion = GramLoss(
                apply_norm=gram_cfg.get("apply_norm", True),
                img_level=gram_cfg.get("img_level", True),
                remove_neg=gram_cfg.get("remove_neg", False),
                remove_only_teacher_neg=gram_cfg.get("remove_only_teacher_neg", False),
            )
            gram_loss_weight = gram_cfg.get("loss_weight", 1.0)
            gram_update_freq = gram_cfg.get("update_freq", 10000)
            gram_first_update = gram_cfg.get("first_update_step", 0)
            gram_max_updates = gram_cfg.get("max_updates", None)
            from copy import deepcopy
            gram_teacher = deepcopy(
                model.backbone_teacher if hasattr(model, "backbone_teacher")
                else unwrapped.backbone_teacher
            )
            gram_teacher.eval()
            gram_teacher = gram_teacher.to(self.accelerator.device)
            for p in gram_teacher.parameters():
                p.requires_grad = False
            from src.utils.modeling import deactivate_requires_grad_and_to_eval
            deactivate_requires_grad_and_to_eval(gram_teacher)
            gram_update_count = 0
            gram_initialized = False
            self.print(f"Gram Anchoring: weight={gram_loss_weight}, update_freq={gram_update_freq}")
        else:
            gram_criterion = None
            self.print("Gram Anchoring: disabled")

        # 从 checkpoint 恢复
        start_epoch = 0
        if cfg.train.resume_ckp:
            ckpt_path = os.path.join(cfg.train.output_dir, "checkpoint.pt")
            if os.path.exists(ckpt_path):
                start_epoch = load_state(
                    ckpt_path, model=unwrapped, optimizer=optimizer, criterion=criterion,
                )
                self.print(f"Resumed from epoch {start_epoch}")

        total_steps = cfg.optim.epochs * len(data_loader)
        warmup_steps = cfg.optim.warmup_epochs * len(data_loader)
        self.print(
            f"Training: {cfg.optim.epochs} epochs, "
            f"{len(data_loader)} steps/epoch, "
            f"{total_steps} total"
        )

        # --- 训练主循环 ---
        global_step = start_epoch * len(data_loader)
        t0 = time.time()

        for epoch in range(start_epoch, cfg.optim.epochs):
            if hasattr(data_loader, "set_epoch"):
                data_loader.set_epoch(epoch)

            for batch in data_loader:
                # 容错：collate_dino 过滤掉所有 None 后可能返回空字典
                if not batch:
                    continue

                with self.accelerator.accumulate(model):
                    # 调度学习率、权重衰减和动量
                    lr = cosine_warmup_schedule(
                        global_step, max_steps=total_steps,
                        start_value=cfg.optim.lr, end_value=cfg.optim.min_lr,
                        warmup_steps=warmup_steps, warmup_start_value=0.0,
                    )
                    wd = cosine_schedule(
                        global_step, total_steps,
                        cfg.optim.weight_decay, cfg.optim.weight_decay_end,
                    )
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr * pg.get("lr_mult", 1.0)
                        pg["weight_decay"] = wd * pg.get("wd_mult", 1.0)

                    mom = cosine_schedule(
                        global_step, total_steps,
                        cfg.model.momentum_teacher, cfg.model.momentum_teacher_end,
                    )
                    update_momentum(unwrapped.backbone_student, unwrapped.backbone_teacher, mom)
                    update_momentum(unwrapped.head_student, unwrapped.head_teacher, mom)

                    # 前向计算
                    gv = batch["global_views"]
                    lv = batch["local_views"]
                    teacher_out, student_out = model(global_views=gv, local_views=lv)

                    n_gv = gv.shape[0]
                    n_total = n_gv + lv.shape[0]
                    dino_loss = criterion(
                        teacher_out=teacher_out.chunk(n_gv, dim=0),
                        student_out=student_out.chunk(n_total, dim=0),
                        epoch=epoch,
                    )

                    # DINOv3：计算 Gram Anchoring 损失
                    loss = dino_loss
                    gram_loss_val = 0.0
                    if use_gram:
                        if global_step >= gram_first_update:
                            do_gram_update = False
                            if not gram_initialized:
                                do_gram_update = True
                                gram_initialized = True
                            elif (global_step - gram_first_update) % gram_update_freq == 0:
                                if gram_max_updates is None or gram_update_count < gram_max_updates:
                                    do_gram_update = True
                            if do_gram_update:
                                for gt_p, t_p in zip(gram_teacher.parameters(), unwrapped.backbone_teacher.parameters()):
                                    gt_p.data.copy_(t_p.data)
                                gram_teacher.eval()
                                gram_update_count += 1
                                self.print(f"  Gram Teacher updated (#{gram_update_count}) at step {global_step}")

                        if gram_initialized:
                            with torch.no_grad():
                                student_features = unwrapped.backbone_student(gv).flatten(start_dim=1)
                            student_feat_full = unwrapped.backbone_student.forward_features(gv)
                            n_prefix = unwrapped.backbone_student.num_prefix_tokens
                            student_patches = student_feat_full[:, n_prefix:]

                            with torch.no_grad():
                                gram_feat_full = gram_teacher.forward_features(gv)
                                gram_patches = gram_feat_full[:, n_prefix:]

                            g_loss = gram_criterion(student_patches, gram_patches, img_level=True)
                            loss = loss + gram_loss_weight * g_loss
                            gram_loss_val = g_loss.item()

                    # 反向传播
                    self.accelerator.backward(loss)

                    if cfg.optim.clip_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            chain(
                                unwrapped.backbone_student.parameters(),
                                unwrapped.head_student.parameters(),
                            ),
                            cfg.optim.clip_grad_norm,
                        )
                    unwrapped.head_student.cancel_last_layer_gradients(epoch)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # 优化：比 zero_grad() 更快

                    # 记录日志
                    dt = time.time() - t0
                    t0 = time.time()
                    if global_step % cfg.train.log_freq == 0:
                        gram_str = f" gram={gram_loss_val:.4f}" if use_gram else ""
                        self.print(
                            f"E{epoch+1}/{cfg.optim.epochs} "
                            f"S{global_step+1}/{total_steps} "
                            f"loss={loss.item():.4f} "
                            f"dino={dino_loss.item():.4f}{gram_str} "
                            f"lr={lr:.2e} wd={wd:.4f} mom={mom:.4f} "
                            f"t={dt:.2f}s"
                        )
                        log_dict = {
                            "loss": loss.item(), "dino_loss": dino_loss.item(),
                            "epoch": epoch, "lr": lr,
                            "weight_decay": wd, "momentum": mom, "step_time": dt,
                        }
                        if use_gram:
                            log_dict["gram_loss"] = gram_loss_val
                        self.accelerator.log(log_dict, step=global_step)

                    global_step += 1

            # 保存 checkpoint
            save_state(
                os.path.join(cfg.train.output_dir, "checkpoint.pt"),
                epoch=epoch + 1,
                model=unwrapped, optimizer=optimizer, criterion=criterion,
            )
            if (epoch + 1) % cfg.train.saveckp_freq == 0:
                save_state(
                    os.path.join(cfg.train.output_dir, f"checkpoint_epoch={epoch+1:04d}.pt"),
                    epoch=epoch + 1,
                    model=unwrapped, optimizer=optimizer, criterion=criterion,
                )
                max_keep = cfg.train.get("max_keep_ckpts", 3)
                self._cleanup_old_checkpoints(cfg.train.output_dir, max_keep)

            min_free_gb = cfg.train.get("min_free_disk_gb", 50)
            self._check_disk_space(cfg.train.output_dir, min_free_gb)

            self.accelerator.wait_for_everyone()
            self.print(f"Epoch {epoch+1} done.")

        self.accelerator.end_training()
        self.print("Training complete!")