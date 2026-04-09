"""
DINOv3 MRI 预训练 Trainer。

从 DINO 升级到 DINOv3 = DINOv2 + Gram Anchoring：
  - iBOT patch masking + patch-level loss
  - KoLeo 正则化
  - Gram Anchoring（防止 patch 特征退化）
  - Register tokens + RoPE coord augmentations

基于 SPECTRE pretrain_dinov2_v3.py 的训练循环，
结合原 trainer.py 的 MRI 数据加载和配置管理。
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
from src.models import DINOv3, DINOLoss, iBOTPatchLoss, KoLeoLoss, GramLoss
from src.models import vit_base_patch16_96, vit_base_rope_patch16_96, vit_small_patch16_96
from src.utils import (
    cosine_schedule,
    cosine_warmup_schedule,
    load_state,
    save_state,
    update_momentum,
)
from src.utils.masking import random_block_mask
from src.utils.misc import fix_random_seeds
from src.utils.scheduler import linear_warmup_schedule
from src.utils.param_groups import get_param_groups_with_decay


# ViT 架构注册表
_ARCHITECTURES = {
    "vit_small_patch16_96": vit_small_patch16_96,
    "vit_base_patch16_96": vit_base_patch16_96,
    "vit_base_rope_patch16_96": vit_base_rope_patch16_96,
}


class TrainerV3:
    """DINOv3 预训练器。

    完整训练流程：
      1. DINOv2 基础：DINO CLS loss + iBOT patch loss + KoLeo 正则化
      2. DINOv3 扩展：Gram Anchoring loss（内置于 DINOv3 模型中）
      3. MRI 适配：容错数据加载、混合精度、多源数据集

    相比原 trainer.py 的改进：
      - 使用 DINOv3 类（而非 DINO + 手动 Gram 计算）
      - 修复 DINO loss 的 chunk 逻辑（chunk(2) 而非 chunk(n_gv)）
      - 消除 Gram loss 中的冗余前向传播
      - 增加 iBOT masking + KoLeo loss
    """

    def __init__(self, cfg):
        self.cfg = cfg
        fix_random_seeds(cfg.train.seed)

        torch.backends.cudnn.benchmark = True

        mixed_precision = cfg.train.get("mixed_precision", "fp16")

        dataloader_config = DataLoaderConfiguration(
            non_blocking=cfg.train.pin_memory,
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train.grad_accum_steps,
            log_with="wandb" if cfg.train.log_wandb else None,
            dataloader_config=dataloader_config,
            mixed_precision=mixed_precision,
        )

        if mixed_precision != "no":
            self.accelerator.print(
                f"Mixed precision: {mixed_precision}"
            )

        if cfg.train.log_wandb:
            self.accelerator.init_trackers(
                project_name="dinov3-mri",
                config={k: v for d in cfg.values() for k, v in d.items()},
                init_kwargs={"wandb": {"dir": os.path.join(cfg.train.output_dir, "logs")}},
            )

        self.print = self.accelerator.print

    # ------------------------------------------------------------------
    # 数据加载（复用原 trainer.py，无需修改）
    # ------------------------------------------------------------------

    def _build_transform(self):
        cfg = self.cfg
        use_foreground_crop = cfg.model.get("use_foreground_crop", False)
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
        return SafeDINOTransform(transform)

    def build_dataloader(self):
        cfg = self.cfg
        transform = self._build_transform()

        data_dirs = cfg.train.get("data_dirs", None)
        sampler = None
        cache_rate = cfg.train.get("cache_rate", 0.0)
        cache_num_workers = cfg.train.get("cache_num_workers", 4)

        if data_dirs is not None:
            if hasattr(data_dirs, "items"):
                data_dirs_dict = dict(data_dirs)
            elif isinstance(data_dirs, (list, tuple)):
                data_dirs_dict = {
                    os.path.basename(d.rstrip("/")): d for d in data_dirs
                }
            else:
                raise ValueError(f"train.data_dirs 格式错误: {type(data_dirs)}")

            multi_dataset = MultiSourceMRIDataset(
                data_dirs=data_dirs_dict,
                transform=transform,
                fraction=cfg.train.data_fraction,
                cache_rate=cache_rate,
                cache_num_workers=cache_num_workers,
            )
            self.print(multi_dataset.summary())
            dataset = multi_dataset.dataset

            balance = cfg.train.get("balance_regions", False)
            if balance:
                region_weights = cfg.train.get("region_weights", None)
                if region_weights is not None:
                    region_weights = dict(region_weights)
                sampler = multi_dataset.get_balanced_sampler(weights=region_weights)
        else:
            data_dir = cfg.train.data_dir
            sites = cfg.train.get("sites", None)

            if cache_rate > 0:
                from monai.data import CacheDataset
                from src.data.mri_dataset import _discover_nifti_files, _discover_ixi_files

                if sites is not None:
                    data_list = _discover_ixi_files(
                        data_dir, sites=list(sites), fraction=cfg.train.data_fraction
                    )
                else:
                    data_list = _discover_nifti_files(
                        data_dir, fraction=cfg.train.data_fraction
                    )

                dataset = CacheDataset(
                    data=data_list, transform=transform,
                    cache_rate=cache_rate, num_workers=cache_num_workers,
                )
            else:
                if sites is not None:
                    dataset = IXIDataset(
                        data_dir=data_dir, transform=transform,
                        sites=list(sites), fraction=cfg.train.data_fraction,
                    )
                else:
                    dataset = MRIDataset(
                        data_dir=data_dir, transform=transform,
                        fraction=cfg.train.data_fraction,
                    )

        self.print(f"Dataset: {len(dataset)} volumes")

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

    # ------------------------------------------------------------------
    # 模型构建：使用 DINOv3 类
    # ------------------------------------------------------------------

    def build_model(self):
        cfg = self.cfg
        arch_name = cfg.model.architecture
        if arch_name not in _ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture: {arch_name}. "
                f"Available: {list(_ARCHITECTURES.keys())}"
            )

        backbone_kwargs = dict(num_classes=0, dynamic_img_size=True)

        reg_tokens = cfg.model.get("reg_tokens", 0)
        if reg_tokens > 0:
            backbone_kwargs["reg_tokens"] = reg_tokens
            self.print(f"Register tokens: {reg_tokens}")

        # RoPE 坐标增强
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

        # Layer scale (DINOv2/v3 推荐)
        init_values = cfg.model.get("layer_scale_init_value", None)
        if init_values is not None:
            backbone_kwargs["init_values"] = init_values

        backbone = _ARCHITECTURES[arch_name](**backbone_kwargs)
        embed_dim = backbone.embed_dim
        self.print(f"Backbone: {arch_name}, embed_dim={embed_dim}")

        # Gram 配置
        gram_cfg = cfg.get("gram", {})
        use_gram = gram_cfg.get("enabled", True)

        model = DINOv3(
            backbone,
            input_dim=embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            bottleneck_dim=cfg.model.bottleneck_dim,
            output_dim=cfg.model.output_dim,
            ibot_seperate_head=cfg.model.get("ibot_seperate_head", False),
            student_drop_path_rate=cfg.model.get("student_drop_path_rate", 0.1),
            freeze_last_layer=cfg.model.freeze_last_layer,
            gram_update_freq=gram_cfg.get("update_freq", 10000),
            gram_first_update_step=gram_cfg.get("first_update_step", 0),
            gram_max_updates=gram_cfg.get("max_updates", None),
        )

        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.print(f"DINOv3 model: {n_total:,} total, {n_train:,} trainable")
        self.print(f"  iBOT separate head: {cfg.model.get('ibot_seperate_head', False)}")
        self.print(f"  Student drop path: {cfg.model.get('student_drop_path_rate', 0.1)}")
        self.print(f"  Gram Anchoring: {'enabled' if use_gram else 'disabled'}")

        # torch.compile（可选）
        use_compile = cfg.train.get("use_compile", False)
        if use_compile and hasattr(torch, "compile"):
            self.print("Applying torch.compile to student backbone...")
            model.backbone_student = torch.compile(model.backbone_student)

        return model

    def build_optimizer(self, model):
        cfg = self.cfg
        param_groups = get_param_groups_with_decay(
            model,
            llrd_factor=cfg.optim.llrd_factor,
            patch_embed_lr_mult=cfg.optim.patch_embed_lr_mult,
            projection_head_wd_mult=cfg.optim.projection_head_wd_mult,
        )
        return AdamW(
            param_groups,
            lr=cfg.optim.lr,
            betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
        )

    def _cleanup_old_checkpoints(self, output_dir, max_keep=3):
        import glob as _glob
        pattern = os.path.join(output_dir, "checkpoint_epoch=*.pt")
        ckpts = sorted(_glob.glob(pattern))
        if len(ckpts) <= max_keep:
            return
        for old_ckpt in ckpts[:-max_keep]:
            try:
                os.remove(old_ckpt)
                self.print(f"  Deleted old checkpoint: {os.path.basename(old_ckpt)}")
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        os.makedirs(cfg.train.output_dir, exist_ok=True)

        # ---- 构建组件 ----
        data_loader = self.build_dataloader()
        model = self.build_model()

        # Loss 组件
        criterion_dino = DINOLoss(
            output_dim=cfg.model.output_dim,
            warmup_teacher_temp=cfg.model.warmup_teacher_temp,
            teacher_temp=cfg.model.teacher_temp,
            warmup_teacher_temp_epochs=cfg.model.warmup_teacher_temp_epochs,
            student_temp=cfg.model.student_temp,
            center_momentum=cfg.model.center_momentum,
        )
        criterion_ibot = iBOTPatchLoss(
            output_dim=cfg.model.output_dim,
            teacher_temp=cfg.model.teacher_temp,
            student_temp=cfg.model.student_temp,
            center_momentum=cfg.model.center_momentum,
        )
        criterion_koleo = KoLeoLoss()

        # Gram loss
        gram_cfg = cfg.get("gram", {})
        use_gram = gram_cfg.get("enabled", True)
        if use_gram:
            gram_criterion = GramLoss(
                apply_norm=gram_cfg.get("apply_norm", True),
                img_level=gram_cfg.get("img_level", True),
                remove_neg=gram_cfg.get("remove_neg", False),
                remove_only_teacher_neg=gram_cfg.get("remove_only_teacher_neg", False),
            )
            gram_loss_weight = gram_cfg.get("loss_weight", 1.0)
            self.print(f"Gram loss weight: {gram_loss_weight}")
        else:
            gram_criterion = None
            gram_loss_weight = 0.0

        # Loss 权重
        dino_loss_weight = cfg.optim.get("dino_loss_weight", 1.0)
        ibot_loss_weight = cfg.optim.get("ibot_loss_weight", 1.0)
        koleo_loss_weight = cfg.optim.get("koleo_loss_weight", 0.1)

        optimizer = self.build_optimizer(model)

        # ---- Accelerate 准备 ----
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model, data_loader, criterion_dino, criterion_ibot, criterion_koleo, \
            optimizer = self.accelerator.prepare(
                model, data_loader, criterion_dino,
                criterion_ibot, criterion_koleo, optimizer,
            )
        unwrapped = self.accelerator.unwrap_model(model)

        # ---- Checkpoint 恢复 ----
        start_epoch = 0
        if cfg.train.resume_ckp:
            ckpt_path = os.path.join(cfg.train.output_dir, "checkpoint.pt")
            if os.path.exists(ckpt_path):
                start_epoch = load_state(
                    ckpt_path,
                    model=unwrapped,
                    optimizer=optimizer,
                    criterion_dino=criterion_dino,
                    criterion_ibot=criterion_ibot,
                    criterion_koleo=criterion_koleo,
                )
                self.print(f"Resumed from epoch {start_epoch}")

        total_steps = cfg.optim.epochs * len(data_loader)
        warmup_steps = cfg.optim.warmup_epochs * len(data_loader)
        warmup_teacher_temp_steps = cfg.model.warmup_teacher_temp_epochs * len(data_loader)
        num_local_views = cfg.model.num_local_views

        self.print(
            f"Training: {cfg.optim.epochs} epochs, "
            f"{len(data_loader)} steps/epoch, "
            f"{total_steps} total"
        )
        self.print(
            f"Loss weights: dino={dino_loss_weight}, "
            f"ibot={ibot_loss_weight}, koleo={koleo_loss_weight}, "
            f"gram={gram_loss_weight if use_gram else 'disabled'}"
        )

        # ---- 训练主循环 ----
        global_step = start_epoch * len(data_loader)
        t0 = time.time()

        for epoch in range(start_epoch, cfg.optim.epochs):
            if hasattr(data_loader, "set_epoch"):
                data_loader.set_epoch(epoch)

            for batch in data_loader:
                if not batch:
                    continue

                with self.accelerator.accumulate(model):
                    # ---- 调度 LR / WD / Momentum ----
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
                    # DINOv3 的 EMA 更新目标是 MaskedVisionTransformer 内部的 vit
                    update_momentum(
                        unwrapped.backbone_student.vit,
                        unwrapped.backbone_teacher.vit,
                        mom,
                    )
                    update_momentum(
                        unwrapped.head_student_dino,
                        unwrapped.head_teacher_dino,
                        mom,
                    )
                    if cfg.model.get("ibot_seperate_head", False):
                        update_momentum(
                            unwrapped.head_student_ibot,
                            unwrapped.head_teacher_ibot,
                            mom,
                        )

                    # ---- Gram Teacher 更新 ----
                    if use_gram:
                        updated = unwrapped.maybe_update_gram_teacher(global_step)
                        if updated:
                            self.print(
                                f"  Gram Teacher updated "
                                f"(#{unwrapped._gram_update_count}) "
                                f"at step {global_step}"
                            )

                    # ---- 构造 iBOT mask ----
                    global_views = batch["global_views"]
                    local_views = batch["local_views"]

                    B = global_views.shape[0]
                    sequence_length = unwrapped.backbone_teacher.sequence_length
                    num_prefix_tokens = unwrapped.backbone_teacher.vit.num_prefix_tokens
                    mask = global_views.new_zeros(
                        (B, sequence_length), dtype=torch.bool
                    )

                    H, W, D = unwrapped.backbone_teacher.vit.patch_embed.grid_size
                    assert H * W * D == sequence_length - num_prefix_tokens, \
                        "Grid size does not match sequence length."
                    block_mask = random_block_mask(
                        size=(B, H, W, D),
                        batch_mask_ratio=cfg.model.get("mask_probability", 0.5),
                        min_image_mask_ratio=cfg.model.get("mask_ratio_min", 0.2),
                        max_image_mask_ratio=cfg.model.get("mask_ratio_max", 0.7),
                        device=mask.device,
                    )
                    mask[:, num_prefix_tokens:] = block_mask.flatten(start_dim=1)

                    # ---- Forward（一次调用获取所有输出）----
                    outputs = model(
                        global_views=global_views,
                        local_views=local_views,
                        mask=mask,
                    )
                    teacher_cls_out = outputs["teacher_cls_out"]
                    teacher_masked_out = outputs["teacher_masked_out"]
                    student_cls_out = outputs["student_cls_out"]
                    student_masked_out = outputs["student_masked_out"]

                    # ---- 计算 Loss ----

                    # Teacher temperature warmup（按 step 而非 epoch）
                    teacher_temp = linear_warmup_schedule(
                        step=global_step,
                        warmup_steps=warmup_teacher_temp_steps,
                        start_value=cfg.model.warmup_teacher_temp,
                        end_value=cfg.model.teacher_temp,
                    )

                    # DINO CLS loss
                    # 关键修复：chunk(2) 分离 2 个 global views，
                    # chunk(2 + num_local_views) 分离所有 views
                    dino_loss = criterion_dino(
                        teacher_out=teacher_cls_out.chunk(2, dim=0),
                        student_out=student_cls_out.chunk(
                            2 + num_local_views, dim=0
                        ),
                        teacher_temp=teacher_temp,
                    )

                    # iBOT patch loss
                    ibot_loss = criterion_ibot(
                        teacher_out=teacher_masked_out,
                        student_out=student_masked_out,
                        mask=block_mask,
                        teacher_temp=teacher_temp,
                    )

                    # KoLeo loss（仅对 local views 计算）
                    student_views_split = student_cls_out.chunk(
                        2 + num_local_views, dim=0
                    )
                    koleo_loss = sum(
                        criterion_koleo(p)
                        for p in student_views_split[2:]  # 跳过 2 个 global views
                    )

                    # Gram Anchoring loss
                    gram_loss_val = 0.0
                    if use_gram and unwrapped._gram_initialized:
                        student_patches = outputs["student_patches"]
                        gram_teacher_patches = outputs["gram_teacher_patches"]
                        gram_loss = gram_criterion(
                            student_patches, gram_teacher_patches, img_level=True
                        )
                        gram_loss_val = gram_loss.item()
                    else:
                        gram_loss = 0.0

                    # 总 loss
                    loss = (
                        dino_loss_weight * dino_loss
                        + ibot_loss_weight * ibot_loss
                        + koleo_loss_weight * koleo_loss
                        + gram_loss_weight * gram_loss
                    )

                    # ---- 反向传播 ----
                    self.accelerator.backward(loss)

                    # 梯度裁剪
                    student_heads = [unwrapped.head_student_dino]
                    if cfg.model.get("ibot_seperate_head", False):
                        student_heads.append(unwrapped.head_student_ibot)

                    if cfg.optim.clip_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            chain(
                                unwrapped.backbone_student.parameters(),
                                *[h.parameters() for h in student_heads],
                            ),
                            cfg.optim.clip_grad_norm,
                        )

                    for head in student_heads:
                        head.cancel_last_layer_gradients(epoch)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # ---- 日志 ----
                    dt = time.time() - t0
                    t0 = time.time()
                    if global_step % cfg.train.log_freq == 0:
                        self.print(
                            f"E{epoch+1}/{cfg.optim.epochs} "
                            f"S{global_step+1}/{total_steps} "
                            f"loss={loss.item():.4f} "
                            f"dino={dino_loss.item():.4f} "
                            f"ibot={ibot_loss.item():.4f} "
                            f"koleo={koleo_loss.item() if hasattr(koleo_loss, 'item') else koleo_loss:.4f} "
                            f"gram={gram_loss_val:.4f} "
                            f"lr={lr:.2e} wd={wd:.4f} mom={mom:.4f} "
                            f"t={dt:.2f}s"
                        )
                        log_dict = {
                            "loss": loss.item(),
                            "dino_loss": dino_loss.item(),
                            "ibot_loss": ibot_loss.item(),
                            "koleo_loss": koleo_loss.item() if hasattr(koleo_loss, 'item') else koleo_loss,
                            "gram_loss": gram_loss_val,
                            "epoch": epoch,
                            "lr": lr,
                            "weight_decay": wd,
                            "momentum": mom,
                            "step_time": dt,
                        }
                        self.accelerator.log(log_dict, step=global_step)

                    global_step += 1

            # ---- Checkpoint 保存 ----
            save_state(
                os.path.join(cfg.train.output_dir, "checkpoint.pt"),
                epoch=epoch + 1,
                model=unwrapped,
                optimizer=optimizer,
                criterion_dino=criterion_dino,
                criterion_ibot=criterion_ibot,
                criterion_koleo=criterion_koleo,
            )
            if (epoch + 1) % cfg.train.saveckp_freq == 0:
                save_state(
                    os.path.join(
                        cfg.train.output_dir,
                        f"checkpoint_epoch={epoch+1:04d}.pt",
                    ),
                    epoch=epoch + 1,
                    model=unwrapped,
                    optimizer=optimizer,
                    criterion_dino=criterion_dino,
                    criterion_ibot=criterion_ibot,
                    criterion_koleo=criterion_koleo,
                )
                max_keep = cfg.train.get("max_keep_ckpts", 3)
                self._cleanup_old_checkpoints(cfg.train.output_dir, max_keep)

            self.accelerator.wait_for_everyone()
            self.print(f"Epoch {epoch+1} done.")

        self.accelerator.end_training()
        self.print("Training complete!")
