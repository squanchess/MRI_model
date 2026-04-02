"""
DINO Trainer for MRI pretraining.

Refactored from SPECTRE's pretrain_dino.py (MIT License):
  - Extracted training loop into Trainer class
  - Replaced multi-CT-dataset dataloader with direct IXIDataset construction
  - Kept all DINO training logic: cosine LR/WD/momentum, EMA, gradient clipping,
    cancel_last_layer_gradients, checkpoint save/resume
"""
import os
import time
from itertools import chain

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration

from src.models import vit_small_patch16_96, vit_base_patch16_96, vit_base_rope_patch16_96
from src.models import DINO, DINOLoss, GramLoss
from src.data import IXIDataset, DINOTransform
from src.data.collate import collate_dino
from src.utils import (
    update_momentum,
    save_state,
    load_state,
    cosine_schedule,
    cosine_warmup_schedule,
)
from src.utils.param_groups import get_param_groups_with_decay
from src.utils.misc import fix_random_seeds


# Registry of available ViT architectures
_ARCHITECTURES = {
    "vit_small_patch16_96": vit_small_patch16_96,
    "vit_base_patch16_96": vit_base_patch16_96,
    "vit_base_rope_patch16_96": vit_base_rope_patch16_96,
}


class Trainer:
    """DINO pretraining trainer.

    Handles the full training loop: model construction, data loading,
    optimizer setup, training iterations, checkpointing, and logging.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        fix_random_seeds(cfg.train.seed)

        dataloader_config = DataLoaderConfiguration(
            non_blocking=cfg.train.pin_memory,
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train.grad_accum_steps,
            log_with="wandb" if cfg.train.log_wandb else None,
            dataloader_config=dataloader_config,
        )

        if cfg.train.log_wandb:
            self.accelerator.init_trackers(
                project_name="dinov3-mri",
                config={k: v for d in cfg.values() for k, v in d.items()},
                init_kwargs={"wandb": {"dir": os.path.join(cfg.train.output_dir, "logs")}},
            )

        self.print = self.accelerator.print

    def build_dataloader(self):
        cfg = self.cfg

        transform = DINOTransform(
            num_base_patches=cfg.model.num_base_patches,
            global_views_size=tuple(cfg.model.global_views_size),
            local_views_size=tuple(cfg.model.local_views_size),
            local_views_scale=tuple(cfg.model.local_views_scale),
            num_local_views=cfg.model.num_local_views,
            roi_size=tuple(cfg.model.roi_size),
            spacing=tuple(cfg.model.spacing),
        )

        sites = cfg.train.get("sites", None)
        if sites is not None:
            sites = list(sites)

        dataset = IXIDataset(
            data_dir=cfg.train.data_dir,
            transform=transform,
            sites=sites,
            fraction=cfg.train.data_fraction,
        )
        self.print(f"Dataset: {len(dataset)} volumes")

        data_loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            shuffle=True,
            drop_last=cfg.train.drop_last,
            persistent_workers=cfg.train.persistent_workers and cfg.train.num_workers > 0,
            collate_fn=collate_dino,
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

        # DINOv3 features: register tokens + RoPE coord augmentations
        backbone_kwargs = dict(num_classes=0, dynamic_img_size=True)

        reg_tokens = cfg.model.get("reg_tokens", 0)
        if reg_tokens > 0:
            backbone_kwargs["reg_tokens"] = reg_tokens
            self.print(f"Using {reg_tokens} register tokens (DINOv3)")

        # RoPE coordinate augmentations (DINOv3 style)
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

    def train(self):
        cfg = self.cfg
        os.makedirs(cfg.train.output_dir, exist_ok=True)

        # Build
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

        # DINOv3: Gram Anchoring loss
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
            # Gram teacher: frozen snapshot of EMA teacher, refreshed periodically
            from copy import deepcopy
            gram_teacher = deepcopy(model.backbone_teacher if hasattr(model, 'backbone_teacher') else unwrapped.backbone_teacher)
            from src.utils.modeling import deactivate_requires_grad_and_to_eval
            deactivate_requires_grad_and_to_eval(gram_teacher)
            gram_update_count = 0
            gram_initialized = False
            self.print(f"Gram Anchoring: weight={gram_loss_weight}, update_freq={gram_update_freq}")
        else:
            gram_criterion = None
            self.print("Gram Anchoring: disabled")
        optimizer = self.build_optimizer(model)

        # Prepare distributed
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model, data_loader, criterion, optimizer = self.accelerator.prepare(
            model, data_loader, criterion, optimizer,
        )
        unwrapped = self.accelerator.unwrap_model(model)

        # Resume
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

        # --- Training loop ---
        global_step = start_epoch * len(data_loader)
        t0 = time.time()

        for epoch in range(start_epoch, cfg.optim.epochs):
            if hasattr(data_loader, "set_epoch"):
                data_loader.set_epoch(epoch)

            for batch in data_loader:
                with self.accelerator.accumulate(model):
                    # Schedule LR / WD / momentum
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

                    # Forward
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

                    # DINOv3: Gram Anchoring loss
                    loss = dino_loss
                    gram_loss_val = 0.0
                    if use_gram:
                        # Maybe update Gram teacher (periodic snapshot of EMA teacher)
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

                        # Compute Gram loss on student vs Gram teacher patch tokens
                        if gram_initialized:
                            # Student patch features (pre-head, from global views only)
                            with torch.no_grad():
                                student_features = unwrapped.backbone_student(gv).flatten(start_dim=1)
                            # Reshape: student_features is (B, embed_dim) after pooling
                            # We need pre-pooling patch tokens. Use forward_features instead:
                            student_feat_full = unwrapped.backbone_student.forward_features(gv)  # (B, seq_len, D)
                            n_prefix = unwrapped.backbone_student.num_prefix_tokens
                            student_patches = student_feat_full[:, n_prefix:]  # (B, P, D)

                            with torch.no_grad():
                                gram_feat_full = gram_teacher.forward_features(gv)  # (B, seq_len, D)
                                gram_patches = gram_feat_full[:, n_prefix:]  # (B, P, D)

                            g_loss = gram_criterion(student_patches, gram_patches, img_level=True)
                            loss = loss + gram_loss_weight * g_loss
                            gram_loss_val = g_loss.item()

                    # Backward
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
                    optimizer.zero_grad()

                    # Log
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

            # Checkpoint
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
            self.accelerator.wait_for_everyone()
            self.print(f"Epoch {epoch+1} done.")

        self.accelerator.end_training()
        self.print("Training complete!")
