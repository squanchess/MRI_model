# DINOv3-MRI

DINOv3 self-supervised pretraining for 3D brain MRI, adapted from [SPECTRE](https://github.com/cclaess/SPECTRE) (CVPR 2026).

SPECTRE is a CT foundation model using DINO + SigLIP. This repo strips it down to the DINO self-supervised stage and adapts it for **isotropic brain MRI** (IXI T1 dataset as a starting point).

## Quick start

```bash
# 1. Clone
git clone https://github.com/yourname/dinov3-mri.git
cd dinov3-mri

# 2. Install
pip install -e .
# or manually:
pip install torch timm monai nibabel omegaconf accelerate

# 3. Download IXI T1 data
bash scripts/download_ixi.sh
# → places ~600 NIfTI files in data/IXI-T1/

# 4. Train
python train.py --output_dir outputs/dino_ixi

# 5. Multi-GPU
NUM_GPUS=4 bash scripts/pretrain.sh
```

## Project structure

```
dinov3-mri/
│
├── configs/
│   └── pretrain.yaml          # Default hyperparameters
│
├── data/                      # ← YOUR DATA GOES HERE (git-ignored)
│   ├── .gitkeep
│   └── IXI-T1/                # Raw NIfTI files after download
│       ├── IXI002-Guys-0828-T1.nii.gz
│       └── ...
│
├── outputs/                   # ← TRAINING OUTPUTS (git-ignored)
│   ├── .gitkeep
│   └── dino_ixi/              # Checkpoints, logs, saved configs
│
├── scripts/
│   ├── download_ixi.sh        # Download IXI T1 dataset
│   └── pretrain.sh            # Launch training (single/multi-GPU)
│
├── src/
│   ├── models/
│   │   ├── vision_transformer.py   # 3D ViT with isotropic MRI patches
│   │   ├── dino.py                 # DINO + DINOv2 teacher-student
│   │   ├── dino_head.py            # Projection head
│   │   ├── losses.py               # DINOLoss, iBOT, KoLeo, Center
│   │   └── layers/                 # Attention, PatchEmbed, 3D RoPE
│   │
│   ├── data/
│   │   ├── mri_dataset.py          # IXI dataset (auto file discovery)
│   │   ├── transforms.py           # DINO multi-crop for MRI
│   │   ├── collate.py              # DataLoader collate function
│   │   └── preprocessing.py        # Offline preprocessing + stats
│   │
│   ├── engine/
│   │   ├── trainer.py              # Training loop (Accelerate)
│   │   └── evaluator.py            # kNN + linear probe evaluation
│   │
│   └── utils/
│       ├── config.py               # OmegaConf config management
│       ├── checkpoint.py           # Save/load with distributed RNG
│       ├── scheduler.py            # Cosine warmup schedules
│       ├── param_groups.py         # LLRD parameter groups
│       ├── modeling.py             # EMA, position embed resampling
│       └── misc.py                 # Seeds, Format enum, tuple helpers
│
├── train.py                   # Pretraining entry point
├── evaluate.py                # Evaluation entry point
├── pyproject.toml             # Dependencies & project metadata
├── .gitignore
├── .env.example               # Environment variable template
└── README.md
```

## Key differences from SPECTRE (CT → MRI)

| Aspect | SPECTRE (CT) | This repo (MRI) |
|--------|-------------|-----------------|
| Patch size | (16, 16, 8) anisotropic | (16, 16, 16) isotropic |
| Input size | (128, 128, 64) | (96, 96, 96) |
| Intensity normalization | HU window [-1000, 1000] | Percentile clip (0.5th–99.5th) |
| Voxel spacing | (0.5, 0.5, 1.0) mm | (1.0, 1.0, 1.0) mm |
| Augmentations | RandScaleIntensityRange | RandScaleIntensity + RandBiasField |
| Datasets | 8 CT datasets (~100K vols) | IXI T1 (~600 vols) |
| VLA stage | SigLIP with radiology reports | Not included (DINO only) |

## Configuration

All hyperparameters live in `configs/pretrain.yaml`. Override via CLI:

```bash
python train.py --output_dir outputs/debug \
    train.batch_size_per_gpu=4 \
    optim.epochs=10 \
    model.architecture=vit_base_patch16_96
```

Available architectures: `vit_small_patch16_96` (default, 22M params), `vit_base_patch16_96` (86M), `vit_base_rope_patch16_96` (86M, RoPE).

## Evaluation

After training, evaluate feature quality:

```bash
python evaluate.py \
    --checkpoint outputs/dino_ixi/checkpoint.pt \
    --data_dir data/IXI-T1 \
    --architecture vit_small_patch16_96
```

## Data organization

Place your data under the `data/` directory (git-ignored):

```bash
data/
├── IXI-T1/                # ← bash scripts/download_ixi.sh
│   ├── IXI002-Guys-0828-T1.nii.gz
│   └── ...
├── IXI-T1-preprocessed/   # ← optional: python -m src.data.preprocessing ...
│   └── ...
└── your-other-dataset/    # ← add more datasets as needed
```

To use a different data location, either:
- Symlink: `ln -s /your/data/path data/IXI-T1`
- Override via CLI: `train.data_dir=/abs/path/to/data`
- Set env var: `DATA_DIR=/abs/path bash scripts/pretrain.sh`

## License

Code: MIT (same as SPECTRE). Pretrained weights may carry additional dataset license restrictions.

## Citation

If you use this code, please cite SPECTRE:

```bibtex
@misc{claessens_scaling_2025,
  title={Scaling Self-Supervised and Cross-Modal Pretraining for Volumetric CT Transformers},
  author={Claessens, Cris and Viviers, Christiaan and D'Amicantonio, Giacomo and Bondarev, Egor and Sommen, Fons van der},
  year={2025},
  url={http://arxiv.org/abs/2511.17209},
}
```
