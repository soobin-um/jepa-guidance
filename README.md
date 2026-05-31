# Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion (ICML 2026)

Sol Park and [Soobin Um](https://soobin-um.github.io/)† († Corresponding author)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.24631-b31b1b.svg)](https://arxiv.org/abs/2605.24631)

This repository contains the official implementation of **"Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion"** (ICML 2026).

We propose a **world-centric** perspective on minority sampling, where rarity is defined against real-world priors rather than generator-induced densities. We introduce **JEPA guidance**, which steers diffusion trajectories toward semantically rare regions using a Joint-Embedding Predictive Architecture (JEPA) as a world model, with principled approximations that make guidance computationally practical.

## Setup

```bash
git clone https://github.com/soobin-um/jepa-guidance
cd jepa-guidance
conda env create -f environment.yaml
conda activate jepa-guidance
```

Then download the SDXL-Lightning UNet checkpoint:

- [`sdxl_lightning_4step_unet.safetensors`](https://huggingface.co/ByteDance/SDXL-Lightning/tree/main) → place in `ckpt/`

## Sampling

Run the example script to generate 5k MS-COCO samples with JEPA guidance:

```bash
bash scripts/run_sdxl-lt_jepa.sh
```

The script runs generation followed by automatic JEPA scoring of the output images.

**Key arguments:**

| Argument | Description | Default |
|---|---|---|
| `--use_jepa` | Enable JEPA guidance | flag |
| `--jepa_backbone` | Backbone: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `metaclip` | `dinov2_vits14` |
| `--jepa_eta` | Guidance strength | `0.5` |
| `--g_interval` | Apply guidance every N steps | `1` |
| `--g_start_t` | Timestep ratio to start guidance (1.0=start, 0.0=end) | `0.8` |
| `--rsvd_topk` | Top-k singular values for JEPA score | `9` |
| `--jg_schedule` | Gradient scaling: `variance` or `constant` | `constant` |

## Evaluation

JEPA scores are computed automatically after sampling. To run scoring separately:

```bash
python jepa_scoring.py --img_root <path/to/images> --device cuda
```

Results are saved to `metrics/<dirname>_jepa.csv`.

## Citation

```bibtex
@article{park2026beyond,
  title={Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion},
  author={Park, Sol and Um, Soobin},
  journal={arXiv preprint arXiv:2605.24631},
  year={2026}
}
```
