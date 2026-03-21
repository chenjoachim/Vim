<div align="center">
<h1>Vision Mamba </h1>
<h3>Efficient Visual Representation Learning with Bidirectional State Space Model</h3>

[Lianghui Zhu](https://github.com/Unrealluver)<sup>1</sup> \*,[Bencheng Liao](https://github.com/LegendBC)<sup>1</sup> \*,[Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>2</sup>, [Xinlong Wang](https://www.xloong.wang/)<sup>3</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1 :email:</sup>

<sup>1</sup>  Huazhong University of Science and Technology, <sup>2</sup>  Horizon Robotics,  <sup>3</sup> Beijing Academy of Artificial Intelligence

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.

ICML 2024 ([conference paper](https://icml.cc/virtual/2024/poster/33768)), ArXiv Preprint ([arXiv 2401.09417](https://arxiv.org/abs/2401.09417)), HuggingFace Page ([🤗 2401.09417](https://huggingface.co/papers/2401.09417))


</div>


#



### News
* **` May. 2nd, 2024`:** Vision Mamba (Vim) is accepted by ICML2024. 🎉 Conference page can be found [here](https://icml.cc/virtual/2024/paper_metadata_from_author/33768).

* **` Feb. 10th, 2024`:** We update Vim-tiny/small weights and training scripts. By placing the class token at middle, Vim achieves improved results. Further details can be found in code and our updated [arXiv](https://arxiv.org/abs/2401.09417).

* **` Jan. 18th, 2024`:** We released our paper on Arxiv. Code/Models are coming soon. Please stay tuned! ☕️


## Abstract
Recently the state space models (SSMs) with efficient hardware-aware designs, i.e., the Mamba deep learning model, have shown great potential for long sequence modeling. Meanwhile building efficient and generic vision backbones purely upon SSMs is an appealing direction. However, representing visual data is challenging for SSMs due to the position-sensitivity of visual data and the requirement of global context for visual understanding. In this paper, we show that the reliance on self-attention for visual representation learning is not necessary and propose a new generic vision backbone with bidirectional Mamba blocks (Vim), which marks the image sequences with position embeddings and compresses the visual representation with bidirectional state space models. On ImageNet classification, COCO object detection, and ADE20k semantic segmentation tasks, Vim achieves higher performance compared to well-established vision transformers like DeiT, while also demonstrating significantly improved computation & memory efficiency. For example, Vim is 2.8x faster than DeiT and saves 86.8% GPU memory when performing batch inference to extract features on images with a resolution of 1248x1248. The results demonstrate that Vim is capable of overcoming the computation & memory constraints on performing Transformer-style understanding for high-resolution images and it has great potential to be the next-generation backbone for vision foundation models.


<div align="center">
<img src="assets/vim_teaser_v1.7.png" />
</div>

## Overview
<div align="center">
<img src="assets/vim_pipeline_v1.9.png" />
</div>

## Envs. for Pretraining

- NVIDIA GPUs:
  - Python 3.10.13

    ```bash
    uv venv .venv --python 3.10.13
    source .venv/bin/activate
    export CUDA_HOME=/path/to/cuda  # e.g. /usr/local/cuda-12.1
    ```

  - torch 2.1.1 + cu121
    - `uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121`

- AMD GPUs:
  - A [Docker image](https://hub.docker.com/r/rocm/pytorch/tags) is recommended, such as `rocm/pytorch:rocm6.2_ubuntu20.04_py3.9_pytorch_release_2.1.2`. For step-by-step instructions, please refer to [Vision Mamba on AMD GPU with ROCm](https://rocm.blogs.amd.com/artificial-intelligence/vision-mamba/README.html#vision-mamba).


- Requirements: vim_requirements.txt
  - `uv pip install -r vim/vim_requirements.txt`

- Install ``causal_conv1d`` and ``mamba``
  - `uv pip install --no-build-isolation -e causal-conv1d`
  - `uv pip install --no-build-isolation -e mamba-1p1p1`
  
  


## Train Your Vim

`bash vim/scripts/pt-vim-t.sh`

## Train Your Vim at Finer Granularity
`bash vim/scripts/ft-vim-t.sh`

## Model Weights

| Model | #param. | Top-1 Acc. | Top-5 Acc. | Hugginface Repo |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| [Vim-tiny](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   76.1   | 93.0 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-tiny<sup>+</sup>](https://huggingface.co/hustvl/Vim-tiny-midclstok)    |       7M       |   78.3   | 94.2 | https://huggingface.co/hustvl/Vim-tiny-midclstok |
| [Vim-small](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   80.5   | 95.1 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-small<sup>+</sup>](https://huggingface.co/hustvl/Vim-small-midclstok)    |       26M       |   81.6   | 95.4 | https://huggingface.co/hustvl/Vim-small-midclstok |
| [Vim-base](https://huggingface.co/hustvl/Vim-base-midclstok)    |       98M       |   81.9   | 95.8 | https://huggingface.co/hustvl/Vim-base-midclstok |

**Notes:**
- <sup>+</sup> means that we finetune at finer granularity with short schedule.
## Evaluation on Provided Weights
To evaluate `Vim-Ti` on ImageNet-1K, run:
```bash
python main.py --eval --resume /path/to/ckpt --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /path/to/imagenet
```

To evaluate `Vim-Ti` on Imagenette (on a SLURM cluster), run:
```bash
env -u SLURM_PROCID python vim/main.py --eval \
--model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--resume ./checkpoints/vim_t_midclstok_76p1acc.pth \
--data-path ./imagenette2-320 --batch-size 64 --data-set IMAGENETTE
```
## Post-Training Quantization (PTQ4VM)

This branch adds PTQ4VM support for quantizing Vim models after training.

### Quick Start

Run the full pipeline (activation scale generation + quantization + evaluation) with a single script:

```bash
# Via SLURM
sbatch vim/scripts/ptq.sh

# Or directly
bash vim/scripts/ptq.sh
```

### Step-by-Step Usage

**1. Generate activation scales**

```bash
python vim/generate_act_scale.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume checkpoints/vim_t_midclstok_76p1acc.pth \
  --data-path data/imagenet \
  --data-set IMNET \
  --batch-size 32 \
  --scales-output-path ./act_scales/vim_t_imnet/ \
  --calib-use-val
```

**2. Run quantization and evaluation**

```bash
python vim/quant.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume checkpoints/vim_t_midclstok_76p1acc.pth \
  --data-path data/imagenet \
  --data-set IMNET \
  --act_scales ./act_scales/vim_t_imnet/smoothing_t.pt \
  --qmode ptq4vm \
  --n-lvw 256 --n-lva 256 \
  --alpha 0.5 \
  --epochs 30 \
  --batch-size 32 \
  --train-batch 16 \
  --calib-use-val
```

### Key Arguments

| Argument | Default | Description |
|:--|:--|:--|
| `--qmode` | `ptq4vm` | Quantization mode |
| `--n-lvw` | `256` | Number of weight quantization levels |
| `--n-lva` | `256` | Number of activation quantization levels |
| `--alpha` | `0.5` | Smoothing factor for activation-weight balancing |
| `--epochs` | `100` | Layer-wise optimization iterations |
| `--train-batch` | `256` | Sub-batch size for layer-wise optimization |
| `--calib-use-val` | `false` | Use validation split for calibration (no train split needed) |
| `--act_scales` | - | Path to precomputed activation scales (.pt file) |
| `--scales-output-path` | - | Directory to save generated activation scales |

### Notes

- The `--calib-use-val` flag allows running the full pipeline without downloading the ImageNet train split. Calibration only uses forward passes (no labels), so using the validation split is safe.
- The activation scales file is named automatically based on model size: `smoothing_t.pt` (tiny), `smoothing_s.pt` (small), `smoothing_b.pt` (base).
- Both Imagenette (`--data-set IMAGENETTE`) and ImageNet (`--data-set IMNET`) are supported.

## Acknowledgement :heart:
This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Causal-Conv1d ([code](https://github.com/Dao-AILab/causal-conv1d)), DeiT ([paper](https://arxiv.org/abs/2012.12877), [code](https://github.com/facebookresearch/deit)). Thanks for their wonderful works.

## Citation
If you find Vim is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
 @inproceedings{vim,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
