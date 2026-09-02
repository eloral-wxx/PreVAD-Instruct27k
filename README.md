<h1 align="center">
Enhancing Video Anomaly Understanding<br>
via Multi-Task Instruction Tuning
</h1>

<p align="center">
  <a href="https://github.com/eloral-wxx">Xuxu Wang</a>,
  Xiaoyu Wu,
  Zihao Liu
</p>

<p align="center">
  Communication University of China
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11230620/references">
    <img src="https://img.shields.io/badge/Paper-IEEE-red.svg">
  </a>
  <a href="https://huggingface.co/datasets/eloral-wxx/PreVAD-Instruct27k">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg">
  </a>
  <a href="https://github.com/eloral-wxx/PreVAD-Instruct27k">
    <img src="https://img.shields.io/badge/Code-GitHub-black.svg">
  </a>
</p>

<p align="center">
  <img src="assets/paper-overview.png" width="95%">
</p>

Overview of the dataset construction and multi-task instruction-tuning pipeline used in this project.

This repository contains:

- the `PreVAD-Instruct27k` instruction dataset annotations used in this project
- training code for Video-LLaVA-based instruction tuning
- inference scripts for abnormal description, question answering, and multi-dataset classification
- evaluation scripts for UniEval, BLEURT, and MoverScore

## Repository Structure

```text
.
├── datasets/
│   ├── PreVAD-Instruct27k/
│   ├── MSAD/
│   ├── UCF-Crime/
│   └── xd-violence/
├── inference/
│   ├── Abnormal description  & QA/
│   └── Classification/
├── metrics/
│   ├── BLEURT/
│   ├── MoverScore/
│   ├── UniEval/
│   └── model-answers/
├── training/
├── init_project.sh
├── env_build.sh
└── environment.yml
```

## Environment Setup

### 1. Create the Conda environment

```bash
bash env_build.sh
conda activate videollava_tuning
```

Or manually:

```bash
conda env create -f environment.yml -n videollava_tuning
conda activate videollava_tuning
```

### 2. Clone external dependencies

Some evaluation and model scripts depend on external repositories:

```bash
bash init_project.sh
```

This will clone the following projects into `external/`:

- `hawk`
- `HolmesVAU`
- `UniEval`

## Data Preparation

This repository includes dataset annotation files, but does **not** include the original raw videos or all model checkpoints.

You should prepare:

- raw videos for `PreVAD-Instruct27k`
- evaluation videos for `UCF-Crime`, `XD-Violence`, and `MSAD`
- required pretrained model weights from Hugging Face or upstream project repositories



### Public dataset sources

For the public benchmarks used in this project, download the original videos from the official pages:

- `UCF-Crime`: UCF CRCV project page
- `XD-Violence`: XD-Violence project page
- `MSAD`: MSAD dataset page

### Recommended local layout

```text
datasets/
├── PreVAD-Instruct27k/
│   ├── filter_train.json
│   ├── filter_test.json
│   ├── AbnormalVideos/
│   └── NormalVideos/
├── UCF-Crime/
│   └── videos/
│       └── test/
├── xd-violence/
│   └── other_datasets/
│       └── xd_videos/
└── MSAD/
    └── other_datasets/
        └── msad_videos/
```

The repository now uses repository-relative dataset paths by default. If you adapt the project to a different storage layout, prefer environment variables or a small local path config instead of hardcoded absolute paths.

### Validate your local dataset layout

After placing the videos, run:

```bash
python scripts/validate_dataset_layout.py
```

This checks whether the videos referenced by the annotation JSON files are actually present in your local repository layout.

## Training

The main training entry is:

```bash
python training/video-llava-prevad-instruct27k.py
```

For better portability, the training script now supports:

- `WANDB_API_KEY`: optional Weights & Biases login key
- `PREVAD_VIDEO_ROOT`: root directory of your local video files

Example:

```bash
export PREVAD_VIDEO_ROOT=/path/to/your/videos
export WANDB_API_KEY=your_wandb_key
python training/video-llava-prevad-instruct27k.py
```

If you do not use Weights & Biases, leave `WANDB_API_KEY` unset.

## Inference

Inference scripts are organized by task and benchmark.

### Abnormal description and QA

Scripts are under:

```text
inference/Abnormal description  & QA/
```

### Classification

Scripts are under:

```text
inference/Classification/
```

They are grouped by dataset:

- `PreVAD-Instruct27k-level-1`
- `PreVAD-Instruct27k-level-2`
- `XD-Violence`
- `MSAD`
- `UCF-Crime`

Before running them, verify:

- the local dataset paths
- the checkpoint paths
- the Hugging Face cache directory
- GPU and dependency compatibility

## Evaluation

Provided evaluation utilities:

- `metrics/UniEval/unieval-count.py`
- `metrics/BLEURT/bleurt-caculate.py`
- `metrics/MoverScore/movescore.py`

Model outputs can be organized under:

```text
metrics/model-answers/
```

## Citation

If you use this repository, please cite:

```bibtex
@article{prevad_instruct27k,
  title={Enhancing Video Anomaly Understanding via Multi-Task Instruction Tuning},
  author={Wang, Xuxu and Wu, Xiaoyu and Liu, Zihao},
  journal={IEEE Signal Processing Letters},
  year={2025},
  doi={10.1109/LSP.2025.3630089}
}
```

## Contact

For questions about the project, please open an issue on GitHub.
