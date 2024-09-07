## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)

---

# Introduction

Anonymized source code for our submission on 3D point cloud part segmentation.

---

# Installation

## Requirements
A detailed list of requirements can be found in `requirements.txt`. However, some packages require specific installation procedures, therefore the following steps should be followed in order to correctly install the environment.

## Setup

We suggest using [Mamba](https://mamba.readthedocs.io/en/latest/) to create and manage the environment.

```bash
mamba create -n pytorch3d python=3.10 -y
mamba activate pytorch3d
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html
pip install torch_geometric
pip install pyg_lib torch_scatter torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu116.html --force-reinstall --no-cache-dir
pip install ftfy h5py huggingface-hub imageio matplotlib numba numpy open3d opencv-python packaging pandas pillow platformdirs plotly pyrender safetensors scikit-image scikit-learn scipy tqdm traitlets transformers trimesh umap-learn yacs  torchmetrics lightning
```

---

# Usage

We refer to the `README.md` file under `tests/` for detailed instructions on how to run benchmarks. You will need to download the datasets from the respective websites (some might require registration or filling a request form):
* [PartNet](https://partnet.cs.stanford.edu/)
* [PartNetE](https://colin97.github.io/PartSLIP_page/)
* [ScanObject-NN](https://hkust-vgd.github.io/scanobjectnn/)
* [FAUST (from SATR)](https://github.com/Samir55/SATR)
* ShapeNetPart is downloaded from torch geometric automatically

Lastly, the `config.py` file shall be edited according to the datasets' location on the file system, so that they can be properly loaded.
