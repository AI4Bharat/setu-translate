# Setu-Translate: A Large Scale Translation pipeline

Setu-Translate uses [IndicTrans2 (IT2) ](https://github.com/AI4Bharat/IndicTrans2) for performing large-scale translation across English and 22 Indic Languages.

Currently, we provide inference support for [PyTorch](https://pytorch.org/get-started/locally/) and [Flax](https://flax.readthedocs.io/en/latest/index.html) versions of IT2. TPUs can be used for large-scale translation by leveraging Flax port of IT2.

# Table of Contents

1. [Quickstart](#quickstart)
2. [Overview](#overview)
3. [Usage](#usage)

## Quickstart

1. Clone repository
```bash
git clone https://github.com/AI4Bharat/setu-translate.git
```
2. Prepare environment
```bash
conda create -n translate-env python=3.10
conda activate translate-env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge pyspark
conda install pip
pip install datasets transformers
```
3. Install IndicTransTokenizer
```bash
cd IndicTransTokenizer

pip install --editable ./
```


## Overview

