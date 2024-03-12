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

```

https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/ai4b-public-nlu-nlg/sangraha/translation/it2_flax_weights.tar.gz

```bash
conda create -n translate-env python=3.10

conda activate translate-env

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install datasets transformers

cd IndicTransTokenizer

pip install --editable ./
```

## Overview