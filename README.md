# setu-translate

## Installation

```bash
conda create -n translate-env python=3.10

conda activate translate-env

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install datasets transformers

cd IndicTransTokenizer

pip install --editable ./
```