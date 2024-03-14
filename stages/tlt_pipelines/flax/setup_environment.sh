#!/usr/bin/bash

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install transformers datasets flax gcsfs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
apt-get update -y
apt-get install -y golang-go
pip install jax-smi

gsutil -m cp -r gs://translation-ai4b/setu-translate /opt/
pip install --editable /opt/setu-translate/IndicTransTokenizer/
