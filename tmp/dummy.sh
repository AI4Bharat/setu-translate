#!/usr/bin/bash

# gsutil cp setu-translate/stages/tpu/translate_tpu_pod.py gs://translation-ai4b/setu-translate/stages/tpu

# gcloud compute tpus tpu-vm ssh tlt-flax-build \
#     --zone=us-central2-b \
#     --worker=all \
#     --command="sudo gsutil cp gs://translation-ai4b/setu-translate/stages/tpu/translate_tpu_pod.py /opt/setu-translate/stages/tpu"

# gcloud compute tpus tpu-vm ssh tlt-flax-build \
#     --zone=us-central2-b \
#     --worker=all \
#     --command="sudo python3 /opt/setu-translate/stages/tpu/translate_tpu_pod.py --data_files gs://translation-ai4b/data_to_translate/wikimedia/binarized/hin_Deva/data-00000-of-00192.arrow --base_save_dir gs://translation-ai4b/test-out/wikimedia/hin_Deva --batch_size 2048 --total_procs 196 --accelerator_type tpu --setu_translate_root /opt/setu-translate --direction en-indic --format arrow --keep_in_memory True --gcp_project sangraha-396106 --streaming True"

# gcloud compute tpus tpu-vm ssh tlt-flax-build \
#     --zone=us-central2-b \
#     --worker=all \
#     --command="sudo rm -rf /opt/.cache/huggingface/*"

gsutil cp setu-translate/stages/tpu/translate_tpu_pod.py gs://translation-ai4b/setu-translate/stages/tpu

gcloud compute tpus tpu-vm ssh tlt-flax-build \
    --zone=us-central2-b \
    --worker=all \
    --command="sudo gsutil cp gs://translation-ai4b/setu-translate/stages/tpu/translate_tpu_pod.py /opt/setu-translate/stages/tpu"

# gcloud compute tpus tpu-vm ssh tlt-flax-build \
#     --zone=us-central2-b \
#     --worker=all \
#     --command="sudo python /opt/setu-translate/stages/tpu/translate_tpu_pod.py --data_files '/mnt/disks/persist/translation/data/wikimedia/binarized/hin_Deva' --split train[:50%] --base_save_dir gs://translation-ai4b/translate-out/wikimedia/hin_Deva/model_out --batch_size 4096 --save_batch_size 100000 --total_procs 196 --accelerator_type tpu --setu_translate_root /opt/setu-translate --direction en-indic --format arrow --keep_in_memory True --gcp_project sangraha-396106 --streaming False"

# gcloud compute tpus tpu-vm ssh tlt-flax-build \
#     --zone=us-central2-b \
#     --worker=all \
#     --command="sudo rm -rf /opt/.cache/huggingface/*; rm -rf ~/.cache/huggingface/*"

# {
#         'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None), 
#         'sid': Value(dtype='string', id=None), 
#         'sub_str': Value(dtype='string', id=None), 
#         'tlt_idx': Value(dtype='int64', id=None), 
#         'placeholder_entity_map': Value(dtype='string', id=None), 
#         'translated_input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None), 
#         'tlt_file_loc': Value(dtype='string', id=None)
#     }