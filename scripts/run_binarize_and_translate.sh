#!/usr/bin/bash

prev=""

data_cache_dir="data/wiki_en/cache"
joblib_temp_folder="tmp"
src_lang="eng_Latn"
tgt_lang="hin_Deva"
num_procs_for_data_ops=64
batch_size=512
devices_for_translation="0,1,2,3,4,5,6,7"

for i in {5..100..5}
do

    HF_DATASETS_CACHE=tmp python setu-translate/stages/binarize.py \
        --root_dir "$PWD" \
        --data_files "output/wiki_en/batches/${i}/sentences/*.arrow" \
        --cache_dir $data_cache_dir \
        --binarized_dir "output/wiki_en/batches/${i}/binarized_sentences" \
        --joblib_temp_folder $joblib_temp_folder \
        --batch_size $batch_size \
        --total_procs $num_procs_for_data_ops \
        --run_joblib False \
        --src_lang $src_lang \
        --tgt_lang $tgt_lang

    HF_DATASETS_CACHE=tmp python setu-translate/stages/tlt_pipelines/translate_joblib.py \
        --root_dir "$PWD" \
        --data_files "output/wiki_en/batches/${i}/binarized_sentences/*.arrow" \
        --cache_dir $data_cache_dir \
        --base_save_dir "output/wiki_en/batches/${i}/model_out" \
        --joblib_temp_folder $joblib_temp_folder \
        --batch_size $batch_size \
        --total_procs $num_procs_for_data_ops \
        --devices $devices_for_translation

    prev="${i}%"

done