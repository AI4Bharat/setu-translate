#!/bin/bash

metrics_inp_glob_path=$1
results_base_dir=$2

for folder in $metrics_inp_glob_path; do
    echo "Processing $folder ....."
    dir_name=$(basename "$folder")
    echo "Directory: $dir_name"

    # Splitting at '__' and getting the second part
    IFS='_' read -ra ADDR <<< "$dir_name"
    src_lang="${ADDR[0]}_${ADDR[1]}"
    tgt_lang="${ADDR[3]}_${ADDR[4]}"
    echo "Source Language: $src_lang, Target Language: $tgt_lang"

    if [[ "$folder" == *"conv"* ]]; then
        results_save_path=$results_base_dir/conv/$dir_name
    else
        results_save_path=$results_base_dir/gen/$dir_name
    fi

    bash compute_metrics.sh $folder/pred.txt $folder/ref.txt $tgt_lang >> $results_save_path

done