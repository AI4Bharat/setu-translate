#!/bin/bash

# Define options
TEMP=$(getopt -o mcjstnbdo: --long map_file:,data_cache_dir:,joblib_temp_folder:,src_lang:,tgt_lang:,num_procs_for_data_ops:,batch_size:,device:,out_base_dir: -n '$0' -- "$@")

# Exit if the options have not been properly specified.
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

# Note the quotes around '$TEMP': they are essential!
eval set -- "$TEMP"

check_arg() {
    arg_name=$1
    arg_value=$2
    if [ -z "$arg_value" ]; then
        echo "Error: Required argument - ${arg_name} not provided. Current Value=$arg_value"
        exit 1
    fi
}

# Initialize variables for flags

map_file=''
data_cache_dir=''
joblib_temp_folder=''
src_lang=''
tgt_lang=''
num_procs_for_data_ops=''
batch_size=''
device=''
out_base_dir=''

# Extract options and their arguments into variables.
while true ; do
    case "$1" in
        -m|--map_file)
            map_file="$2"; shift 2 ;;
        -c|--data_cache_dir)
            data_cache_dir="$2"; shift 2 ;;
        -j|--joblib_temp_folder)
            joblib_temp_folder="$2"; shift 2 ;;
        -s|--src_lang)
            src_lang="$2"; shift 2 ;;
        -t|--tgt_lang)
            tgt_lang="$2"; shift 2 ;;
        -n|--num_procs_for_data_ops)
            num_procs_for_data_ops="$2"; shift 2 ;;
        -b|--batch_size)
            batch_size="$2"; shift 2 ;;
        -d|--device)
            device="$2"; shift 2 ;;
        -o|--out_base_dir)
            out_base_dir="$2"; shift 2;;
        --) shift ; break ;;
        *) echo "Internal error!" ; exit 1 ;;
    esac
done

check_arg "map_file" $map_file 
check_arg "data_cache_dir" $data_cache_dir
check_arg "joblib_temp_folder" $joblib_temp_folder
check_arg "src_lang" $src_lang
check_arg "tgt_lang" $tgt_lang
check_arg "num_procs_for_data_ops" $num_procs_for_data_ops
check_arg "batch_size" $batch_size
check_arg "device" $device
check_arg "out_base_dir" $out_base_dir

echo "Running map-run using the following arguments:"
echo "map_file=$map_file"
echo "data_cache_dir=$data_cache_dir"
echo "joblib_temp_folder=$joblib_temp_folder"
echo "src_lang=$src_lang"
echo "tgt_lang=$tgt_lang"
echo "num_procs_for_data_ops=$num_procs_for_data_ops"
echo "batch_size=$batch_size"
echo "device=$device"
echo "out_base_dir=$out_base_dir"

# The file to read from, provided as the first command line argument
FILENAME=$map_file

# Check if the file exists
if [ ! -f "$FILENAME" ]; then
    echo "File not found: $FILENAME"
    exit 2
fi

run_translation() {

    shard_path=$1
    shard_id=$2

    HF_DATASETS_CACHE=$data_cache_dir python ${SETU_TRANSLATE_ROOT}/setu-translate/stages/binarize.py \
        --root_dir "$PWD" \
        --data_files "${shard_path}/sentences/*.arrow" \
        --cache_dir $data_cache_dir \
        --binarized_dir "${out_base_dir}/${shard_id}/${tgt_lang}/binarized_sentences" \
        --joblib_temp_folder $joblib_temp_folder \
        --batch_size $batch_size \
        --total_procs $num_procs_for_data_ops \
        --run_joblib False \
        --src_lang $src_lang \
        --tgt_lang $tgt_lang

    # Check exit status of the first command
    if [ $? -ne 0 ]; then
        echo $shard_path
        return 1
    fi

    HF_DATASETS_CACHE=$data_cache_dir python ${SETU_TRANSLATE_ROOT}/setu-translate/stages/tlt_pipelines/translate_joblib.py \
        --root_dir "$PWD" \
        --data_files "${out_base_dir}/${shard_id}/${tgt_lang}/binarized_sentences/*.arrow" \
        --cache_dir $data_cache_dir \
        --base_save_dir "${out_base_dir}/${shard_id}/${tgt_lang}/model_out" \
        --joblib_temp_folder $joblib_temp_folder \
        --batch_size $batch_size \
        --total_procs $num_procs_for_data_ops \
        --devices $device

    # Check exit status of the second command
    if [ $? -ne 0 ]; then
        echo $shard_path
        return 1
    fi

}

# Read the file line by line
while IFS= read -r line; do

    child_dir=$(basename "$line")

    echo "Processing $line..........."

    result=$(run_translation $line $child_dir)

    if [ $? -ne 0 ]; then
        echo "An error occurred with shard: $result"
        echo $result > ${map_file}_errored
    else
        echo "Operation completed successfully"
    fi

done < "$FILENAME"