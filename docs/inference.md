# Inference

The main purpose is to run inference on tokenized dataset.

```bash
HF_DATASETS_CACHE=tmp python setu-translate/tlt_pipelines/translate_joblib.py \
    --root_dir "$PWD" \
    --data_files "output/wiki_en/batches/5/binarized_sentences/*.arrow" \
    --cache_dir "data/wiki_en/cache" \
    --base_save_dir "output/wiki_en/batches/5/model_out" \
    --joblib_temp_folder "tmp" \
    --batch_size 512 \
    --total_procs 64 \
    --devices "0,1,2,3,4,5,6,7"
```

where,

- `root_dir` : path from where you run this script.
- `data_files` : glob path of all parquet files of binarized datasets. Generally, output of `setu-translate/stages/binarize.py`.
- `cache_dir` : cache directory to use for HF datasets `load_dataset` operation.
- `base_save_dir` : path to store the output of this module i.e model-out.
- `joblib_temp_folder` : directory to use as temporary storage for joblib purpose.
- `batch_size` : no.of samples to use for batches to be send for translation.
- `total_procs` : no.of processes to run for HF based `map` operations related to dataset.
- `devices` : comma(`,`)-separated indices of GPU devices on which to load translation model.