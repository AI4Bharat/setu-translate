# Binarize

The main purpose is to tokenize the sentence dataset for the given `source` and `target` languages.

```bash
HF_DATASETS_CACHE=tmp python setu-translate/stages/binarize.py \
        --root_dir "$PWD" \
        --data_files "output/wiki_en/batches/5/sentences/*.arrow" \
        --cache_dir "data/wiki_en/cache" \
        --binarized_dir "output/wiki_en/batches/5/binarized_sentences" \
        --joblib_temp_folder "tmp" \
        --batch_size 512 \
        --total_procs 64 \
        --run_joblib False \
        --src_lang eng_Latn \
        --tgt_lang hin_Deva
```

where,

- `root_dir` : path from where you run this script.
- `data_files` : glob path of all parquet files of sentence datasets. Generally, output of `setu-translate/stages/data_files.py`.
- `cache_dir` : cache directory to use for HF datasets `load_dataset` operation.
- `binarized_dir` : path to store the output of this module i.e a tokenized dataset.
- `joblib_temp_folder` : directory to use as temporary storage for joblib purpose.
- `batch_size` : no.of samples to send per batch for tokenization.
- `total_procs` : no.of procs to start in `map` process which tokenizes the dataset.
- `run_joblib` : whether to run joblib version or not. If set to `False`, normal HF pipeline which utilizes mass 
- `src_lang` : source language of sentence.
- `tgt_lang` : target language of sentence.

