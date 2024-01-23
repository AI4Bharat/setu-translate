# Sentence

The main purpose of sentence module to create a global dataset of sentences by flattening the substrings extracted from the documents into a single dataset.

To create sentence dataset use `setu-translate/stages/create_global_ds.py`

```bash
HF_DATASETS_CACHE=tmp python setu-translate/stages/create_global_ds.py \
        --paths_data "output/wiki_en/batches/5/templated/*.arrow" \
        --cache_dir "data/wiki_en/cache" \
        --global_sent_ds_path "output/wiki_en/batches/5/sentences"
```

where

- `paths_data` : glob path which contains substrings of documents. Generally, output of `setu-translate/stages/create_global_ds.py`.
- `cache_dir` : cache directory to use for HF datasets `load_dataset` operation.
- `global_sent_ds_path` : path to store the output of this module i.e a substring dataset.
