# Templating

The main purpose is to extract substr that we want to translate while retaining the structure of the document.

Example:
```
### What is templating?

< The Answer is present here >
```

Translation:
```
### टेम्प्लेटिंग क्या है?

< उत्तर यहाँ मौजूद है >
```

Using `templating` , we extract 2 sub-strings from the above document. They are:
1. What is templating?
2. The Answer is present here

This 2 sub-strings are sent to down-stream modules for translation.


To run templating on the dataset, use - `setu-translate/stages/perform_templating.py`.

```bash
HF_DATASETS_CACHE=tmp python setu-translate/stages/perform_templating.py \
    --glob_path "data/wiki_en/wiki_en_data.parquet" \
    --cache_dir_for_original_data "data/wiki_en/cache" \
    --base_save_path "output/wiki_en/batches/5/doc_csvs" \
    --save_path "output/wiki_en/batches/5/templated" \
    --text_col body \
    --url_col url \
    --timestamp_col timestamp \
    --source_type wiki_en \
    --translation_type sentence \
    --use_cache False \
    --split "train[:5%]"
```

where
- `glob_path` : Glob path of all the parquets files of the dataset
- `cache_dir_for_original_data` : cache directory to use for the dataset to prevent reprocessing when use HF datasets - `load_dataset` module.
- `base_save_path` : where will the translated substrings be stored for `replace` operation downstream
- `save_path` : where will the output of this module be stored.
- `text_col` : name of the text column in the dataset.
- `url_col` : name of the url column in the dataset.
- `timestamp_col` : name of the timestamp column in the dataset.
- `source_type` : name of the dataset
- `translation_type` : type of substrings to extract - "sentence" or "chunk".
- `use_cache` : whether to use cache for `map` operation of huggingface datasets.
- `split` : how much data to template. Can be useful for creating batches.