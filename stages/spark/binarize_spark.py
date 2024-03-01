import os
import torch
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from joblib import Parallel, delayed
import glob
from functools import partial
import json
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    udf,
    col,
    rand,
    posexplode,
    pandas_udf, 
    PandasUDFType
)
from pyspark.sql.types import (
    BooleanType,
    StringType, 
    StructType, 
    StructField,
    Row
)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():

    parser = argparse.ArgumentParser(description="Perform binarization")

    parser.add_argument(
        "--data_files",
        type=str,
    )

    parser.add_argument(
        "--binarized_dir",
        type=str,
    )

    parser.add_argument(
        "--src_lang",
        type=str,
    )

    parser.add_argument(
        "--tgt_lang",
        type=str,
    )

    args = parser.parse_args()

    return args

def binarize(
    idx,
    partition,
    tokenizer, 
    src_lang="eng_Latn",
    tgt_lang="hin_Deva"
):

    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction="en-indic")

    for row in partition:

        sentence = ip.preprocess_batch(
            [row["sub_strs"]], 
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )

        placeholder_entity_maps = list(map(lambda ple_map: json.dumps(ple_map), ip.get_placeholder_entity_maps(clear_ple_maps=True)))

        input_id, attention_mask = tokenizer(
            [sentence], 
            src=True, 
            return_tensors="np",
        ).values()

        yield [ row[key] for key in row.keys() ] + [ 
            row["tlt_idx"], input_id.tolist(), attention_mask.tolist() ,placeholder_entity_maps 
        ]


if __name__ == "__main__":

    args = parse_args()

    spark = SparkSession \
                .builder \
                .appName("binarize") \
                .getOrCreate()

    df = spark.read.format("parquet").load(args.data_files)

    print("Loaded Dataset....")

    print(df.schema)

    # orig_schema = df.schema

    # binarized_rdd = ds.rdd.mapPartitionsWithIndex(
    #     partial(
    #         binarize,
    #         src_lang=args.src_lang,
    #         tgt_lang=args.tgt_lang,
    #     ),
    # )

    # binarized_df = spark.createDataFrame(
    #     binarized_rdd, schema=orig_schema
    # )

    # os.makedirs(args.binarized_dir, exist_ok=True)

    # binarized_df \
    #     .write \
    #     .mode("overwrite") \
    #     .parquet(args.binarized_dir)
    # print(f"Saved `binarized` dataset to {args.binarized_dir}")

    