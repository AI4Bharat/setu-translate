import numpy as np
from functools import partial
import glob
import os
import re
from document import Document
import spacy
import argparse
from hashlib import sha256
from math import ceil
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
from pyspark.sql import Window

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():

    parser = argparse.ArgumentParser(description="Performs templating on the entire dataset and extract strings")

    parser.add_argument(
        "--glob_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--id_col",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        "--text_col",
        type=str,
        required=False,
        default="content"
    )

    parser.add_argument(
        "--url_col",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        "--timestamp_col",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        "--source_type",
        type=str,
        required=True,
    )  

    parser.add_argument(
        "--translation_type",
        type=str,
        required=False,
        default="sentence",
        choices=["sentence", "chunk"]
    )

    parser.add_argument(
        "--filter_invalid_terminal",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--use_spacy",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--add_placeholders",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        required=False,
        default=None
    )

    parser.add_argument(
        "--docs_per_partition",
        type=int,
        required=False,
        default=2056
    )

    parser.add_argument(
        "--output_sentence_dataset",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--sentences_per_shard",
        type=int,
        required=False,
        default=10000
    )

    parser.add_argument(
        "--global_sent_ds_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args

def remove_non_terminated_chunks(
    idx,
    partition,
    schema,
    text_col="content",
):
    
    TERMINAL_PUNCTUATIONS = (
       ".", "!", "?", ":", ",", ";", ")", "\"", "\'",
    )

    # chunks ending with these patterns should be completely removed.
    TERMINAL_PUNCTUATIONS_EXCEPTION = (
        "...",
        "####",
    )
    
    def is_terminal_valid(text):
        if text.endswith(TERMINAL_PUNCTUATIONS_EXCEPTION):
            return False
        return text.endswith(TERMINAL_PUNCTUATIONS)
        
    for row in partition:
        chunks = [chunk for chunk in row[text_col].split("\n") if is_terminal_valid(chunk) ] 
        cleaned_text = "\n".join(chunks)

        yield [
            row[_col] if _col.name != text_col else cleaned_text for _col in schema 
        ]


def perform_templating(
    idx,
    partition,
    add_placeholders=False, 
    sent_tokenization_model=None,
    id_col=None,
    text_col="content",
    url_col="url",
    timestamp_col="timestamp",
    source_type="refinedweb_cc",
    translation_type="sentence",
):

    for row in partition:

        doc_schema = Document.get_template_doc_schema()

        if not row[text_col] or not len(row[text_col]) or row[text_col] == str(None):
            continue

        doc = Document(
            text=row[text_col],
            doc_id=row[id_col] if id_col else None,
            url=row[url_col] if url_col else None,
            timestamp=row[timestamp_col] if timestamp_col else None,
            source_type=source_type,
            translation_type=translation_type,
            sent_tokenization_model=sent_tokenization_model,
        )

        if add_placeholders:
            doc.add_placeholders()

        doc_dict = doc.get_templated_document_attrs()
        
        yield list(doc_dict.values())

def salting(df, n_splits):
    # Number of salt buckets
    num_buckets = n_splits  # Adjust based on your data size and level of skewness
    print(f"Performing `salting` for `{n_splits}` buckets....")

    # Adding a salt column to the DataFrame
    df = df.withColumn("salt", (rand() * num_buckets).cast("int"))

    # df.show(1000)

    # Repartition based on the salted key to ensure more even distribution
    df = df.repartition(num_buckets, "salt")

    df = df.drop("salt")

    return df

def set_split_count_and_salt(df, docs_per_partition):
    df_total_rows = df.count()
    n_splits = ceil(df_total_rows/docs_per_partition)
    print(f"When required data will be repartitioned into - {n_splits} partitions")
    df = salting(df, n_splits)
    return df, df_total_rows, n_splits

    
if __name__ == "__main__":

    args = parse_args()

    sample_size = args.sample_size
    perform_invalid_terminal_chunk_removal = args.filter_invalid_terminal
    use_spacy = args.use_spacy

    spark = SparkSession \
                .builder \
                .appName("templating") \
                .getOrCreate()

    df = spark.read.format("parquet").load(args.glob_path)

    print(f"Loaded Dataset from path - {args.glob_path}")
    
    if args.sample_size:
        df = df.limit(sample_size)
        print(f"Sampled dataset of size - {args.sample_size}")

    df, df_total_rows, n_splits = set_split_count_and_salt(df, args.docs_per_partition)

    df.show(5)
    print("Performed initial salting.....")

    if perform_invalid_terminal_chunk_removal:
        orig_schema = df.schema
        df_rdd = df.rdd.mapPartitionWithIndex(
            partial(
                remove_non_terminated_chunks,
                schema=orig_schema,
                text_col=args.text_col,
            ),
        )
        df = spark.createDataFrame(df_rdd, schema=orig_schema)
        df.show(5)
        df = salting(df, n_splits)

        print(f"Performed `terminal punctuation check`")

    sent_tokenization_model = spacy.load("en_core_web_md") if use_spacy else None

    df_templated_rdd = df.rdd.mapPartitionsWithIndex(
        partial(
            perform_templating,
            add_placeholders=args.add_placeholders,
            sent_tokenization_model=sent_tokenization_model,
            id_col=args.id_col,
            text_col=args.text_col,
            url_col=args.url_col,
            timestamp_col=args.timestamp_col,
            source_type=args.source_type,
            translation_type=args.translation_type,
        ),
    )
    templated_schema = StructType([
        StructField(key, StringType(), True) 
            for key in Document.get_template_doc_schema().keys()
    ])
    df_templated = spark.createDataFrame(
        df_templated_rdd, schema=templated_schema
    )
    df_templated.show(5)
    print(f"Performed `templating`")

    df_templated_filtered = df_templated.filter(df_templated.doc_id.isNotNull())
    print(f"Filtered `null` text docs")

    os.makedirs(args.save_path, exist_ok=True)

    df_templated_filtered \
        .write \
        .mode("overwrite") \
        .parquet(args.save_path)
    print(f"Saved `templated` dataset to {args.save_path}")


    if args.output_sentence_dataset:

        df_templated_filtered = df_templated_filtered.withColumn("tlt_data", F.arrays_zip("sids", "sub_strs"))
        sentence_df = df_templated_filtered.select("*", F.posexplode("tlt_data")).drop(*["text", "tlt_data"])
        sentence_df = sentence_df.select("*", F.col("col.sids").alias("sids"), F.col("col.sub_strs").alias("sub_strs")).drop("col")

        total_sentences = sentence_df.count()
        total_shard_count = total_sentences // args.sentences_per_shard + (1 if total_sentences % args.sentences_per_shard else 0)

        sentence_df = sentence_df.repartition(total_shard_count)

        os.makedirs(args.global_sent_ds_path, exist_ok=True)

        sentence_df \
            .write \
            .mode("overwrite") \
            .parquet(args.global_sent_ds_path)
        
        sentence_df.show(5)
        print(f"Saved `Sentence-Level` dataset to {args.global_sent_ds_path}")


    

    