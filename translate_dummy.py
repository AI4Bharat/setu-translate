import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import glob
import tqdm

def parse_args():

    parser = argparse.ArgumentParser(description="Perform translation")

    parser.add_argument(
        "--data_glob_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        default="~/.cache",
    )

    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=False,
        default="ai4bharat/indictrans2-en-indic-dist-200M",
    )

    parser.add_argument(
        "--src_lang",
        type=str,
        required=False,
        default="eng_Latn",
    )

    parser.add_argument(
        "--dest_lang",
        type=str,
        required=True,
        default="hin_Deva"
    )

    parser.add_argument(
        "--direction",
        type=str,
        default="en-indic",
        required=False,
        choices=["en-indic", "indic-en"],
    )

    parser.add_argument(
        "--loader_workers",
        type=int,
        default=32,
        required=False
    )

    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        required=False
    )

    parser.add_argument(
        "--csv_batch_size",
        type=int,
        default=32,
        required=False,
    )

    parser.add_argument(
        "--model_batch_size",
        type=int,
        default=32,
        required=False,
    )

    parser.add_argument(
        "--devices",
        type=lambda x: [ idx.strip() for idx in x.split(",") if idx and len(idx.strip()) ],
        required=True,
    )

    args = parser.parse_args()

    return args

class Translator:

    def __init__(
        self, 
        ckpt_dir, 
        direction, 
        devices,
        src_lang,
        dest_lang,
        batch_size
    ):
        self.ckpt_dir, self.direction, self.devices = ckpt_dir, direction, devices
        self.device_count = len(self.devices)
        self.tokenizer_map, self.ip_map, self.model_map = self.load_pipeline(self.ckpt_dir, self.direction, self.devices)

        self.src_lang, self.dest_lang = src_lang, dest_lang
        self.batch_size = batch_size

    @staticmethod
    def initialize_tokenizer_and_processor(direction):
        tokenizer = IndicTransTokenizer(direction=direction)
        ip = IndicProcessor(inference=True)
        return tokenizer, ip

    @staticmethod
    def initialize_model(ckpt_dir, device):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )    
        model = model.to(device)
        model.half()
        model.eval()
        return model

    def load_pipeline(
        self,
        ckpt_dir,
        direction,
        devices
    ):

        tokenizer_map, ip_map, model_map = [], [], []

        def load_stages(
            device,
            ckpt_dir,
            direction   
        ):
            tokenizer, i p= self.initialize_tokenizer_and_processor(direction)
            tokenizer_map += [tokenizerer]; ip_map += [ip]
            model_map += [self.initialize_model(ckpt_dir, f"cuda:{device}")]
            return True

        for idx in range(self.device_count):
            _ = load_stages(idx, ckpt_dir, direction)

        # _ = Parallel(n_jobs=self.device_count)(delayed(load_stages)(idx, ckpt_dir, direction) for idx in range(self.device_count))

        return tokenizer_map, ip_map, model_map

    def translate(self, sentences):
        dispatch_list = self.split_sentence_batch(sentences)
        gpu_outputs = Parallel(n_jobs=self.device_count)(delayed(self.forward)(idx, sentences) for idx, sentences in enumerate(dispatch_list))
        outputs = []
        for gpu_out in gpu_outputs:
            outputs += gpu_out
        return outputs

    def forward(
        self,
        device_idx,
        sentences,
        **kwargs,
    ):
        batch = self.preprocess_batch(sentences, device_idx)
        with torch.inference_mode():
            outputs = self.model_map[device_idx].generate(
                input_ids=batch["input_ids"].to(f"cuda:{device_idx}"), 
                attention_mask=batch["attention_mask"].to(f"cuda:{device_idx}"),
                num_beams=1, 
                num_return_sequences=1, 
                max_length=256
            )
        outputs = self.postprocess_batch(outputs, device_idx)
        return outputs

    def preprocess_batch(self, sentences, device_idx):
        batch = self.ip_map[device_idx].preprocess_batch(
            sentences, 
            src_lang=self.src_lang, 
            tgt_lang=self.dest_lang
        )
        batch = self.tokenizer_map[device_idx](batch, src=True, return_tensors="pt")
        return batch

    def split_sentence_batch(self, sentences):
        batch_size_per_gpu = len(sentences) // self.device_count
        batch_per_gpu = []
        start, end = 0, batch_size_per_gpu
        while(start < len(sentences)):
            batch_per_gpu += [sentences[start:end]]
            start = end
            end = start + batch_size_per_gpu if len(sentences) - end > batch_size_per_gpu else len(sentences)
        return batch_per_gpu

    def split_tensor_batch(self, tensor_batch, batch_size):
        """
        Splits the tensor batch into smaller batches for each GPU.
        """
        # Assuming tensor_batch is a list or a numpy array of tensors
        batch_size_per_gpu = batch_size // self.device_count

        batch_per_gpu = []
        start, end = 0, batch_size_per_gpu
        while(start < batch_size):
            batch_per_gpu += [{ key: tensor_batch[key][start:end] for key in tensor_batch.keys() }]
            start = end
            end = start + batch_size_per_gpu if batch_size - end > batch_size_per_gpu else batch_size

        return batch_per_gpu

    def postprocess_batch(self, outputs, device_idx):
        outputs = self.tokenizer_map[device_idx].batch_decode(outputs, src=False)
        outputs = self.ip_map[device_idx].postprocess_batch(outputs, lang=self.dest_lang)
        return outputs

class TranslateData(Dataset):

    def __init__(
        self, 
        data_glob_path, 
        cache_dir,
        num_proc,
    ):

        self.ds = load_dataset(
            "csv",
            data_files=glob.glob(args.data_glob_path),
            cache_dir=args.cache_dir,
            num_proc=num_proc,
            split="train"
        )
        
    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        sentences = self.read_csv(self.ds[idx]["csv_path"])
        return {
            "sentences": sentences,
            "csv_path": self.ds[idx]["csv_path"]
        }
    
    def __len__(self):
        return len(self.ds)

    @staticmethod
    def read_csv(csv_path):
        df = pd.read_csv(csv_path)
        return df["substr"].tolist()

def custom_collate(batch):

    sentences = []
    csvs = []
    start_pos_s = []
    end_pos_s = []

    start = 0

    for doc_strs in batch:
        sentences += doc_strs["sentences"]
        csvs += [doc_strs["csv_path"]]
        start_pos_s += [start]
        start += len(doc_strs["sentences"])
        end_pos_s += [start]

    return {
        "sentences": sentences,
        "csv_paths": csvs,
        "start_pos_s": start_pos_s,
        "end_pos_s": end_pos_s,
    }

def update_csvs(
    output, 
    csv_paths, 
    start_pos_s, 
    end_pos_s,
    base_save_path,

):
    restructured_out = []
    for start_pos, end_pos in tuple(zip(csv_paths, start_pos_s, end_pos_s)):
        restructured_out += [output[start_pos:end_pos]]

    def add_translation_to_csv(
        csv_path,
        out
    ):
        df = pd.read_csv(csv_path)
        df["translated"] = out
        # df.to_csv(csv_path, index=False)

    path_n_out = tuple(zip(csv_paths, restructured_out))
    job_count = os.cpu_count() if os.cpu_count() <= len(csv_paths) else len(csv_paths)
    result = Parallel(n_jobs=job_count)(delayed(add_translation_to_csv)(path, out) for path, out in path_n_out)
    return True

if __name__ == "__main__":

    args = parse_args()

    ds = TranslateData(
        data_glob_path=args.data_glob_path,
        cache_dir=args.cache_dir,
        num_proc=96,
    )

    data_loader = DataLoader(
        ds, 
        num_workers=args.loader_workers, 
        batch_size=args.csv_batch_size,
        collate_fn=custom_collate,
        prefetch_factor=args.prefetch_factor,
    )

    tlt_pipe = Translator(
        ckpt_dir=args.model_ckpt, 
        direction=args.direction, 
        devices=args.devices,
        src_lang=args.src_lang,
        dest_lang=args.dest_lang,
        batch_size=args.model_batch_size
    )

    for _, batch in tqdm.tqdm(enumerate(data_loader, 0), unit="batch", total=len(data_loader)):
        start_pos = 0
        end_pos = start_pos + args.model_batch_size if len(batch["sentences"]) > args.model_batch_size else len(batch["sentences"]) 
        outputs = []
        while start_pos < len(batch["sentences"]):
            outputs += tlt_pipe.translate(batch["sentences"][start_pos:end_pos])
            start_pos = end_pos
            end_pos = start_pos + args.model_batch_size if len(batch["sentences"][start_pos:]) > args.model_batch_size else len(batch["sentences"]) 
            # print(outputs)
        # print(outputs)
        # break

        # _ = update_csvs(outputs, batch["csv_paths"], batch["start_pos_s"], batch["end_pos_s"])


    
