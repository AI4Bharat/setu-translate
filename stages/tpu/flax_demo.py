import os
import numpy as np
import jax.numpy as jnp
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from modeling_flax_indictrans import FlaxIndicTransForConditionalGeneration
from transformers import FlaxAutoModelForSeq2SeqLM, AutoConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jax_smi import initialise_tracking
initialise_tracking()

tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)
# config = AutoConfig.from_pretrained(
#     "ai4bharat/indictrans2-en-indic-dist-200M",
#     revision="refs/pr/2",
#     _commit_hash="10c0b097c3216fe84c4424d897bc19921f64d2f4",
#     trust_remote_code=True,
#     force_download=True,
# )

# model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
#     "ai4bharat/indictrans2-en-indic-dist-200M", 
#     dtype=jnp.bfloat16, 
#     config=config,
#     revision="refs/pr/2",
#     _commit_hash="10c0b097c3216fe84c4424d897bc19921f64d2f4",
#     trust_remote_code=True,
#     force_download=True,
# )

model = FlaxIndicTransForConditionalGeneration.from_pretrained(
    "/data-3/priyam/tpu-translation/flax_weights/200m",
    local_files_only=True,
    dtype=jnp.bfloat16,
)

print("Loaded `tokenizer`, `processor` & `model`....")

sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
    'comparisons between the scleral rings of "juravenator" and modern birds and reptiles indicate that it may have been nocturnal.',
    'he was a professor emeritus of mathematics at pennsylvania state university, after having taught there for over 35 years.',
]

batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")
batch = tokenizer(batch, src=True, return_tensors="np", padding="max_length")
print("Preprocessed sentence batch...now translating...")

print(batch["input_ids"].shape)
print(batch["input_ids"][-1])

print(batch)

outputs = model.generate(**batch, num_beams=1, num_return_sequences=1, max_length=256, do_sample=False)
print("Model inference completed...now decoding...")

print(np.asarray(outputs.sequences)[-1])

out = tokenizer.batch_decode(np.asarray(outputs.sequences), src=False)
out = ip.postprocess_batch(out, lang="hin_Deva")
print(out)

with open("demo_out.txt", "w") as f:
    f.write("\n".join(out) + "\n")