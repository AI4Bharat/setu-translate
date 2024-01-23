import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import json
import ctranslate2 as ct2

tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)
# model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True).to("cuda:0")
translator = ct2.Translator("/data-3/priyam/translation/checkpoint/en-indic/ct2_int8_model", device="cuda", device_index=0)

sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")
batch = tokenizer(batch, src=True, return_tensors="pt")
placeholder_entity_maps = json.dumps(ip.get_placeholder_entity_maps())

# with torch.inference_mode():
#     outputs = model.generate(
#         input_ids=batch["input_ids"].to("cuda:0"), 
#         num_beams=1, num_return_sequences=1, max_length=256)

outputs = translator.translate_batch(
    
)

outputs = tokenizer.batch_decode(outputs, src=False)
print(outputs)
ip = IndicProcessor(inference=True)
outputs = ip.postprocess_batch(outputs, lang="hin_Deva", placeholder_entity_maps=json.loads(placeholder_entity_maps))
print(outputs)
