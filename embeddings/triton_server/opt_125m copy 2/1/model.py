import os
import sys
import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from transformers import AutoTokenizer, OPTForCausalLM, OPTForSequenceClassification
import pickle

class TritonPythonModel:

    def initialize(self, args):

        self.model_config = json.loads(args['model_config'])
        self.model_instance_device_id = json.loads(args['model_instance_device_id'])
        self.output_name = "EMBEDDING_VECTOR"
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(self.model_config, self.output_name)["data_type"])

        self.model = OPTForCausalLM.from_pretrained(
            "facebook/opt-125m", 
            output_hidden_states=True, 
            torch_dtype=torch.float16
        ).to(self.model_instance_device_id)

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    def execute(self, requests):

        modelwise_batches = {}
        responses = []
        indices_to_retrieve = []
        input_text_batch = []
        request_lengths = []
        for request_id, request in enumerate(requests):
            input_texts = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy()
            request_lengths += [len(input_texts)]
            input_text_batch += [input_text[0].decode("utf-8", "ignore") for input_text in input_texts]

        tokenized_text = self.tokenizer(
            input_text_batch, 
            padding=True,
            max_length=2048,
            truncation=True,
            return_tensors="pt"
        )

        for i in range(len(input_text_batch)):
            indices_of_1s = (tokenized_text["attention_mask"][i] == 1).nonzero()
            max_index = indices_of_1s.max()
            indices_to_retrieve += [int(max_index.numpy())]

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized_text["input_ids"].to(self.model_instance_device_id),
                attention_mask=tokenized_text["attention_mask"].to(self.model_instance_device_id),
            )

        embedding_vectors = []
        for i in range(len(input_text_batch)):
            embedding_vectors += [pickle.dumps(outputs.hidden_states[-1][i, indices_to_retrieve[i]].to("cpu").numpy().tolist())]
        
        start=0
        for i in range(len(requests)):
            request_response = []
            for j in range(request_lengths[i]):
                request_response += [embedding_vectors[start+j]]
            start += request_lengths[i]
            responses += [pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    self.output_name,
                    np.array(request_response, dtype=self.output_dtype),
                )
            ])]
            
        return responses
    