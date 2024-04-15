import contextlib
import os
import json
import torch
from fastapi import FastAPI, Request
from typing import Dict
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    PeftModel #Parameter-efficient fine tuning model
)

app = FastAPI()

# Path of the local directory where both base models and adapted models are stored
MODELS_LOCAL_DIRECTORY = "./models"
# Name of the base model directory
BASE_MODEL_NAME = "starcoderbase-1b"
# Name of the adapted model directory
ADAPTED_MODEL_NAME = "checkpoint-500"
ADAPTED_MODEL_CONFIG_FILE_NAME = "adapter_config.json"
ADAPTED_MODEL_CONFIG_KEY_TO_MODIFY = "base_model_name_or_path"

class CopilotRequest(BaseModel):
    inputs: str = Field(description="The text of the prompt to use")
    parameters: dict = Field(description="Additional parameters to use")


@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class MozPilot:
    def __init__(self, args: Dict[str, str]) -> None:
        base_model_directory = os.path.join(MODELS_LOCAL_DIRECTORY, BASE_MODEL_NAME)
        adapted_model_directory = os.path.join(MODELS_LOCAL_DIRECTORY, ADAPTED_MODEL_NAME)
        adapted_model_config_file_path = os.path.join(adapted_model_directory, ADAPTED_MODEL_CONFIG_FILE_NAME)
        self._modify_adapted_model_config(adapted_model_config_file_path, (ADAPTED_MODEL_CONFIG_KEY_TO_MODIFY, base_model_directory))

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_directory)
        model = AutoModelForCausalLM.from_pretrained(
            adapted_model_directory,
            quantization_config=None,
            device_map=None,
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(model, adapted_model_directory, adapter_name="copilot")
        if not hasattr(self.model, "hf_device_map"):
            self.model.cuda()

        self.model.add_weighted_adapter(["copilot"], [0.8], "code_buddy")
        self.model.set_adapter("code_buddy")

    def _modify_adapted_model_config(self, adapted_model_config_file_path: str, config_key_value_pair: tuple[str,str]):
        """Modify the adapted model json config file to update the key value pair"""
        with open(adapted_model_config_file_path, 'r+') as json_config_file:
            dict_formatted_data = json.load(json_config_file)
            dict_formatted_data[config_key_value_pair[0]] = config_key_value_pair[1]
            json_config_file.seek(0)
            json.dump(dict_formatted_data, json_config_file, indent=2)
            json_config_file.truncate()

    def get_code_completion(self, prompt, disable=False):
        context = contextlib.nullcontext
        if disable:
            context = self.model.disable_adapter
        self.model.eval()
        with context():
            tokens = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                input_ids=tokens.input_ids.cuda(),
                max_new_tokens=128,
                temperature=0.2,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.0,
            )
            # Filter out the request tokens.
            masked_outputs = outputs[:, tokens.input_ids.shape[1]:]
        return self.tokenizer.batch_decode(masked_outputs, skip_special_tokens=False)[0]

    @app.post("/models/generate")
    def generate(self, request: CopilotRequest):
        answer = self.get_code_completion(request.inputs)
        print(f"***DEBUG Request {request}\nResponse: {answer}")
        return {"generated_text": answer, "status": 200}


def app_builder(args: Dict[str, str]) -> Application:
    return MozPilot.bind(args)

# TODO Uncomment the next line if attempting to autogenerate the config.
app = MozPilot.bind(args={})
