import contextlib
import os
import json
import torch
import logging
import sys
from artifact_store import ArtifactStore
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

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = FastAPI()

# GCS project and bucket name where both model repositories (base model & adapted model) live
GCS_PROJECT_NAME = "moz-fx-mlops-inference-nonprod"
GCS_BUCKET_NAME = "mf-models-test1"

# Remote directory in the GCS bucket where the base model repository lives
GCS_BASE_MODEL_STORAGE_DIRECTORY = "copilot-demo/starcoderbase-1b"

# Remote directory in the GCS bucket where the adapter model repository lives
GCS_ADAPTED_MODEL_STORAGE_DIRECTORY = "CodeAutocompletionFlow/argo-codeautocompletionflow-wp5kr/trained/checkpoint-1000"

# Local directory where both model repositories (base model & adapted model) will be downloaded to from GCS
DESTINATION_MODEL_STORAGE_DIRECTORY = "./models"

class CopilotRequest(BaseModel):
    inputs: str = Field(description="The text of the prompt to use")
    parameters: dict = Field(description="Additional parameters to use")


@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class MozPilot:
    def __init__(self, args: Dict[str, str]) -> None:
        # Name of the local directory to store the base model repository downloaded from GCS
        DESTINATION_BASE_MODEL_DIRECTORY_NAME = "starcoderbase-1b"

        # Name of the local directory to store the adapted model repository downloaded from GCS
        DESTINATION_ADAPTED_MODEL_DIRECTORY_NAME = "checkpoint-500"

        # Name of the config file within the adapted model repository to be updated
        ADAPTED_MODEL_CONFIG_FILE_NAME = "adapter_config.json"

        # Key of the key:value pair within the adapted model config file to be updated
        ADAPTED_MODEL_CONFIG_KEY_TO_MODIFY = "base_model_name_or_path"

        dst_base_model_dir = os.path.join(DESTINATION_MODEL_STORAGE_DIRECTORY, DESTINATION_BASE_MODEL_DIRECTORY_NAME)
        dst_adapted_model_dir = os.path.join(DESTINATION_MODEL_STORAGE_DIRECTORY, DESTINATION_ADAPTED_MODEL_DIRECTORY_NAME)

        artifact_store = ArtifactStore(GCS_PROJECT_NAME, GCS_BUCKET_NAME)
        logger.info(f"Fetching '{GCS_BASE_MODEL_STORAGE_DIRECTORY}' from GCS bucket '{GCS_BUCKET_NAME}' to '{dst_base_model_dir}'")
        artifact_store.fetch_directory(GCS_BASE_MODEL_STORAGE_DIRECTORY, dst_base_model_dir)
        logger.info(f"Done Fetching '{GCS_BASE_MODEL_STORAGE_DIRECTORY}' from GCS bucket '{GCS_BUCKET_NAME}' to '{dst_base_model_dir}'")
        logger.info(f"Fetching '{GCS_ADAPTED_MODEL_STORAGE_DIRECTORY}' from GCS bucket '{GCS_BUCKET_NAME}' to '{dst_adapted_model_dir}'")
        artifact_store.fetch_directory(GCS_ADAPTED_MODEL_STORAGE_DIRECTORY, dst_adapted_model_dir)
        logger.info(f"Done Fetching {GCS_ADAPTED_MODEL_STORAGE_DIRECTORY} from GCS bucket '{GCS_BUCKET_NAME}' to '{dst_adapted_model_dir}'")

        dst_adapted_model_config_file_path = os.path.join(dst_adapted_model_dir, ADAPTED_MODEL_CONFIG_FILE_NAME)
        self._modify_adapted_model_config(dst_adapted_model_config_file_path, (ADAPTED_MODEL_CONFIG_KEY_TO_MODIFY, dst_base_model_dir))

        self.tokenizer = AutoTokenizer.from_pretrained(dst_base_model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            dst_adapted_model_dir,
            quantization_config=None,
            device_map=None,
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(model, dst_adapted_model_dir, adapter_name="copilot")
        if not hasattr(self.model, "hf_device_map"):
            self.model.cuda()

        self.model.add_weighted_adapter(["copilot"], [0.8], "code_buddy")
        self.model.set_adapter("code_buddy")

    def _modify_adapted_model_config(self, dst_adapted_model_config_file_path: str, config_key_value_pair: tuple[str,str]):
        """Modify the json config file of the adapted model repo to update the key value pair"""
        with open(dst_adapted_model_config_file_path, 'r+') as json_config_file:
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
#app = MozPilot.bind(args={})
