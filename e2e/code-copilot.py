import contextlib
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from ray import serve
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    PeftModel #Parameter-efficient fine tuning model
)

app = FastAPI()


class CopilotRequest(BaseModel):
    inputs: str = Field(description="The text of the prompt to use")
    parameters: dict = Field(description="Additional parameters to use")


@serve.deployment()
@serve.ingress(app)
class MozPilot:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("../../starcoderbase-1b")
        checkpoint_path = "../../checkpoint-500"
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            quantization_config=None, 
            device_map=None,
            torch_dtype=torch.bfloat16,
        )

        self.model = PeftModel.from_pretrained(model, checkpoint_path, adapter_name="copilot")
        if not hasattr(self.model, "hf_device_map"):
            self.model.cuda()

        self.model.add_weighted_adapter(["copilot"], [0.8], "code_buddy")
        self.model.set_adapter("code_buddy")

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


pilot_app = MozPilot.bind()
