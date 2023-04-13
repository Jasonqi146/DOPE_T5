# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import json
import torch
import transformers
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, AdamW

WEIGHT_PATH = "./T5_new"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model =  T5ForConditionalGeneration.from_pretrained(WEIGHT_PATH)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded")

    def predict(
        self,
        input_text: str = Input(description="Standard American English"),
        max_gen_len: int = Input(description="Maximum length of generated sentence", default=600, ge=0, le=1000),
        temperature: float = Input(description="Temperature of generated sentence", default=1.0, ge=0.0, le=1.),
        top_k: int = Input(description="Top k of generated sentence", default=5, ge=0, le=10),
        top_p: float = Input(description="Top p of generated sentence", default=0.9, ge=0.0, le=1.),
        repetition_penalty: float = Input(description="Repetition penalty of generated sentence", default=1.0, ge=1.0, le=1000.),
    ) -> str:
        """Run a single prediction on the model"""
        input = self.tokenizer(input_text)
        input = {key: torch.tensor(val) for key, val in input.items()}
        input_ids = torch.unsqueeze(input['input_ids'], dim=0).to(self.device)
        attention_mask = input['attention_mask'].to(self.device)
        print("input tokenized")
        with torch.no_grad():
            generated_sequences = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                max_length=max_gen_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                )
        text = self.tokenizer.decode(generated_sequences[0], clean_up_tokenization_spaces=True)
        text = text.replace('</s>', '')
        text = text.replace('<s>', '')
        text = text.replace('<pad>', '')
        print(text)
        return text