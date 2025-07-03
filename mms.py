"""
uv sync
uv run mms.py ../metrics/data/text_hand.json ../mms_wav
"""

from transformers import VitsModel, AutoTokenizer
import torch

# Load model and tokenizer once
model = VitsModel.from_pretrained("facebook/mms-tts-heb")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-heb")

class Mms:
    def __init__(self, model_name):
        self.model = VitsModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform
        # return samples, sample_rate
        return output.squeeze().cpu().numpy(), self.model.config.sampling_rate
