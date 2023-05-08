# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import pipeline , AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    # pipeline('fill-mask', model='bert-base-uncased')
    AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
    AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1", torch_dtype=torch.float16)

if __name__ == "__main__":
    download_model()