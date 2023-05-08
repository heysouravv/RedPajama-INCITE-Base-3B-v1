import torch
from transformers import pipeline , AutoTokenizer, AutoModelForCausalLM


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
    model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1",torch_dtype=torch.float16).to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer
    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    max_new_tokens = model_inputs.get("max_new_tokens", None)
    temperature = model_inputs.get("temperature", None)
    top_k = model_inputs.get("top_k", None)
    top_p = model_inputs.get("top_p", None)
    do_sample = model_inputs.get("do_sample", None)
    if prompt == None or max_new_tokens == None or temperature == None or top_k == None or top_p == None or do_sample == None:
        return {'message': "Insufficient arguments"}
    
    # Run the model
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, return_dict_in_generate=True,
    )
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
    # Return the results as a dictionary
    return output_str