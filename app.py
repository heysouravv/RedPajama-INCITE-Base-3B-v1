from potassium import Potassium, Request, Response
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
    model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1", torch_dtype=torch.float16)
    model = model.to('cuda:0')
    context = {
        "tokenizer": tokenizer,
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    max_new_tokens = request.json.get("max_new_tokens")
    temperature = request.json.get("temperature")
    top_k = request.json.get("top_k")
    top_p = request.json.get("top_p")
    do_sample = request.json.get("do_sample")
    model = context.get("model")
    tokenizer = context.get('tokenizer')
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, return_dict_in_generate=True,
    )
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
    print(output_str)
    outputs = model(prompt)

    return Response(
        json = {"outputs": outputs}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()