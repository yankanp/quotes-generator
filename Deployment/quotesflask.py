from flask import Flask,request,jsonify
import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel,GPT2Config
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def hello():
    return "Quotes generator: CSC 561"

@app.route("/quote", methods=['GET'])
@cross_origin()
def get_quote():
    prompt = request.args.get('type', default='', type=str)
    use_cuda=torch.cuda.is_available()
    device=torch.device('cuda:0' if use_cuda else 'cpu')
    model_path = "gpt2_finetuned_epoch_9.pt"
    my_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', 
                                              bos_token='<|startoftext|>', 
                                              eos_token='<|endoftext|>', 
                                              pad_token='<|pad|>')
    model_state_dict = torch.load(model_path, map_location=torch.device(device))


    # Create a new model and load the state dict
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained('gpt2',config=config)
    model.resize_token_embeddings(len(my_tokenizer))

    model.load_state_dict(model_state_dict)
    generated = torch.tensor(my_tokenizer.encode(prompt)).unsqueeze(0)
    sample_outputs = model.generate(
                                generated, 
                                do_sample=True,   
                                top_k=50, 
                                max_length = 100,
                                top_p=0.95, 
                                num_return_sequences=1
                                )
    quotes=[]
    for i, sample_output in enumerate(sample_outputs):
      print("{}: {}\n\n".format(i, my_tokenizer.decode(sample_output, skip_special_tokens=True)))
      quotes.append(my_tokenizer.decode(sample_output, skip_special_tokens=True))

    return quotes

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)