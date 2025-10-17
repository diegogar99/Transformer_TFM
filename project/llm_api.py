# API Rest

from flask import Flask, request, jsonify,Response
import logging
import json
from utils import generate_text
import torch 
from tokenization import spm 

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./resources/models/gpt_model.pth"
BPE_MODEL_PATH = "./resources/models/bpe_model_shakespeare.model"
CONTEXT_LEN = 128

app = Flask(__name__)

model = torch.load(MODEL_PATH, map_location=device,weights_only=False)
sp = spm.SentencePieceProcessor(model_file=BPE_MODEL_PATH)

model.to(device)

@app.route('/LLM/api/v1/generar_texto', methods=['POST'])
def respuesta_mail_atencion_cliente():

    raw_data = request.get_data()
    
    try:
        print("Datos recibidos:", raw_data)
        data = json.loads(raw_data)
        print("Datos recibidos2:", data)
        prompt = data.get('init_text')
        temperatura = data.get('temperatura')
        top_k = data.get('top_k')
        top_p = data.get('top_p')
        max_len = data.get('max_len')
        presence_penalty = data.get('presence_penalty')
        frequency_penalty = data.get('frequency_penalty')
        print("Llamando a generación de texto con los parámetros")
        respuesta = generate_text(model, sp, prompt, top_k, top_p, presence_penalty,frequency_penalty, temperatura, max_len,CONTEXT_LEN,device)

        return respuesta, 200, {"Content-Type": "text/plain; charset=ISO-8859-1"}


    except Exception as e:
        return jsonify({"error": "No se recibe un campo válido", "exception": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8011)
