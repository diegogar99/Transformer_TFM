from utils import * 
from tokenization import * 
from window_dataset import * 
import torch.nn as nn
import pandas as pd
from tensorflow import keras
import numpy as np
from utils import windowed_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./resources/models/lstm_model.keras"
TEST_SET = False
PRUEBA_SIMPLE = True
max_tokens = 100
context_len = 128

model = keras.models.load_model(MODEL_PATH)




if TEST_SET:
    _,_,test_text = read_datasets()
    test_ids,vocab_size = tokenizador(test_text)
    print("Creando ventanas deslizantes...")
    test_batches = windowed_dataset(test_ids, context_len)
    print("Evaluando con dataset de test...")
    loss, accuracy = model.evaluate(test_batches)

    print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    perplexity = np.exp(loss)
    print(f"Perplejidad: {perplexity:.2f}")

else:
    sp = spm.SentencePieceProcessor(model_file="./resources/models/bpe_model_shakespeare.model")

    prompt = """KING RICHARD:"""

    

    if PRUEBA_SIMPLE: 
        tokens = sp.encode(prompt, out_type=int)
        tokens = list(tokens)
        for _ in range(max_tokens):
            idx_last = tokens[-context_len:]
            x = np.expand_dims(idx_last, axis=0)
            preds = model.predict(x, verbose = False)
            logits = np.log(preds + 1e-9)
            logits = logits / 0.01
            values, indices = tf.math.top_k(logits, k=50)
            values = values.numpy().flatten()
            indices = indices.numpy().flatten()
            probs = tf.nn.softmax(values).numpy()
            next_token = np.random.choice(indices, p=probs)


            #probs = tf.nn.softmax(logits).numpy()

            #next_token = int(np.argmax(probs))
            tokens.append(next_token)
        print(tokens)
        ids = [int(t) for t in tokens]
        res = sp.decode(ids)
        print(res)
        