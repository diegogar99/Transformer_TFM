from utils import * 
from tokenization import * 
from window_dataset import * 
import torch.nn as nn
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./resources/models/gpt_model.pth"
TEST_SET = False
PRUEBA_SIMPLE = False

loss_fn = nn.CrossEntropyLoss()
context_len = 128

if TEST_SET:
    _,_,test_text = read_datasets(True)


    print(f"Longitud corpus test: {len(test_text)} caracteres")

    test_ids,vocab_size = tokenizador(test_text)

    test_dataset = LMWindowDataset(test_ids, context_len=context_len)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)
    print(f"Iteracciones: {len(test_loader)}")
    model = torch.load(MODEL_PATH, map_location=device,weights_only=False)

    test_loss, test_ppl = evaluate_ppl(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")

else:
    sp = spm.SentencePieceProcessor(model_file="./resources/models/bpe_model_shakespeare.model")

    model = torch.load(MODEL_PATH, map_location=device,weights_only=False)
    model.to(device)
    model.eval()

    prompt = "KING RICHARD:"

    if PRUEBA_SIMPLE: 
        generated = generate_text_test(model, sp, prompt, max_new_tokens=100, temperature=0.8, device=device)
        print(generated)
    else:
        test_set = [ # temperature, top_k, top_p, presence_penalty, frequency_penalty
            (0.5, None, None, 0.0, 0.0),
            (1.0, None, None, 0.0, 0.0),
            (1.5, None, None, 0.0, 0.0),

            (1.0, 10, None, 0.0, 0.0),
            (1.0, 90, None, 0.0, 0.0),
            (1.0, 200, None, 0.0, 0.0),

            (1.0, None, 0.05, 0.0, 0.0),
            (1.0, None, 0.30, 0.0, 0.0),
            (1.0, None, 0.90, 0.0, 0.0),

            (1.0, None, None, 0.2, 0.0),
            (1.0, None, None, 0.5, 0.0),
            (1.0, None, None, 1.0, 0.0),

            (1.0, None, None, 0.0, 0.2),
            (1.0, None, None, 0.0, 0.5),
            (1.0, None, None, 0.0, 1.0),

            (0.2, 50, 0.8, 0.0, 0.0),
            (0.7, None, 0.9, 0.3, 0.3),
            (1.5, 100, 1.0, 0.5, 0.5)
        ]
        test_set = [
            (0.9,100,0.95,0.3,0.2),(0.7,80,0.9,0.2,0.3),(0.5,90,0.7,0.4,0.7)
        ]

        results = []

        for temp, top_k, top_p, presence, freq in test_set:
            print(f"Test: {temp}, {top_k}, {top_p}, {presence}, {freq}")
            generated = generate_text(model, sp, prompt, top_k, top_p, presence,freq, temp, 100,context_len, device)
            results.append({
                "Temperatura": temp,
                "Top-k": top_k,
                "Top-p": top_p,
                "Presence Penalty": presence,
                "Frequency Penalty": freq,
                "Resultado": generated
            })
        df = pd.DataFrame(results)
        print(df)
        df.to_csv("./resources/results/resultados_validacion_test_set_shakespeare2.csv", index=False)


