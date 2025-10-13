from utils import * 
from tokenization import * 
from window_dataset import * 
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./resources/models/gpt_model.pth"
TEST_SET = False
loss_fn = nn.CrossEntropyLoss()
context_len = 128

if TEST_SET:
    _,_,test_text = read_datasets(True)


    print(f"Longitud corpus test: {len(test_text)} caracteres")

    test_ids,vocab_size = tokenizador(test_text)

    test_dataset = LMWindowDataset(test_ids, context_len=context_len)  # o 256

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
    generated = generate_text(model, sp, prompt, max_new_tokens=100, temperature=0.8, device=device)
    print(generated)
