from utils import * 
from tokenization import * 
from window_dataset import * 
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./resources/models/gpt_model.pth"

loss_fn = nn.CrossEntropyLoss()

_,_,test_text = read_datasets(True)


print(f"Longitud corpus test: {len(test_text)} caracteres")

test_ids,vocab_size = tokenizador(test_text)

test_dataset = LMWindowDataset(test_ids, context_len=256)  # o 256

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)

model = torch.load(MODEL_PATH, map_location="cpu")

test_loss, test_ppl = evaluate_ppl(model, test_loader, loss_fn, device)
print(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")