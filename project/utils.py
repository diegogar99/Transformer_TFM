import torch
import math 
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

def evaluate_ppl(model, dataloader, loss_fn, device):
    model.eval() # Le dice al modelo que se comporte como inferencia
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader: 
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def lr_lambda(step):
    warmup = 1000
    return min((step+1)/warmup, 1.0)


def generate_text(model, sp, prompt, max_new_tokens=50, temperature=1.0, top_k=20, context_len=128, device="cpu"):

    model.eval()
    tokens = sp.encode(prompt, out_type=int)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        # Limita el contexto a los últimos context_len tokens
        idx_cond = tokens[:, -context_len:]
        
        # Pasa por el modelo
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature  # solo último token
        
        # Top-k sampling (opcional)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Añade nuevo token
        tokens = torch.cat([tokens, next_token], dim=1)
    
    return sp.decode(tokens[0].tolist())
