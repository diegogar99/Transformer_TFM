import torch
import math 
from torch.optim.lr_scheduler import LambdaLR

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
