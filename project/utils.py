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


def generate_text_test(model, sp, prompt, max_new_tokens=50, temperature=1.0, top_k=20, context_len=128, device="cpu"):

    model.eval()
    tokens = sp.encode(prompt, out_type=int)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # Añade nueva dimensión para que sea un lote
    
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
        next_token = torch.multinomial(probs, num_samples=1) # Escoge siguiente token según las probabilidades de softmax
        
        # Añade nuevo token
        tokens = torch.cat([tokens, next_token], dim=1)
    
    return sp.decode(tokens[0].tolist())


def generate_text(model, sp, init_text, top_k, top_p, presence_penalty,frequency_penalty, temperature, max_new_tokens,context_len,device):
        model.eval()
        tokens = sp.encode(init_text, out_type=int)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            init_text_ids = tokens[:, -context_len:]
            with torch.no_grad():
                logits = model(init_text_ids) # Predice siguiente token
                logits = logits[:, -1, :]
                
                if presence_penalty != 0.0 or frequency_penalty != 0.0: #https://medium.com/@pratik.vyas_10544/transformer-decoder-only-generative-model-part-3-0b38c10c2ae9

                    generated = tokens[0].tolist()  # lista de IDs generados hasta ahora
                    counts = torch.bincount(torch.tensor(generated, device=device), minlength=logits.size(-1)) # Según docu: Count the frequency of each value in an array of non-negative ints.

                    appeared = (counts > 0).float()
                    logits = logits - presence_penalty * appeared - frequency_penalty * counts.float()

                # Temperatura https://medium.com/@madasuvishnuraj/mathematics-behind-the-temperature-in-llm-cfb23120ac62
                if temperature != 0.0: # Se comporta como argmax
                    logits = logits / temperature
                    #logits = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) # Al ser multidim, del ultimo token coge de la ultima dim que son los scores del next token               
                
                if top_k is not None:
                    # Top k: Mantiene solo los k tokens con mayor probabilidad y esoce aleatoriamente https://docs.pytorch.org/docs/stable/generated/torch.topk.html
                    k_v, _ = torch.topk(logits, top_k, dim=-1) # Si no se da dim coge la ultima dim del tensor y devuelve los k elementos mas grandes. logits ya tiene formato [:-1,:]
                    logits[logits < k_v[:, [-1]]] = -torch.inf # A los que no aparecen en K_v (menores que el menor de k_v), les ponfo -inf para que no tengan probabilidad de ser elegidos https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b


                if top_p is not None:  # https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Elimina tokens con probabilidad acumulada mayor que top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # Desplaza a la derecha para mantener al menos un token: [...,1]= [:, :, 1:]
                    sorted_indices_to_remove[..., 0] = 0 # Se fuerza que el primer token, el mas probable, no sea eliminado
                    sorted_logits[sorted_indices_to_remove] = -torch.inf
                    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1)) # Ordena de nuevo a su orden original segun indices. 
                    
                
                prob_distrib = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(prob_distrib, num_samples=1)  # Escoge siguiente token según las probabilidades de softmax

            tokens = torch.cat([tokens, next_token], dim=1)

        return sp.decode(tokens[0].tolist())
# TODO SOBRE VALIDAR Y EJEMPLO AUTOREGRESION:  https://medium.com/@pratik.vyas_10544/transformer-decoder-only-generative-model-part-3-0b38c10c2ae9
