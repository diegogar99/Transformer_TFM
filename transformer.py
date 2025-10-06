from datasets import load_dataset
from multiprocessing import Pool
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from spellchecker import SpellChecker
import ftfy
import unicodedata
from pathlib import Path
import sentencepiece as spm
import os
from datasets import DatasetDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.profiler as profiler # Ver en qué capas se consume más memoria GPU o tiempo.
import matplotlib.pyplot as plt

###################
# Lectura de datasets
###################

#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O tinyshakespeare.txt
train_path = Path("./project/resources/datasets/shakespeare_clean_train.txt")
test_path = Path("./project/resources/datasets/shakespeare_clean_test.txt")
valid_path = Path("./project/resources/datasets/shakespeare_clean_validation.txt")

MODEL_PATH = "./project/resources/models/bpe_model_shakespeare.model"

if not os.path.exists("./project/resources/models/shakespeare_clean_train.model"):
    print("Carga datasets")
    dataset = load_dataset("text", data_files={"raw": "./project/resources/datasets/tinyshakespeare.txt"})

    # Dividir en train (90%) y test (10%)
    train_test = dataset["raw"].train_test_split(test_size=0.1, seed=42)

    # Dividir train en train (90%) y validation (10%)
    train_valid = train_test["train"].train_test_split(test_size=0.1, seed=42)

    # Reunir en un DatasetDict
    tinishakespeare = {
        "train": train_valid["train"],
        "validation": train_valid["test"],
        "test": train_test["test"]
    }



    wikitext2 = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")


    ####################
    # Limpieza y tokenización
    #####################
    print("Limpieza previa")
    def light_clean_fn(example):
        t = example["text"]
        t = ftfy.fix_text(t)
        t = unicodedata.normalize("NFKC", t)
        t = (t.replace("“", '"').replace("”", '"').replace("’", "'")
            .replace("–", "-").replace("—", "-"))
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\s*\n\s*", "\n", t)
        t = t.strip(" \n")
        return {"text": t}

    cleaned = DatasetDict({
        'train': tinishakespeare['train'].map(light_clean_fn, num_proc=4),
        'validation': tinishakespeare['validation'].map(light_clean_fn, num_proc=4),
        'test': tinishakespeare['test'].map(light_clean_fn, num_proc=4),
    })

    # Escribimos SOLO el train a un archivo para entrenar SP

    with train_path.open("w", encoding="utf-8") as f:
        for line in cleaned["train"]["text"]:
            if line.strip():
                f.write(line.strip() + "\n")

    with valid_path.open("w", encoding="utf-8") as f:
        for line in cleaned["validation"]["text"]:
            if line.strip():
                f.write(line.strip() + "\n")

    with test_path.open("w", encoding="utf-8") as f:
        for line in cleaned["test"]["text"]:
            if line.strip():
                f.write(line.strip() + "\n")

if not os.path.exists("./project/resources/models/bpe_model_shakespeare.model"):
    print("Entrenamiento modelo BPE")
   # Aprende el vocabulario y como dividir en subpalabras
    spm.SentencePieceTrainer.Train(
        input=train_path, # Corpus de train, se pueden pasar varios
        model_prefix="./project/resources/models/bpe_model_shakespeare",        # genera prefijo modelos generados:  bpe_model_shakespeare.model y bpe_model_shakespeare.vocab
        vocab_size=16000,                 # 8k–32k para corpora pequeños, 32 k para wikitext2
        model_type="bpe",
        character_coverage=1.0,           # inglés
        byte_fallback=True,               # evita UNK en chars raros
        normalization_rule_name="nfkc",  # Normalización previa
        remove_extra_whitespaces=True, # colapsa espacios extra
        num_threads=os.cpu_count(),
        pad_id=0, unk_id=1, bos_id=2, eos_id=3

    )

# Cargo el texto
print("Carga dataset train")
with open(train_path, "r", encoding="utf-8") as f:
    train_text = f.read()
print(f"Longitud corpus train: {len(train_text)} caracteres")

print("Carga dataset validation")
with open(valid_path, "r", encoding="utf-8") as f:
    valid_text = f.read()
print(f"Longitud corpus validation: {len(valid_text)} caracteres")

print("Carga dataset test")
with open(test_path, "r", encoding="utf-8") as f:
    test_text = f.read()

print(f"Longitud corpus test: {len(test_text)} caracteres")

# Carga el modelo con vocabulario, reglas de fusión y normalización. Con esto sabe como debe convertir el texto a IDs como en el train
sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)

# Tokenizaciones 
print("Tokenización del corpus de train, test y validation")
train_ids = sp.encode(train_text, out_type=int)
val_ids = sp.encode(valid_text, out_type=int)
test_ids = sp.encode(test_text, out_type=int)
print("Número total de tokens en el corpus train:", len(train_ids))
print("Número total de tokens en el corpus test:", len(test_ids))
print("Número total de tokens en el corpus validation:", len(val_ids))

# Se formatiza como <input, target>

class LMWindowDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, context_len):
        self.toks = tokens
        self.ctx = context_len
    def __len__(self):
        return len(self.toks) - self.ctx # number of windows
    def __getitem__(self, i): # inputs:targets
        x = torch.tensor(self.toks[i:i+self.ctx], dtype=torch.long)
        y = torch.tensor(self.toks[i+1:i+self.ctx+1], dtype=torch.long)
        return x, y
    
print("Preparando DataLoader")
print("GPU available:", torch.cuda.is_available())
dataset = LMWindowDataset(train_ids, context_len=256)  # o 256
val_dataset = LMWindowDataset(val_ids, context_len=256)  # o 256
test_dataset = LMWindowDataset(test_ids, context_len=256)  # o 256

x, y = dataset[0]
print(len(dataset))
print("x shape:", x.shape, "y shape:", y.shape)
print("x[:10] =", x[:10])
print("y[:10] =", y[:10])

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)

####################
# Embeddings
#####################
print("Embeddings")
vocab_size = sp.vocab_size()     
embedding_dim = 512              
context_len = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

# Token embeddings
embedding_layer = nn.Embedding(
    num_embeddings=vocab_size, 
    embedding_dim=embedding_dim
).to(device) 

# Positional embeddings aprendidos
pos_embedding_layer = nn.Embedding(
    num_embeddings=context_len,  
    embedding_dim=embedding_dim
).to(device) 


print("Multi-Head Attention")
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
       
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # dimensión por cabeza
        
        # Proyecciones lineales para Q, K, V
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        
        # Proyección final después de concatenar todas las cabezas
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x): # Que pasa con los datos al llamar al módulo 
        """
        x: (B, T, d_model)
        Devuelve: (B, T, d_model)
        """
        B, T, _ = x.size()
        
        # 1. Proyecciones lineales
        Q = self.Wq(x)  # (B, T, d_model)
        K = self.Wk(x)  # (B, T, d_model)
        V = self.Wv(x)  # (B, T, d_model)
        
        # 2. Reorganizar en múltiples cabezas
        # (B, T, d_model) -> (B, num_heads, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Calcular scores de atención
        # (B, num_heads, T, d_k) x (B, num_heads, d_k, T) -> (B, num_heads, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Máscara causal (triangular superior a -inf)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        # 5. Normalizar con softmax
        with torch.amp.autocast(enabled=False):
            scores = scores.float()
            attn = torch.softmax(scores, dim=-1)  # (B, num_heads, T, T)
        
        # 6. Aplicar atención a V
        out = torch.matmul(attn, V)  # (B, num_heads, T, d_k)
        
        # 7. Recombinar heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)
        
        # 8. Proyección final
        out = self.Wo(out)  # (B, T, d_model)
        return out
    


# Add & norm

print("Add & Norm")

class NormLayer(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, X):
        # X: (batch, seq_len, hidden_dim)
        with torch.amp.autocast(enabled=False): # Desactivo mixed precision manualmente pues es recomendable en la LayerNorm y como la he implementado yo, quiza no lo detecta automaticamente (solo reconoce capas nativas de pytorch) [35]
            X = X.float()
            mean = X.mean(dim=-1, keepdim=True)       # media por posición
            var = X.var(dim=-1, keepdim=True, unbiased=False)  # varianza
            X_hat = (X - mean) / torch.sqrt(var + self.eps)    # normaliza
        return self.gamma * X_hat + self.beta


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout) # Para regularizar
        self.ln = NormLayer(norm_shape) # nn.LayerNorm(norm_shape)

    def forward(self, X, Y): # Y es la salida de la subcapa previa y X la entrada a la subcapa
        return self.ln(self.dropout(Y) + X) # Aplica add y luego layernorm

# FFNN
print("FFNN")

class FeedForward(nn.Module):
    def __init__(self,d_model, d_ff, dropout):
        super().__init__()
        # Se usan 3 capas densas, con dropout y activación GELU
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_ff)
        self.linear3 = nn.Linear(d_ff, d_model) # Ver si estas capas expanden y contraen
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
       x = self.linear1(x)
       x = self.gelu(x)
       x = self.dropout(x)
       x = self.linear2(x)
       x = self.gelu(x)
       x = self.dropout(x)
       x = self.linear3(x)
       return x

# Bloque 1 transformer decoder-only
print("Bloque 1 decoder only")

class TransformerDecoderOnlyBlock(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.apply(self._init_weights) # Recorre las capas aplicando la función

    def _init_weights(self, m): # Inicialización de pesos: Xavier para MHA y FFNN y normal para embeddings
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias) # Para evitar desplazamiento inicial arbitrario
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self,x):
        attention = self.mha(x)
        x = self.addnorm1(x, attention)
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)
        return x

print("Modelo completo")

class miniGPT2(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, context_len=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerDecoderOnlyBlock(num_heads, d_model, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = NormLayer(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

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


print("Se invoca el modelo")
model = miniGPT2(
    vocab_size=sp.vocab_size(),
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    context_len=256,
    dropout=0.1
).to(device)

print("Loss function")
loss_fn = nn.CrossEntropyLoss()
print("Optimizer")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("TRAIN")
num_epochs = 20
best_val_loss = float("inf")


best_loss = float("inf")
best_ppl = float("inf")

patience_limit = 5 # Para early stopping
patience_counter = 0
best_val_loss = float("inf")


train_losses, val_losses = [], []
train_ppls, val_ppls = [], []


'''
Entrenar todo en float32 (FP32) significa más memoria y más tiempo de cómputo.
La RTX 4090 tiene Tensor Cores que aceleran mucho con FP16 o bfloat16.
'''

scaler = torch.amp.GradScaler("cuda") 

for epoch in range(num_epochs):
    '''
    Solo para validar en uno o pocos batches
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
    '''
    model.train()
    total_loss = 0.0
    torch.cuda.empty_cache()            
    batch_num = 0
    print(f"Iteracciones: {len(loader)}")
    for num_batch, (batch_x, batch_y) in enumerate(loader, start=1):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device) # todevice mueve a GPU si está disponible, es necesario pues le modelo está en GPU también.
        batch_num += 1
        optimizer.zero_grad(set_to_none=True) # Limpia los gradientes previos. Pues en pytorch se acumulan por defecto y al llamar a bckward se suman a los previos, es decir, se estarían combinando gradientes de varios batches. Lo pone a none para ahorrar memoria con set_to_none.

        with torch.amp.autocast("cuda"): # Decide de forma segura que algunas operaciones se hagan en FP16 para ahorrar memoria y acelerar GPU
            logits = model(batch_x) # Salida del modelo [batch_size, seq_len, vocab_size]. Cada posición de la secuencia tiene una distribución sobre el vocabulario.
            # Devuelve, para cada token de entrada, un vector de tamaño vocab_size con valores reales. En PyTorch, nn.CrossEntropyLoss()  ya incluye internamente el softmax
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                batch_y.view(-1) # Batch_y tiene: [batch_size, seq_len], esto lo aplana en [batch_size * seq_len] para cross entropy
            )

        if torch.isnan(loss): # Al usar mixed precision puede haber underflow o overflow y dar NaN en la pérdida [35]
            print(f"[NaN] Detectado en batch {num_batch}, reduciendo LR")
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
            continue

        scaler.scale(loss).backward() # Escala y calcula el gradiente de los pesos del modelo (en que dirección cambio el peso para reducir la pérdida). Scale multiplica la pérdida por un factor grande antes de hacer el backward. Los pesos aún no cambian. 
        
        # Gradient Clipping para estabilidad [35]: limita el tamaño máximo que puede tener el conjunto de gradientes antes de actualizar los pesos.El clipping no cambia la dirección del gradiente (sigue apuntando hacia la misma mejora), solo reduce su magnitud para que no provoque saltos gigantes en los pesos.
        scaler.unscale_(optimizer) # En AMP, los gradientes están temporalmente amplificados (por GradScaler).Se desescalan para no recortar valores falsos.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer) # usa los gradientes calculados para ajustar los pesos. Sin GradScaler sería: optimizer.step(). El scaler primero desescala los gradientes si aún no se ha hecho (los divide por el mismo factor que usó en scale) y luego llama a optimizer.step().
        scaler.update() # Se encarga de ajustar el factor de escalado de la pérdida para la próxima iteración.

        total_loss += loss.item()

        if num_batch % 100 == 0:
            print(f"  [Batch {num_batch}] Loss: {loss.item():.4f}")
            for name, param in model.named_parameters(): # Estadísticas para detectar underflow en train en los gradientes
                if param.grad is not None:
                    grad_min = param.grad.abs().min().item()
                    grad_max = param.grad.abs().max().item()
                    if grad_min == 0.0:
                        print(f"[Underflow] Gradiente nulo en {name}")
                    elif grad_min < 1e-8:
                        print(f"[Pequeño] {name}: grad_min={grad_min:.2e}, grad_max={grad_max:.2e}")

        

    avg_loss = total_loss / len(loader)
    train_ppl = math.exp(avg_loss) # Perplejidad: Si baja es que comprende mejor los datos. Es el exponente de la entropía
    # Validación para implementar early stopping y guardar mejor modelo

    val_loss, val_ppl = evaluate_ppl(model, val_loader, loss_fn, device)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")
    train_losses.append(avg_loss)
    val_losses.append(val_loss)
    train_ppls.append(train_ppl)
    val_ppls.append(val_ppl)


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, "./best_minigpt2.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("Entrenamiento finalizado.")
print(f"Best Train Loss: {best_loss:.4f} | Best Train PPL: {best_ppl:.2f}")

plt.figure(figsize=(10,5))
plt.plot(train_ppls, label="Train PPL")
plt.plot(val_ppls, label="Validation PPL")
plt.xlabel("Época")
plt.ylabel("Perplexity")
plt.title("Evolución de Perplexity durante entrenamiento")
plt.legend()
plt.grid(True)
plt.show()

##################################
# VALIDACIÓN EFICIENTE (sin gradientes)
##################################

model.load_state_dict(torch.load("./best_minigpt2.pth"))
model.to(device)
print("Modelo recargado con los mejores pesos guardados.")

test_loss, test_ppl = evaluate_ppl(model, test_loader, loss_fn, device)
print(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")


'''print("VALIDATION (no_grad)") [35]

model.eval()
val_loss = 0.0

with torch.no_grad():
    for batch_x, batch_y in val_loader:  # o val_loader si tienes dataset separado
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        with torch.cuda.amp.autocast("cuda"):
            logits = model(batch_x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        val_loss += loss.item()

val_loss /= len(loader)
print(f"Validation Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}")
'''
###############################


'''

with torch.no_grad():
    model.eval()
    total_val_loss = 0.0
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        with torch.cuda.amp.autocast("cuda"):
            logits = model(batch_x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        total_val_loss += loss.item()

        
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, context_len=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

'''

'''
Entrenamiento: No aplicas softmax → usas CrossEntropyLoss directamente.

Generación: Sí aplicas softmax al último token → obtienes una distribución de probabilidad para muestrear el siguiente.
'''

# REVISAR: https://medium.com/@aisagescribe/pre-normalization-vs-post-normalization-in-transformers-e84872e0a3cd

# Esto es para debug
'''for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)
    B, T = batch_x.shape

    # 1. Token + Positional embeddings
    tok_emb = embedding_layer(batch_x)  # (B, T, d_model)
    pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    pos_emb = pos_embedding_layer(pos_ids)  # (B, T, d_model)
    x = tok_emb + pos_emb  # (B, T, d_model)

    # 2. Multi-Head Attention con máscara causal
    out = mha(x)  # (B, T, d_model)

    print("Input embeddings:", x.shape)
    print("Output atención:", out.shape)
    break
'''


    
'''
    torch.Size([32, 256, 512])
    torch.Size([32, 256, 512])
    torch.Size([32, 256, 512])
    '''
'''
    for layer in 1..N: # Meter la conexión residual
        # (Pre-Norm) Self-Attention enmascarada
        a = SelfAttention( LayerNorm(x), mask=causal(T) )   # [B, T, D]
        x = x + a                                           # residual

        # (Pre-Norm) Feed-Forward
        f = FeedForward( LayerNorm(x) )                     # [B, T, D]
        x = x + f                                           # residual

    # Proyección a vocabulario
    logits = Linear(x)                           # [B, T, vocab_size]

    # Pérdida (teacher forcing: batch_y es x desplazado +1)
    loss = CrossEntropy( logits.view(-1, vocab_size),
                         batch_y.view(-1) )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    '''

'''with open("embeddings.pt", "wb") as f:
    for batch_x, _ in loader:
        tok_emb = embedding_layer(batch_x)
        pos_ids = torch.arange(batch_x.size(1), device=batch_x.device).unsqueeze(0).expand_as(batch_x)
        pos_emb = pos_embedding_layer(pos_ids)
        out = (tok_emb + pos_emb).cpu()
        torch.save(out, f)
'''

'''


# Masks
def padding_mask(seq, pad_token=0):
    mask = (seq == pad_token).unsqueeze(1).unsqueeze(2)
    return mask  # (batch_size, 1, 1, seq_len)

def sequence_mask(seq):
    seq_len = seq.size(1)
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
    return mask  # (seq_len, seq_len)

def look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask  # (seq_len, seq_len)

# Masked self attention
def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size()[-1]
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()

        original_size_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(original_size_attention)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
    
##########################################################
# SELF ATTENTION: Un unico mecanismo de atención, solo aprende un tipo de relación por así decirlo. Por 
# ejemplo, "Fui al banco a retirar dinero" y "Pase la tarde sentado en un banco del parque" con self attention es más difícil que capture la diferencia, 
# mientras que con multihead puede aprender una cabeza para cada significado.
##########################################################
class SelfAttention:
    def __init__(self, embedding_dim):
        
        
        torch.manual_seed(42)
        self.embedding_dim = embedding_dim

        # Initialize weight matrices
        self.W_q = torch.randn(embedding_dim, embedding_dim)
        self.W_k = torch.randn(embedding_dim, embedding_dim)
        self.W_v = torch.randn(embedding_dim, embedding_dim)

    def forward(self, embeddings):
    
        # Compute Query, Key, and Value matrices
        Q = torch.matmul(embeddings, self.W_q)  
        K = torch.matmul(embeddings, self.W_k)  
        V = torch.matmul(embeddings, self.W_v)  

        # Compute similarity (dot product attention)
        similarity_matrix = torch.matmul(Q, K.T)  # (num_words, num_words)

        # Scale by sqrt(embedding_dim)
        similarity_matrix_scaled = similarity_matrix / torch.sqrt(
            torch.tensor(self.embedding_dim, dtype=torch.float32)
        )

        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarity_matrix_scaled, dim=1)

      
        new_context = torch.matmul(attention_weights, V)  

        return new_context, attention_weights
############################################################################################
#    MULTIHEAD
############################################################################################

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.d_k = embedding_dim // num_heads
        
 
        self.Wq = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim))
        self.Wk = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim))
        self.Wv = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim))
    
    def forward(self, embeddings):
        seq_len = embeddings.size(0)
        
        
        Q = embeddings @ self.Wq  # (seq_len, embedding_dim)
        K = embeddings @ self.Wk  
        V = embeddings @ self.Wv  

        # Converting to multiheaded attention
        Q = Q.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # (num_heads, seq_len, d_k)
        K = K.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  
        V = V.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (num_heads, seq_len, seq_len)

        # Apply mask (upper triangular mask for causal attention)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand_as(attention_scores)
        attention_scores = attention_scores.masked_fill(mask, -1e11)

        # Scale the attention scores
        attention_scores = attention_scores / math.sqrt(self.d_k)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (num_heads, seq_len, seq_len)

        # Compute the output (weighted sum of values)
        output = torch.matmul(attention_weights, V)  

       
        output = output.transpose(0, 1).contiguous().view(seq_len, self.embedding_dim)
        return output

#############################################################################################
'''


'''
def sp_encode_batch_train(batch):
    ids = [
        [sp.bos_id()] + # No lo necesito si aplano y LMWindowDataset
        sp.encode(t, out_type=int, enable_sampling=True, nbest_size=-1, alpha=0.1) + # Convierte a ids el texto y añade aleatoriedad
        [sp.eos_id()]
        for t in batch["text"]
    ]
    attn = [[1]*len(x) for x in ids] # Innecesaria pues uso ventanas fijas sin padding
    return {"input_ids": ids, "attention_mask": attn}

def sp_encode_batch_eval(batch):
    ids = [
        [sp.bos_id()] + sp.encode(t, out_type=int) + [sp.eos_id()]
        for t in batch["text"]
    ]
    attn = [[1]*len(x) for x in ids]
    return {"input_ids": ids, "attention_mask": attn}

tokenized_train = cleaned["train"].map(sp_encode_batch_train, batched=True, num_proc=4, remove_columns=["text"])
tokenized_val   = cleaned["validation"].map(sp_encode_batch_eval, batched=True, num_proc=4, remove_columns=["text"])
tokenized_test  = cleaned["test"].map(sp_encode_batch_eval, batched=True, num_proc=4, remove_columns=["text"])


'''