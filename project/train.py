import os
import torch
import torch.nn as nn
import math
import torch.profiler as profiler # Ver en qué capas se consume más memoria GPU o tiempo.
import matplotlib.pyplot as plt
from tokenization import * 
from window_dataset import * 
from model.final_model import *
from utils import * 
print("GPU available:", torch.cuda.is_available())

embedding_dim = 512              
context_len = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./resources/models/gpt_model.pth"
########################
# Lectura de datasets
########################

tinishakespeare,wikitext2 = load_data()

#########################
# Limpieza y tokenización
##########################
pre_clean_dataset(tinishakespeare)

train_text,valid_text,test_text = read_datasets()

print(f"Longitud corpus train: {len(train_text)} caracteres")
print(f"Longitud corpus valid: {len(valid_text)} caracteres")
print(f"Longitud corpus test: {len(test_text)} caracteres")

train_ids,vocab_size = tokenizador(train_text)
val_ids,_ = tokenizador(valid_text)

#########################
# Preparación de batches <input, target>
##########################

# Se formatiza como <input, target>
 
print("Preparando DataLoader")
dataset = LMWindowDataset(train_ids, context_len=256)  # o 256
val_dataset = LMWindowDataset(val_ids, context_len=256)  # o 256

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)


########################
# Carga el modelo
########################

print("Se invoca el modelo")
model = miniGPT2(
    vocab_size=vocab_size,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    context_len=256,
    dropout=0.1
).to(device)

loss_fn = nn.CrossEntropyLoss()
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
        optimizer.zero_grad(set_to_none=True) # Limpia los gradientes previos. Pues en pytorch se acumulan por defecto y al llamar a backward se suman a los previos, es decir, se estarían combinando gradientes de varios batches. Lo pone a none para ahorrar memoria con set_to_none.

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
torch.save(model, MODEL_PATH)

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