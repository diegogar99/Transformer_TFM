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

DATASET = "wikitext2"

if DATASET != "wikitext2":
    embedding_dim = 256              
    context_len = 128
    num_epochs = 40
    patience_limit = 10 # Para early stopping
    num_layers = 3
    num_heads = 8
    d_ff = 1024
    dropout = 0.3
    vocab_size = 8000
    batchSize = 64
else:
    embedding_dim = 512               
    context_len = 256
    num_epochs = 40
    patience_limit = 10 # Para early stopping
    num_layers = 6
    num_heads = 16
    d_ff = 2048
    dropout = 0.2
    vocab_size = 10000
    batchSize = 64


'''
V1 Wikitext2
embedding_dim = 256               
context_len = 128
num_epochs = 40
patience_limit = 10 # Para early stopping
num_layers = 3
num_heads = 8
d_ff = 1024
dropout = 0.3
vocab_size = 10000
batchSize = 64

'''
device = "cuda" if torch.cuda.is_available() else "cpu"


if DATASET != "wikitext2":
    MODEL_PATH = "./resources/models/gpt_model.pth"
    BEST_MODEL_PATH = "./resources/models/best_gpt_model.pth"
else:
    MODEL_PATH = "./resources/models/gpt_model_wikitext2_v2.pth"
    BEST_MODEL_PATH = "./resources/models/best_gpt_model_wikitext2_v2.pth"

########################
# Lectura de datasets
########################

tinishakespeare,wikitext2 = load_data()

#########################
# Limpieza y tokenización
##########################

if DATASET != "wikitext2":
    pre_clean_dataset(tinishakespeare,DATASET)
else:
    pre_clean_dataset(wikitext2)

train_text,valid_text,test_text = read_datasets(data_name = DATASET)

print(f"Longitud corpus train: {len(train_text)} caracteres")
print(f"Longitud corpus valid: {len(valid_text)} caracteres")
print(f"Longitud corpus test: {len(test_text)} caracteres")

train_ids,sp = tokenizador(dataset=train_text, vocabsize=vocab_size)

val_ids,_ = tokenizador(valid_text, vocab_size, DATASET)

vocab_size = sp.vocab_size()

#########################
# Preparación de batches <input, target>
##########################

# Se formatiza como <input, target>
 
print("Preparando DataLoader")
dataset = LMWindowDataset(train_ids, context_len=context_len)  # o 256
val_dataset = LMWindowDataset(val_ids, context_len=context_len)  # o 256

loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)


########################
# Carga el modelo
########################

print("Se invoca el modelo")
model = miniLLM(
    vocab_size=vocab_size,
    d_model=embedding_dim,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    context_len=context_len,
    dropout=dropout
).to(device)

print("TRAIN")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


best_val_loss = float("inf")
best_loss = float("inf")
best_ppl = float("inf")

loss_fn = nn.CrossEntropyLoss()

#optimizer_sgd = torch.optim.SGD(model.parameters(), lr=1e-3)
#optimizer_rmsprop = torch.optim.RMSprop(model.parameters(),lr=1e-3, weight_decay=1e-2,momentum=0.9)

#optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(loader))

if DATASET != "wikitext2":
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
else:
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9) v1
    optimizer = torch.optim.RMSprop(model.parameters(),lr=1e-3, alpha=0.9,momentum=0.98,eps=1e-9) # v2

'''
- betas: Explica como se calculan los promedios moviles (de los valores recientes) de los gradientes y sus cuadrados. 
    - beta1: Promedio movil de los gradientes: cuanta memoria tiene de los grad anteriores. Al bajarlo, responde mas rapido a cambios. Lo que hace es mezclar un poco del gradiente actual con los anteriores (memoria), para que el cambio no sea tan brusco.
    - beta2: Promedio movil de los cuadrados de los gradientes. Como se ajusta el tamaño del lr. Lo que ahce es recordar como suelen ser los gradientes: si la pendiente es fuerte, reduce el lr para no saltar demasiado.Es al cuadrado para que no se anulen.
- eps: es un pequeño número que se añade al denominador para evitar divisiones por cero o números muy pequeños que causen inestabilidad numérica.
- weight_decay: Técnica de regulación para que no aprenda valores de pesos demasiado grandes. Es el valor que el optimizador reduce el peso en cada paso.
- momemtum: cuando los gradientes apuntan en la misma dirección varias veces el movimiento se acelera. En adam y adamW es beta1.
- alpha en RMSprop es como beta2 en adam.

'''

scheduler = LambdaLR(optimizer, lr_lambda)


patience_counter = 0
best_val_loss = float("inf") 

train_losses, val_losses = [], []
train_ppls, val_ppls = [], []

#scaler = torch.amp.GradScaler("cuda") 

'''for num_batch, (batch_x, batch_y) in enumerate(loader, start=1):
    batch_x, batch_y = batch_x.to(device), batch_y.to(device) # todevice mueve a GPU si está disponible, es necesario pues le modelo está en GPU también.
    print(f"Batch_x: {sp.decode(batch_x[0].tolist())}")
    print(f"Batch_y: {sp.decode(batch_y[0].tolist())}")
    print(batch_x.shape, batch_y.shape)
    break
exit()'''

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
    print(f"Iteracciones: {len(loader)}")
    for num_batch, (batch_x, batch_y) in enumerate(loader, start=1):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device) # todevice mueve a GPU si está disponible, es necesario pues le modelo está en GPU también.
      
        optimizer.zero_grad(set_to_none=True) # Limpia los gradientes previos. Pues en pytorch se acumulan por defecto y al llamar a backward se suman a los previos, es decir, se estarían combinando gradientes de varios batches. Lo pone a none para ahorrar memoria con set_to_none.

        #with torch.amp.autocast("cuda"): # Decide de forma segura que algunas operaciones se hagan en FP16 para ahorrar memoria y acelerar GPU
        logits = model(batch_x) # Salida del modelo [batch_size, seq_len, vocab_size]. Cada posición de la secuencia tiene una distribución sobre el vocabulario.
            # Devuelve, para cada token de entrada, un vector de tamaño vocab_size con valores reales. En PyTorch, nn.CrossEntropyLoss()  ya incluye internamente el softmax
        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            batch_y.view(-1) # Batch_y tiene: [batch_size, seq_len], esto lo aplana en [batch_size * seq_len] para cross entropy
        )
        loss.backward()
        #scaler.scale(loss).backward() # Escala y calcula el gradiente de los pesos del modelo (en que dirección cambio el peso para reducir la pérdida). Scale multiplica la pérdida por un factor grande antes de hacer el backward. Los pesos aún no cambian. 
        
        # Gradient Clipping para estabilidad [35]: limita el tamaño máximo que puede tener el conjunto de gradientes antes de actualizar los pesos.El clipping no cambia la dirección del gradiente (sigue apuntando hacia la misma mejora), solo reduce su magnitud para que no provoque saltos gigantes en los pesos.
        #scaler.unscale_(optimizer) # En AMP, los gradientes están temporalmente amplificados (por GradScaler).Se desescalan para no recortar valores falsos.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        

        #scaler.step(optimizer) # usa los gradientes calculados para ajustar los pesos. Sin GradScaler sería: optimizer.step(). El scaler primero desescala los gradientes si aún no se ha hecho (los divide por el mismo factor que usó en scale) y luego llama a optimizer.step().
        #scaler.update() # Se encarga de ajustar el factor de escalado de la pérdida para la próxima iteración.

        total_loss += loss.item()
   
        if num_batch % 100 == 0:
            #current_lr = scheduler.get_last_lr()[0]
            print(f"  [Batch {num_batch}] Loss: {loss.item():.4f}") # | LR: {current_lr:.6e}
    scheduler.step()
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
        best_loss = avg_loss
        best_ppl = train_ppl
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            print(f"Early stopping en epoch: {epoch+1}")
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
plt.savefig('./resources/imagenes/resultado_entrenamiento_wikitext2_v2.pdf')