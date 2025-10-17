# LSTM para comparar con el Transformer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tokenization import *

print("GPU disponible:", tf.config.list_physical_devices('GPU'))

context_len = 128
embedding_dim = 256              

# https://sharmasaravanan.medium.com/text-generation-using-lstm-a-step-by-step-guide-9b787467f9de
def windowed_dataset(tokens, context_len, batch_size=64, shuffle=True): 
    n_windows = len(tokens) - context_len
    X = [tokens[i:i+context_len] for i in range(n_windows)]
    y = [tokens[i+context_len] for i in range(n_windows)] # LSTM recibe X: ["a","b","c"] y predice ["d"]

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def lstm_model(vocab_size, embedding_dim, context_len):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=context_len),
        #Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        # Dropout(0.2),
        LSTM(100,return_sequences=False),
        Dense(vocab_size, activation='softmax')
    ])
    return model

def validar_train(history):
    plt.figure(figsize=(10, 5))

    train_losses = np.array(history.history['loss'])
    val_losses = np.array(history.history['val_loss'])

    train_ppls = np.exp(train_losses)
    val_ppls = np.exp(val_losses)

    plt.plot(train_ppls, label="Train PPL")
    plt.plot(val_ppls, label="Validation PPL")
    plt.xlabel("Época")
    plt.ylabel("Perplexity")
    plt.title("Evolución de Perplexity durante el entrenamiento (LSTM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./resources/imagenes/resultado_entrenamiento_lstm.pdf')
    plt.close()

    print("=== Resultados del entrenamiento LSTM ===")
    for i, (loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        print(f"Época {i+1:03d}: Loss={loss:.4f} | Val_Loss={val_loss:.4f} | "
              f"PPL={np.exp(loss):.2f} | Val_PPL={np.exp(val_loss):.2f}")

    # Medias finales
    mean_train_loss = train_losses.mean()
    mean_val_loss = val_losses.mean()
    mean_train_ppl = np.exp(mean_train_loss)
    mean_val_ppl = np.exp(mean_val_loss)

    print("\n=== Promedios ===")
    print(f"Media Train Loss: {mean_train_loss:.4f}")
    print(f"Media Val Loss:   {mean_val_loss:.4f}")
    print(f"Media Train PPL:  {mean_train_ppl:.2f}")
    print(f"Media Val PPL:    {mean_val_ppl:.2f}")

    # Devolver si quieres usarlos luego
    return mean_train_loss, mean_val_loss, mean_train_ppl, mean_val_ppl 

def train_lstm(model,train_ds,val_ds):
    print("Comienza entrenamiento")
    early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        verbose=2,
        callbacks=[early_stopping]
    )
    print("Entrenamiento finalizado")
    print("Validación del entrenamiento")
    validar_train(history)
    model.save('./resources/models/lstm_model.keras')


#########################
# Limpieza y tokenización
#########################

train_text,valid_text,test_text = read_datasets()

print(f"Longitud corpus train: {len(train_text)} caracteres")
print(f"Longitud corpus valid: {len(valid_text)} caracteres")
print(f"Longitud corpus test: {len(test_text)} caracteres")

train_ids,sp = tokenizador(train_text)
val_ids,_ = tokenizador(valid_text)

vocab_size = sp.vocab_size()

#########################
# Ventanas deslizantes
#########################
print("Creando ventanas deslizantes...")
train_batches = windowed_dataset(train_ids, context_len)
val_batches = windowed_dataset(val_ids, context_len)

#########################
# Modelo
#########################
model = lstm_model(vocab_size, embedding_dim, context_len)

train_lstm(model,train_batches,val_batches)

'''model = Sequential([
    Conv1D(filters=128, 
           kernel_size=5, 
           strides=1, 
           padding="causal", 
           activation="relu", 
           input_shape=[window_size, 1]), 
    Dropout(0.1),
    LSTM(128, return_sequences=True),                          #kernel_regularizer=l2(0.01)
    Dropout(0.1),
    LSTM(64),
    Dropout(0.1),
    Dense(1, activation="relu")  
])'''
