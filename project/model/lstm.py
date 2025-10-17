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

def windowed_dataset(ids, context_len, batch_size=64, shuffle=True): # https://sharmasaravanan.medium.com/text-generation-using-lstm-a-step-by-step-guide-9b787467f9de
    input_sequences = []
    for line in text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
    # Pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    # Split sequences into input (X) and output (y)
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    return X,y

def lstm_model(vocab_size, embedding_dim, context_len):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=context_len-1),
        LSTM(100,return_sequences=False),
        Dense(vocab_size, activation='softmax')
    ])
    return model

def validar_train(history):
    plt.figure(figsize=(10, 5))

    train_ppls = np.exp(history.history['loss'])
    val_ppls = np.exp(history.history['val_loss'])

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

def train_lstm(model,train_ds,val_ds):
    early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(
        train_ds,
        epochs=100,
        validation_data=val_ds,
        callbacks=[early_stopping]
    )
    validar_train(history)
    model.save('./Models/lstm_model.keras')



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
