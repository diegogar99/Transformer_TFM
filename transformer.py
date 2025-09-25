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

###################
# Lectura de datasets
###################

#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O tinyshakespeare.txt

if not os.path.exists("./project/resources/models/shakespeare_clean_train.model"):
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

    train_path = Path("./project/resources/datasets/shakespeare_clean_train.txt")
    with train_path.open("w", encoding="utf-8") as f:
        for line in cleaned["train"]["text"]:
            if line.strip():
                f.write(line.strip() + "\n")


if not os.path.exists("./project/resources/models/bpe_model_shakespeare.model"):
   # Aprende el vocabulario y como dividir en subpalabras
    spm.SentencePieceTrainer.Train(
        input="./project/resources/datasets/shakespeare_clean_train.txt", # Corpus de train, se pueden pasar varios
        model_prefix="./project/resources/models/bpe_model_shakespeare",        # generaprefijo modelos generados:  bpe_model_shakespeare.model y bpe_model_shakespeare.vocab
        vocab_size=16000,                 # 8k–32k para corpora pequeños, 32 k para wikitext2
        model_type="bpe",
        character_coverage=1.0,           # inglés
        byte_fallback=True,               # evita UNK en chars raros
        normalization_rule_name="nfkc",  # Normalización previa
        remove_extra_whitespaces=True, # colapsa espacios extra
        num_threads=os.cpu_count(),
        pad_id=0, unk_id=1, bos_id=2, eos_id=3

    )


# Carga el modelo con vocabulario, reglas de fusión y normalización. Con esto sabe como debe convertir el texto a IDs como en el train
sp = spm.SentencePieceProcessor(model_file="./project/resources/models/bpe_model_shakespeare.model")

# Tokenizaciones 
def sp_encode_batch_train(batch):
    ids = [
        [sp.bos_id()] +
        sp.encode(t, out_type=int, enable_sampling=True, nbest_size=-1, alpha=0.1) +
        [sp.eos_id()]
        for t in batch["text"]
    ]
    attn = [[1]*len(x) for x in ids]
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

print(tokenized_train)
print(tokenized_train["input_ids"])         # Gets the whole column
print(tokenized_train["attention_mask"])    # Gets the whole column

print(tokenized_train.shape)

# Aplano el resultado
stream = []
for example in tokenized_train:
    stream.extend(example["input_ids"])

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

print("GPU available:", torch.cuda.is_available())
dataset = LMWindowDataset(stream, context_len=256)  # o 256
loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,num_workers=os.cpu_count(),pin_memory=torch.cuda.is_available(),prefetch_factor=2)


####################
# Embeddings
#####################

vocab_size = sp.vocab_size()     
embedding_dim = 512              
context_len = 256

# Token embeddings
embedding_layer = nn.Embedding(
    num_embeddings=vocab_size, 
    embedding_dim=embedding_dim, 
    padding_idx=sp.pad_id() if sp.pad_id() >= 0 else None
)

# Positional embeddings aprendidos
pos_embedding_layer = nn.Embedding(
    num_embeddings=context_len,  
    embedding_dim=embedding_dim
)

for batch_x, batch_y in loader:
    B, T = batch_x.shape                      
    tok_emb = embedding_layer(batch_x)       

    pos_ids = torch.arange(T, device=batch_x.device).unsqueeze(0).expand(B, T)
    pos_emb = pos_embedding_layer(pos_ids)   

    output_embeddings = tok_emb + pos_emb     

    print(output_embeddings.shape)           
    print(output_embeddings)
    break
    '''
    for layer in 1..N:
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