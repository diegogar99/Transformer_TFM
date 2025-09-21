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


###################
# Lectura de datasets
###################

#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O tinyshakespeare.txt

if not os.path.exists("./models/shakespeare_clean_train.model"):
    dataset = load_dataset("text", data_files={"raw": "./datasets/tinyshakespeare.txt"})

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

    train_path = Path("./datasets/shakespeare_clean_train.txt")
    with train_path.open("w", encoding="utf-8") as f:
        for line in cleaned["train"]["text"]:
            if line.strip():
                f.write(line.strip() + "\n")


if not os.path.exists("./models/bpe_model_shakespeare.model"):
   # Aprende el vocabulario y como dividir en subpalabras
    spm.SentencePieceTrainer.Train(
        input="./datasets/shakespeare_clean_train.txt", # Corpus de train, se pueden pasar varios
        model_prefix="./models/bpe_model_shakespeare",        # generaprefijo modelos generados:  bpe_model_shakespeare.model y bpe_model_shakespeare.vocab
        vocab_size=16000,                 # 8k–32k para corpora pequeños, 32 k para wikitext2
        model_type="bpe",
        character_coverage=1.0,           # inglés
        byte_fallback=True,               # evita UNK en chars raros
        normalization_rule_name="nfkc",  # Normalización previa
        remove_extra_whitespaces=True, # colapsa espacios extra
        num_threads=os.cpu_count()
    )


# Carga el modelo con vocabulario, reglas de fusión y normalización. Con esto sabe como debe convertir el texto a IDs como en el train
sp = spm.SentencePieceProcessor(model_file="./models/bpe_model_shakespeare.model")

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

