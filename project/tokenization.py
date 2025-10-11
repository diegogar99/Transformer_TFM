from datasets import load_dataset
import re
import ftfy
import unicodedata
from pathlib import Path
import sentencepiece as spm
import os
from datasets import DatasetDict

TOK_MODEL_PATH = "./resources/models/bpe_model_shakespeare.model"
DATASET_PATH = "./resources/datasets/tinyshakespeare.txt"
train_path = Path("./resources/datasets/shakespeare_clean_train.txt")
test_path = Path("./resources/datasets/shakespeare_clean_test.txt")
valid_path = Path("./resources/datasets/shakespeare_clean_validation.txt")

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

def load_data():
    print("Carga datasets")
    print("\tTiny Shakespeare")
    dataset = load_dataset("text", data_files={"raw": DATASET_PATH})

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

    print("\tWikitext-2")
    wikitext2 = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    return tinishakespeare,wikitext2
    

def pre_clean_dataset(dataset):
    cleaned = DatasetDict({
        'train': dataset['train'].map(light_clean_fn, num_proc=4),
        'validation': dataset['validation'].map(light_clean_fn, num_proc=4),
        'test': dataset['test'].map(light_clean_fn, num_proc=4),
    })
    
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

def read_datasets(only_test = False):
    
    if only_test:
        with test_path.open("r", encoding="utf-8") as f:
            test_text = f.read()
        return None,None, test_text
    
    with train_path.open("r", encoding="utf-8") as f:
        train_text = f.read()

    with valid_path.open("r", encoding="utf-8") as f:
        valid_text = f.read()

    with test_path.open("r", encoding="utf-8") as f:
        test_text = f.read()

    return train_text,valid_text,test_text


def tokenizador(dataset):
    if not os.path.exists(TOK_MODEL_PATH):
        print("Entrenamiento modelo BPE")
        # Aprende el vocabulario y como dividir en subpalabras
        spm.SentencePieceTrainer.Train(
            input=train_path, # Corpus de train, se pueden pasar varios
            model_prefix="./resources/models/bpe_model_shakespeare",        # genera prefijo modelos generados:  bpe_model_shakespeare.model y bpe_model_shakespeare.vocab
            vocab_size=10000,                 # 8k–32k para corpora pequeños, 32 k para wikitext2
            model_type="bpe",
            character_coverage=1.0,           # inglés
            byte_fallback=True,               # evita UNK en chars raros
            normalization_rule_name="nfkc",  # Normalización previa
            remove_extra_whitespaces=True, # colapsa espacios extra
            num_threads=os.cpu_count(),
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )
    sp = spm.SentencePieceProcessor(model_file=TOK_MODEL_PATH)
    print("Tokenización del corpus de train, test y validation")
    tok_ids = sp.encode(dataset, out_type=int)
    print("Número total de tokens en el corpus:", len(tok_ids))
    return tok_ids, sp.vocab_size()     

