from datasets import load_dataset
import re
import ftfy
import unicodedata
from pathlib import Path
import sentencepiece as spm
import os
from datasets import DatasetDict
from clean_text import clean_wikitext2, light_clean_fn

TOK_MODEL_PATH_SHAKE = "./resources/models/bpe_model_shakespeare.model"
DATASET_PATH_SHAKE = "./resources/datasets/tinyshakespeare.txt"
train_path_shake = Path("./resources/datasets/shakespeare_clean_train.txt")
test_path_shake = Path("./resources/datasets/shakespeare_clean_test.txt")
valid_path_shake = Path("./resources/datasets/shakespeare_clean_validation.txt")


TOK_MODEL_PATH_WIKI = "./resources/models/bpe_model_wikitext.model"
DATASET_PATH_WIKI = "./resources/datasets/wikitext.txt"
train_path_wiki = Path("./resources/datasets/wikitext_clean_train.txt")
test_path_wiki = Path("./resources/datasets/wikitext_clean_test.txt")
valid_path_wiki = Path("./resources/datasets/wikitext_clean_validation.txt")

'''def light_clean_fn(texto):
    t = texto["text"]
    t = ftfy.fix_text(t)
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    t = t.strip(" \n")
    return {"text": t}'''


'''def clean_wikitext2(texto):
    t = texto["text"]
    t = ftfy.fix_text(t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"(?s)\(\s*Op\s*\.\s*\d+.*?\)","",t)
    t = re.sub(r"(?s)Op\s*\.\s*\d+.*?\n","",t)
    t = re.sub(r"\s\s*", " ", t)
    t = re.sub(r"=\s*(=\s*)*.*=\s*(=\s*)*\n", "", t)
    t = re.sub(r"@(.*?)@", r"\1", t)

    t = t.replace("*", "")
    t = t.replace(r"^(?=.*\b\d+\s*\*?\s*[A-Z][a-z]?\b)(?=.*(→|->|=>|\+))(?=.*\b(MeV|keV)\b).*?$", "", t)
    t = t.replace("→", "->")
    t = clean_unidades(t)
    t = clean_symbols(t)

    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-").replace("µm", "micrometers").replace("☉","")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    t = t.strip(" \n")
    t = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s.,;:!?'\"(){}\[\]\-+=\/%°<>_]", "", t)

    return {"text": t}
'''


def load_data():
    print("Carga datasets")
    print("\tTiny Shakespeare")

    dataset = load_dataset("text", data_files={"raw": DATASET_PATH_SHAKE})

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
    

def pre_clean_dataset(dataset, data_name="wikitext2"):

    if data_name == "wikitext2":
        
        cleaned = DatasetDict({
            'train': dataset['train'].map(clean_wikitext2, num_proc=4),
            'validation': dataset['validation'].map(clean_wikitext2, num_proc=4),
            'test': dataset['test'].map(clean_wikitext2, num_proc=4),
        })
    else:
        cleaned = DatasetDict({
            'train': dataset['train'].map(light_clean_fn, num_proc=4),
            'validation': dataset['validation'].map(light_clean_fn, num_proc=4),
            'test': dataset['test'].map(light_clean_fn, num_proc=4),
        })
    
    if data_name == "wikitext2":
        train_path = train_path_wiki
        valid_path = valid_path_wiki
        test_path = test_path_wiki
    else:
        train_path = train_path_shake
        valid_path = valid_path_shake
        test_path = test_path_shake

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

def read_datasets(only_test = False, data_name = "wikitext2"):
    
    if data_name == "wikitext2":
        train_path = train_path_wiki
        valid_path = valid_path_wiki
        test_path = test_path_wiki
    else:
        train_path = train_path_shake
        valid_path = valid_path_shake
        test_path = test_path_shake

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


def tokenizador(dataset, vocabsize, data_name = "wikitext2"):
    if data_name == "wikitext2":
        train_path = train_path_wiki
        TOK_MODEL_PATH = TOK_MODEL_PATH_WIKI
    else:
        train_path = train_path_shake
        TOK_MODEL_PATH = TOK_MODEL_PATH_SHAKE
  
    if not os.path.exists(TOK_MODEL_PATH):
        print("Entrenamiento modelo BPE")
        # Aprende el vocabulario y como dividir en subpalabras
        spm.SentencePieceTrainer.Train(
            input=train_path, # Corpus de train, se pueden pasar varios
            model_prefix="./resources/models/bpe_model_wikitext",        
            vocab_size= vocabsize, #8000,                 
            model_type="bpe",
            character_coverage=1.0,           
            byte_fallback=True,               # evita UNK en caracteres raros
            normalization_rule_name="nfkc",  # Normalización previa
            remove_extra_whitespaces=True, # colapsa espacios extra
            num_threads=os.cpu_count(),
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )
    sp = spm.SentencePieceProcessor(model_file=TOK_MODEL_PATH)
    tok_ids = sp.encode(dataset, out_type=int)
    print("Número total de tokens:", len(tok_ids))
    return tok_ids, sp


'''def clean_unidades(texto):
    texto = re.sub(r"\b(\d+)\s*[µμ]m(?![a-zA-Z])", r"\1 micrometers", texto)
    texto = re.sub(r"\b(\d+)\s*mm(?![a-zA-Z])", r"\1 millimeters", texto)
    texto = re.sub(r"\b(\d+)\s*cm(?![a-zA-Z])", r"\1 centimeters", texto)
    texto = re.sub(r"\b(\d+)\s*m(?![a-zA-Z])", r"\1 meters", texto)
    texto = re.sub(r"\b(\d+)\s*km(?![a-zA-Z])", r"\1 kilometers", texto)

    texto = re.sub(r"\b(\d+)\s*mg(?![a-zA-Z])", r"\1 milligrams", texto)
    texto = re.sub(r"\b(\d+)\s*[µμ]g(?![a-zA-Z])", r"\1 micrograms", texto)
    texto = re.sub(r"\b(\d+)\s*g(?![a-zA-Z])", r"\1 grams", texto)
    texto = texto.replace("kg", "kilograms")

    texto = re.sub(r"\b(\d+)\s*s(?![a-zA-Z])", r"\1 seconds", texto)
    texto = re.sub(r"\b(\d+)\s*h(?![a-zA-Z])", r"\1 hours", texto)
    texto = re.sub(r"\b(\d+)\s*min(?![a-zA-Z])", r"\1 minutes", texto)
    texto = re.sub(r"\b(\d+)\s*ms(?![a-zA-Z])", r"\1 milliseconds", texto)

    texto = texto.replace("°C", " degrees Celsius")
    texto = re.sub(r"\b(\d+)\s*Ks(?![a-zA-Z])", r"\1 kelvin", texto)
    texto = texto.replace("°F", " degrees Fahrenheit")

    texto = texto.replace("m/s", "meters per second")
    texto = texto.replace("km/h", "kilometers per hour")

    texto = re.sub(r"\b(\d+)\s*Pa(?![a-zA-Z])", r"\1 pascals", texto)
    texto = re.sub(r"\b(\d+)\s*kPa(?![a-zA-Z])", r"\1 kilopascals", texto)
    texto = re.sub(r"\b(\d+)\s*MPa(?![a-zA-Z])", r"\1 megapascals", texto)

    texto = re.sub(r"\b(\d+)\s*J(?![a-zA-Z])", r"\1 joules", texto)
    texto = re.sub(r"\b(\d+)\s*kJ(?![a-zA-Z])", r"\1 kilojoules", texto)
    texto = re.sub(r"\b(\d+)\s*MJ(?![a-zA-Z])", r"\1 megajoules", texto)
    texto = re.sub(r"\b(\d+)\s*eV(?![a-zA-Z])", r"\1 electronvolts", texto)
    texto = re.sub(r"\b(\d+)\s*keV(?![a-zA-Z])", r"\1 kilo-electronvolts", texto)
    texto = re.sub(r"\b(\d+)\s*MeV(?![a-zA-Z])", r"\1 mega-electronvolts", texto)

    texto = re.sub(r"\b(\d+)\s*Hz(?![a-zA-Z])", r"\1 hertz", texto)
    texto = re.sub(r"\b(\d+)\s*kHz(?![a-zA-Z])", r"\1 kilohertz", texto)
    texto = re.sub(r"\b(\d+)\s*MHz(?![a-zA-Z])", r"\1 megahertz", texto)
    texto = re.sub(r"\b(\d+)\s*GHz(?![a-zA-Z])", r"\1 gigahertz", texto)

    texto = re.sub(r"\b(\d+)\s*V(?![a-zA-Z])", r"\1 volts", texto)
    texto = re.sub(r"\b(\d+)\s*kV(?![a-zA-Z])", r"\1 kilovolts", texto)
    texto = re.sub(r"\b(\d+)\s*mV(?![a-zA-Z])", r"\1 millivolts", texto)

    texto = re.sub(r"\b(\d+)\s*A(?![a-zA-Z])", r"\1 amperes", texto)
    texto = re.sub(r"\b(\d+)\s*mA(?![a-zA-Z])", r"\1 milliamperes", texto)
    texto = re.sub(r"\b(\d+)\s*[µμ]A(?![a-zA-Z])", r"\1 microamperes", texto)  # acepta µ o μ

    texto = re.sub(r"\b(\d+)\s*W(?![a-zA-Z])", r"\1 watts", texto)
    texto = re.sub(r"\b(\d+)\s*kW(?![a-zA-Z])", r"\1 kilowatts", texto)
    texto = re.sub(r"\b(\d+)\s*MW(?![a-zA-Z])", r"\1 megawatts", texto)

    texto = re.sub(r"\b(\d+)\s*N(?![a-zA-Z])", r"\1 newtons", texto)

    texto = re.sub(r"\b(\d+)\s*mol(?![a-zA-Z])", r"\1 moles", texto)
    texto = re.sub(r"\b(\d+)\s*L(?![a-zA-Z])", r"\1 liters", texto)
    texto = re.sub(r"\b(\d+)\s*ml(?![a-zA-Z])", r"\1 milliliters", texto)

    texto = texto.replace("µL", "microliters")
    texto = texto.replace("μL", "microliters")
    texto = texto.replace("m³", "cubic meters")
    texto = texto.replace("cm³", "cubic centimeters")
    texto = texto.replace("mm³", "cubic millimeters")

    texto = texto.replace("Ω", "ohms")
    texto = texto.replace("µΩ", "micro-ohms")
    texto = texto.replace("μΩ", "micro-ohms")

    texto = re.sub(r"(?<![Nn])°", " degrees", texto)
    
    return texto

def clean_symbols(texto):
    t = texto
    t = t.replace("α", "alpha")
    t = t.replace("β", "beta")
    t = t.replace("γ", "gamma")
    t = t.replace("δ", "delta")
    t = t.replace("ε", "epsilon")
    t = t.replace("ζ", "zeta")
    t = t.replace("η", "eta")
    t = t.replace("θ", "theta")
    t = t.replace("ι", "iota")
    t = t.replace("κ", "kappa")
    t = t.replace("λ", "lambda")
    t = t.replace("μ", "mu")
    t = t.replace("ν", "nu")
    t = t.replace("ξ", "xi")
    t = t.replace("ο", "omicron")
    t = t.replace("π", "pi")
    t = t.replace("ρ", "rho")
    t = t.replace("σ", "sigma")
    t = t.replace("τ", "tau")
    t = t.replace("υ", "upsilon")
    t = t.replace("φ", "phi")
    t = t.replace("χ", "chi")
    t = t.replace("ψ", "psi")
    t = t.replace("ω", "omega")
    t = t.replace("Α", "Alpha")
    t = t.replace("Β", "Beta")
    t = t.replace("Γ", "Gamma")
    t = t.replace("Δ", "Delta")
    t = t.replace("Ε", "Epsilon")
    t = t.replace("Ζ", "Zeta")
    t = t.replace("Η", "Eta")
    t = t.replace("Θ", "Theta")
    t = t.replace("Ι", "Iota")
    t = t.replace("Κ", "Kappa")
    t = t.replace("Λ", "Lambda")
    t = t.replace("Μ", "Mu")
    t = t.replace("Ν", "Nu")
    t = t.replace("Ξ", "Xi")
    t = t.replace("Ο", "Omicron")
    t = t.replace("Π", "Pi")
    t = t.replace("Ρ", "Rho")
    t = t.replace("Σ", "Sigma")
    t = t.replace("Τ", "Tau")
    t = t.replace("Υ", "Upsilon")
    t = t.replace("Φ", "Phi")
    t = t.replace("Χ", "Chi")
    t = t.replace("Ψ", "Psi")
    t = t.replace("Ω", "Omega")
    t = t.replace("°", " degrees")
    t = t.replace("±", " plus-minus")
    t = t.replace("×", "x")     
    t = t.replace("÷", "/")         
    t = t.replace("·", "*")         
    t = t.replace("‰", " per mille")
    t = t.replace("∞", "infinity")
    t = t.replace("√", "sqrt")
    t = t.replace("≈", "approximately")
    t = t.replace("≠", "not equal")
    t = t.replace("≤", "less or equal")
    t = t.replace("≥", "greater or equal")
    t = t.replace("→", "to")
    t = t.replace("←", "from")
    t = t.replace("↔", "reversible")
    t = t.replace("↑", "up")
    t = t.replace("↓", "down")
    t = t.replace("Ω", "ohm")
    t = t.replace("µ", "micro")
    t = t.replace("μ", "micro")
    t = t.replace("Å", "angstrom")
    return t



'''