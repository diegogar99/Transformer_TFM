import re, ftfy, unicodedata

# Compila todos los patrones regex UNA SOLA VEZ (ahorra tiempo)
RE_MULTI_SPACE = re.compile(r"\s\s*")
RE_OP = re.compile(r"(?s)\(\s*Op\s*\.\s*\d+.*?\)|Op\s*\.\s*\d+.*?\n")
RE_TITLES = re.compile(r"=\s*(=\s*)*.*=\s*(=\s*)*\n")
RE_AT = re.compile(r"@(.*?)@")
RE_NONLATIN = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s.,;:!?'\"(){}\[\]\-+=\/%°<>_]")
RE_NUCLEAR = re.compile(
    r"^(?=.*\b\d+\s*\*?\s*[A-Z][a-z]?\b)(?=.*(→|->|=>|\+))(?=.*\b(MeV|keV)\b).*?$",
    flags=re.MULTILINE
)
RE_REF = re.compile(r"\(\s*[A-Za-z]+\s*,\s*\d{4}\s*\)")
RE_WORDS_LEN = re.compile(r"^(?:\s*\S+\s+){0,20}\S*\s*$")
RE_CONTRACCIONES = re.compile(r"\s+('[a-z][^a-zA-Z])")
RE_CONTRACCIONES2 = re.compile(r"(\w+)'(\w{2,})")

# Diccionario de símbolos griegos y científicos → texto
REPLACEMENTS = {
    # griegas minúsculas
    "α":"alpha","β":"beta","γ":"gamma","δ":"delta","ε":"epsilon","ζ":"zeta","η":"eta",
    "θ":"theta","ι":"iota","κ":"kappa","λ":"lambda","μ":"mu","ν":"nu","ξ":"xi","ο":"omicron",
    "π":"pi","ρ":"rho","σ":"sigma","τ":"tau","υ":"upsilon","φ":"phi","χ":"chi","ψ":"psi","ω":"omega",
    # griegas mayúsculas
    "Α":"Alpha","Β":"Beta","Γ":"Gamma","Δ":"Delta","Ε":"Epsilon","Ζ":"Zeta","Η":"Eta","Θ":"Theta",
    "Ι":"Iota","Κ":"Kappa","Λ":"Lambda","Μ":"Mu","Ν":"Nu","Ξ":"Xi","Ο":"Omicron","Π":"Pi",
    "Ρ":"Rho","Σ":"Sigma","Τ":"Tau","Υ":"Upsilon","Φ":"Phi","Χ":"Chi","Ψ":"Psi","Ω":"Omega",
    # científicos
    "±":" plus-minus","×":"x","÷":"/","·":"*","‰":" per mille","∞":"infinity","√":"sqrt",
    "≈":"approximately","≠":"not equal","≤":"less or equal","≥":"greater or equal",
    "→":"to","←":"from","↔":"reversible","↑":"up","↓":"down","Ω":"ohm","µ":"micro","μ":"micro","Å":"angstrom","≡":"equivalent"
}

def bulk_replace(text, replacements):
    """Reemplazo rápido usando regex de alternancia."""
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    return pattern.sub(lambda m: replacements[m.group(0)], text)

# Compilamos una vez las unidades con regex grandes
UNITS_REGEX = re.compile(
    r"\b(\d+)\s*("
    r"[µμ]m|mm|cm|m|km|mg|[µμ]g|g|kg|s|h|min|ms|°C|°F|Ks|m/s|km/h|Pa|kPa|MPa|"
    r"J|kJ|MJ|eV|keV|MeV|Hz|kHz|MHz|GHz|V|kV|mV|A|mA|[µμ]A|W|kW|MW|N|mol|L|ml|"
    r"µL|μL|m³|cm³|mm³|Ω|µΩ|μΩ"
    r")\b(?![a-zA-Z])"
)

UNIT_MAP = {
    "µm":"micrometers","μm":"micrometers","mm":"millimeters","cm":"centimeters","m":"meters","km":"kilometers",
    "mg":"milligrams","µg":"micrograms","μg":"micrograms","g":"grams","kg":"kilograms",
    "s":"seconds","h":"hours","min":"minutes","ms":"milliseconds",
    "°C":" degrees Celsius","Ks":"kelvin","°F":" degrees Fahrenheit",
    "m/s":"meters per second","km/h":"kilometers per hour",
    "Pa":"pascals","kPa":"kilopascals","MPa":"megapascals",
    "J":"joules","kJ":"kilojoules","MJ":"megajoules","eV":"electronvolts","keV":"kilo-electronvolts","MeV":"mega-electronvolts",
    "Hz":"hertz","kHz":"kilohertz","MHz":"megahertz","GHz":"gigahertz",
    "V":"volts","kV":"kilovolts","mV":"millivolts",
    "A":"amperes","mA":"milliamperes","µA":"microamperes","μA":"microamperes",
    "W":"watts","kW":"kilowatts","MW":"megawatts",
    "N":"newtons","mol":"moles","L":"liters","ml":"milliliters","µL":"microliters","μL":"microliters",
    "m³":"cubic meters","cm³":"cubic centimeters","mm³":"cubic millimeters",
    "Ω":"ohms","µΩ":"micro-ohms","μΩ":"micro-ohms", "mi": "milles", "′":"minutos", "″": "segundos"
}

def replace_units(match):
    num, unit = match.groups()
    return f"{num} {UNIT_MAP.get(unit, unit)}"

def light_clean_fn(texto):
    t = texto["text"]
    t = ftfy.fix_text(t)
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-"), replace(r"^\s\.\.\.","")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    t = t.strip(" \n")
    return {"text": t}
    
def clean_wikitext2(texto):
    t = texto["text"]

    # Normalización Unicode y corrección
    t = ftfy.fix_text(t)
    t = unicodedata.normalize("NFKC", t)

    # Limpieza de patrones de "Op. xx" y encabezados
    t = RE_OP.sub("", t)
    t = RE_TITLES.sub("", t)

    # Quita texto entre @...@
    t = RE_AT.sub(r"\1", t)

    # Quita ecuaciones nucleares
    t = RE_NUCLEAR.sub("", t)

    # Limpia unidades (una sola pasada regex)
    t = UNITS_REGEX.sub(replace_units, t)

    # Reemplazos de símbolos griegos y científicos
    t = bulk_replace(t, REPLACEMENTS)

    # Sustituciones menores
    t = t.replace("☉", "")

    # Espaciado y limpieza general
    t = RE_MULTI_SPACE.sub(" ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    #t = t.strip(" \n")
    t = re.sub(r'\s+([\-\.\,\";\':])\s+',r"\1",t)
    t = t.replace(r"\(\s+", "(").replace(r"\s+\)", ")")
    # Elimina caracteres no latinos ni puntuación estándar
    t = RE_NONLATIN.sub("", t)
    t = RE_WORDS_LEN.sub("",t)
    t = RE_CONTRACCIONES.sub(r"\1",t)
    t = RE_CONTRACCIONES2.sub(r"\1' \2",t)
    t = re.sub(r'\bVol\.(?=\s*\d)', 'volume', t)  # Vol. seguido de número
    t = re.sub(r'\bNo\.(?=\s*\d)', 'number', t) 

    return {"text": t}


'''PATH = "/home/dgarciamur/root/TFM/Transformer_TFM/project/resources/datasets/wikitext2_train.txt"

with open(PATH, "r") as f:
    texto = f.read()

texto = clean_wikitext2(texto)

print(texto)'''