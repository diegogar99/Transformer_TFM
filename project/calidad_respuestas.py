import mauve # pip install mauve-text
from nltk import ngrams
import pandas as pd

RESULTS_PATH = "./resources/results/resultados_validacion_test_set_shakespeare2.csv"

# Comparar con parrafos de Shakespeare reales
def mauve_score(p_text, q_text, context_len):
    mauve_result = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=context_len, verbose=False)  #featurize_model_name='gpt2-medium'

    #print("MAUVE score:", mauve_result.mauve)
    return mauve_result.mauve


def distinct_n(text, n):
    tokens = text.split()
    n_grams = list(ngrams(tokens, n))
    if len(n_grams) == 0:
        return 0.0
    unique_ngrams = set(n_grams)
    return len(unique_ngrams) / len(n_grams)


df = pd.read_csv(RESULTS_PATH)

respuestas_modelo = df["Resultado"].tolist()
frases_reales = ["""
KING RICHARD III:
Awaked you not with this sore agony?
And for Rome's good. I'll tell thee what; yet go:
Where are your mess of sons to back you now?
We'll show thee Io as she was a maid,
Well said, Hermione.
To thrust the lie unto him.
He that is giddy thinks the world turns round.
""",
"""
POMPEY:
We'll part the time between's then; and in that
Young Edward lives: think now what I would say.
First Murderer:
The mad-brain'd bridegroom took him such a cuff
SICINIUS:
To swim, to dive into the fire, to ride
Of shame seen through thy country, speed
For they have pardons, being ask'd, as free
CAPULET:
To threaten me with death is most unlawful.
AUFIDIUS:
Three words, dear Romeo, and good night indeed.
GLOUCESTER:
jerkin, a pair of old breeches thrice turned, a pair
ISABELLA:
His hand to wield a sceptre, and himself
""",

"""
DUKE VINCENTIO:
Yet are they passing cowardly. But, I beseech you,
That is no slander, sir, which is a truth;
O, this is it that makes your servants droop!
You had only in your silent judgment tried it,
Of fire and water, when their thundering shock
No, Warwick, thou art worthy of the sway,
Second Lady:
For present comfort and for future good,
Provost:
Long as my exile, sweet as my revenge!
BAPTISTA:
'Tis time
I am no breeching scholar in the schools;
MARCIUS:
JULIET:
""",

"""
KING RICHARD II:
And Henry is my king, Warwick his subject.
To let him slip at will.
Meaning, to court'sy.
And, when it bows, stands up. Thou art left, Marcius:
What must be shall be.
Supplied with worthy men! plant love among 's!
Nay, do not pause; for I did kill King Henry,
ALONSO:
It shall be done, my sovereign, with all speed.
Are pluck'd up root and all by Bolingbroke,
Most ponderous and substantial things!
FRIAR LAURENCE:
VIRGILIA:
"""

]


for resp in respuestas_modelo:
    print("Evaluando respuesta...")
    scores = []
    for real_text in frases_reales:
        score = mauve_score(resp, real_text, context_len=128)
        scores.append(score)
    d1 = distinct_n(resp, 1)
    d2 = distinct_n(resp, 2)
    d3 = distinct_n(resp, 3)
    print(f"Distinct-1: {d1:.4f}, Distinct-2: {d2:.4f}, Distinct-3: {d3:.4f}, MAUVE: {scores}")
    print("-" * 50)