import mauve # pip install mauve-text
from nltk import ngrams
import pandas as pd

RESULTS_PATH = "./resources/results/resultados_validacion_test_set_shakespeare.csv"

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

i = 0
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


'''
0.9 & 100 & 0.95 & 0.3 & 0.2: Distinct-1: 0.9412, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9923948319704172, 0.9795624549133835, 0.9636093869600961, 0.9734214694210883]: Texto muy diverso y muy similar a los reales (alta naturalidad).
0.7 & 80 & 0.9 & 0.2 & 0.3: Distinct-1: 0.8438, Distinct-2: 0.9841, Distinct-3: 1.0000, MAUVE: [0.9884595803147296, 0.976094015998049, 0.9793284171765253, 0.9732397472019927] : Buena similitud con los reales, algo menos diversidad léxica.
0.5 & 90 & 0.7 & 0.4 & 0.7: Distinct-1: 0.8732, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9755514614545514, 0.9725033047755178, 0.961332979095834, 0.9761344700117454]: Ligera caída en MAUVE; mantiene buena diversidad.



#1 0.5 & NaN & NaN & 0.0 & 0.0: Distinct-1: 0.7541, Distinct-2: 0.9667, Distinct-3: 1.0000, MAUVE: [0.9690300879577404, 0.98773088679212, 0.9823661149322855, 0.9553592593416217], MAUVE_MEAN: 0.974
#2 1.0 & NaN & NaN & 0.0 & 0.0: Distinct-1: 0.9464, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9593937294162936, 0.9850211567785717, 0.9770889198373532, 0.9280972393081925], MAUVE_MEAN: 0.962
#3 1.5 & NaN & NaN & 0.0 & 0.0: Distinct-1: 0.9516, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9572815001907358, 0.9590866939468519, 0.9609964822094239, 0.9136689017109472], MAUVE_MEAN: 0.948
#4 1.0 & 10.0 & NaN & 0.0 & 0.0:Distinct-1: 0.6923, Distinct-2: 0.9531, Distinct-3: 0.9841, MAUVE: [0.9574285522188832, 0.9472029182029389, 0.9675548694240276, 0.9771033105833429], MAUVE_MEAN: 0.962
#5 1.0 & 90.0 & NaN & 0.0 & 0.0: Distinct-1: 0.8889, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9137948736131387, 0.9556111638087439, 0.9631731553818834, 0.9596532779823079], MAUVE_MEAN: 0.948
#6 1.0 & 200.0 & NaN & 0.0 & 0.0: Distinct-1: 0.8548, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9896063835584732, 0.9769262083878218, 0.9766561050788269, 0.9775075354435265], MAUVE_MEAN: 0.980
#7 1.0 & NaN & 0.05 & 0.0 & 0.0: Distinct-1: 0.2090, Distinct-2: 0.3182, Distinct-3: 0.4154, MAUVE: [0.3646033300794174, 0.42227703025180785, 0.38790560817672765, 0.3935001252116025], MAUVE_MEAN: 0.392
#8 1.0 & NaN & 0.3 & 0.0 & 0.0: Distinct-1: 0.6949, Distinct-2: 0.9483, Distinct-3: 0.9825, MAUVE: [0.9728586743933233, 0.9446884323276068, 0.9497440677908258, 0.9658742338066523], MAUVE_MEAN: 0.958
#9 1.0 & NaN & 0.9 & 0.0 & 0.0: Distinct-1: 0.9153, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.982310389304347, 0.9757888456698202, 0.9849204823150755, 0.9558890904645365], MAUVE_MEAN: 0.975
#10 1.0 & NaN & NaN & 0.2 & 0.0: Distinct-1: 0.9538, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9855339459769228, 0.9654510190388226, 0.9667350816360201, 0.9655062368536236], MAUVE_MEAN: 0.971
#11 1.0 & NaN & NaN & 0.5 & 0.0: Distinct-1: 0.9219, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9475998121156503, 0.9833970416325568, 0.9830365883540993, 0.9867779247205868], MAUVE_MEAN: 0.975
#12 1.0 & NaN & NaN & 1.0 & 0.0: Distinct-1: 0.9324, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9747397259847849, 0.9851872945799622, 0.97785629244206, 0.9456956563451203], MAUVE_MEAN: 0.971
#13 1.0 & NaN & NaN & 0.0 & 0.2: Distinct-1: 0.9672, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9764940130845725, 0.9668136507735159, 0.9837877428071409, 0.9650595962731949], MAUVE_MEAN: 0.973
#14 1.0 & NaN & NaN & 0.0 & 0.5: Distinct-1: 0.9167, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9782181216535186, 0.9624999557794003, 0.9680526367126703, 0.9390240460325102], MAUVE_MEAN: 0.962
#15 1.0 & NaN & NaN & 0.0 & 1.0: Distinct-1: 0.9710, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9680379001917556, 0.9776166572681381, 0.976097034250536, 0.9856242424020132], MAUVE_MEAN: 0.977
#16 0.2 & 50.0 & 0.8 & 0.0 & 0.0: Distinct-1: 0.3729, Distinct-2: 0.6034, Distinct-3: 0.8070, MAUVE: [0.7743873240162096, 0.7928891084303007, 0.7658725999895821, 0.8156768138318382], MAUVE_MEAN: 0.787
#17 0.7 & NaN & 0.9 & 0.3 & 0.3: Distinct-1: 0.9016, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.97166367487617, 0.9854288627493943, 0.9706150064766885, 0.9703183018318542], MAUVE_MEAN: 0.975
#18 1.5 & 100.0 & 1.0 & 0.5 & 0.5: Distinct-1: 0.9359, Distinct-2: 1.0000, Distinct-3: 1.0000, MAUVE: [0.9834422296941461, 0.9249930918353998, 0.9280216264810469, 0.8890250099106116], MAUVE_MEAN: 0.931

La mayoría de los casos tienen MAUVE > 0.95, lo cual indica alta similitud con textos reales.
Los casos con Distinct-1 bajos (por ejemplo, #7 o #16) tienen también MAUVE mucho menor, señal de repeticiones o textos incoherentes.

Los mejores resultados son el 6 (Excelente balance entre variedad y similitud al estilo real), 15 (Muy alta coherencia y diversidad, probablemente texto más fluido.), 9 (Ligera menor diversidad léxica, pero casi perfecta naturalidad.), 11(Muy natural, aunque podría estar cerca del límite de creatividad), 17(Resultados muy consistentes y equilibrados.) 

Los mejores resultados combinan alta diversidad (Distinct-1 > 0.85) con MAUVE > 0.97, lo que indica que los textos generados son simultáneamente ricos en vocabulario y muy similares a textos reales.
'''