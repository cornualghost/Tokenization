import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# Scaricare le stopwords
nltk.download('stopwords')

def clean_text(text_column):
    # Definisci le stopwords in inglese come un set
    stop = set(stopwords.words('english'))

    # Compila l'espressione regolare per la pulizia del testo
    reg_exp = re.compile(r'[^a-zA-Z]+')

    # Funzione per pulire una singola stringa
    def clean_string(text):
        text = text.lower()  # Converti in minuscolo
        text = reg_exp.sub(' ', text)  # Sostituisci i caratteri non alfabetici con spazi
        return ' '.join(w for w in text.split() if len(w) > 1 and w not in stop)

    return clean_string

#importare i datset
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv("test.csv")


# Esempio di utilizzo
clean_func = clean_text('Description')
train_set['cleaned_text'] = train_set['Description'].apply(clean_func)
train_set.head()