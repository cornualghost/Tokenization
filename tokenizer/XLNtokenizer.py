import pandas as pd
from transformers import XLNetTokenizer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

# Caricare il tokenizzatore pre-addestrato di XLNet
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# Definizione di un Transformer personalizzato per utilizzare il tokenizzatore XLNet
class XLNetTokenization(TransformerMixin):
    def transform(self, X, **transform_params):
        # Applica la tokenizzazione XLNet a ogni elemento
        return [tokenizer.encode(text, add_special_tokens=True) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self
    

