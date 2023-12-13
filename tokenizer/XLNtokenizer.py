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
    
    
# Creazione della pipeline
#pipeline = Pipeline([
#    ('xlnet_tokenization', XLNetTokenization())])

# Esempio di utilizzo
#train_set = pd.read_csv('train_set_bpe.csv')
#test_set = pd.read_csv("test_set_bpe.csv")

#train_set['XLN'] = pipeline.fit_transform(train_set['cleaned_text'])
#test_set['XLN'] = pipeline.fit_transform(test_set['cleaned_text'])

#rimozione liste
#train_set['XLN'] = train_set['XLN'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)
#test_set['XLN'] = test_set['XLN'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)

