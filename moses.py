from mosestokenizer import MosesTokenizer
import pandas as pd

# Caricare il dataset
test_set = pd.read_csv('C:/Users/keita/OneDrive/Documenti/universita/data_processing/train.csv')

# Creazione di un'istanza del tokenizer
with MosesTokenizer('en') as tokenizer:
    test_set['Tokenized_Title'] = test_set['Title'].apply(tokenizer)