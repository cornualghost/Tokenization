import pandas as pd
from mosestokenizer import MosesTokenizer

class MosesTokenizationPipeline:
    def __init__(self, language='en'):
        self.language = language
        self.tokenizer = MosesTokenizer(self.language)

    def tokenize_text(self, text):
        # Tokenizza il testo e unisce i token in una stringa
        return ' '.join(self.tokenizer(text))

    def tokenize_column(self, df, column_name):
        # Applica la tokenizzazione a una colonna del DataFrame
        return df[column_name].apply(lambda x: self.tokenize_text(x) if isinstance(x, str) else x)

    def close(self):
        # Rilascia le risorse del tokenizzatore
        self.tokenizer.close()

# Utilizzo della pipeline

# Caricare il dataset
#train_set = pd.read_csv('train.csv')
#test_set= pd.read_csv('test.csv')
#pipeline = MosesTokenizationPipeline('en')

# Applicare la tokenizzazione
#train_set['Moses'] = pipeline.tokenize_column(train_set, 'Description')
#test_set['Moses'] = pipeline.tokenize_column(test_set, 'Description')


#pipeline.close()
