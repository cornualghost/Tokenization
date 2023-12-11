import pandas as pd
from transformers import AutoTokenizer
from collections import Counter, defaultdict

class BPEPipeline:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.bpe = None

    def fit(self, corpus):
        self.bpe = BPE(corpus, self.vocab_size)
        self.bpe.train()
    
    def transform(self, text):
        if self.bpe is None:
            raise RuntimeError("Il tokenizzatore BPE non è stato ancora addestrato. Chiama prima il metodo fit.")
        return self.bpe.tokenize(text)

# Classe BPE 
class BPE:
    
    
    def __init__(self, corpus, vocab_size):
        """Initialize BPE tokenizer."""
        self.corpus = corpus
        self.vocab_size = vocab_size
        
        # pre-tokenize the corpus into words, BERT pre-tokenizer is used here
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}
    
    
    def train(self):
        """Train BPE tokenizer."""

        # compute the frequencies of each word in the corpus
        for text in self.corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1

        # compute the base vocabulary of all characters in the corpus
        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()

        # add the special token </w> at the beginning of the vocabulary
        vocab = ["</w>"] + alphabet.copy()

        # split each word into individual characters before training
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        # merge the most frequent pair iteratively until the vocabulary size is reached
        while len(vocab) < self.vocab_size:

            # compute the frequency of each pair
            pair_freqs = self.compute_pair_freqs()

            # find the most frequent pair
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            # merge the most frequent pair
            self.splits = self.merge_pair(*best_pair)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
        return self.merges


    def compute_pair_freqs(self):
        """Compute the frequency of each pair."""

        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs


    def merge_pair(self, a, b):
        """Merge the given pair."""

        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits
    

    def tokenize(self, text):
        """Tokenize a given text with trained BPE tokenizer (including pre-tokenization, split, and merge)."""
        
        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits_text = [[l for l in word] for word in pre_tokenized_text]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits_text[idx] = split
        result = sum(splits_text, [])
        return result
#importare i datset
"""train_set = pd.read_csv('train.csv')
test_set = pd.read_csv("test.csv")

# Utilizzo della pipeline BPE
# Sostituire con il  corpus sul quale si desidera addestrare bpe
corpus = train_set['Title']

vocab_size = 1000

bpe_pipeline = BPEPipeline(vocab_size)
bpe_pipeline.fit(corpus)


# Esempio di tokenizzazione
text = "Esempio di testo da tokenizzare"
tokenized_text = bpe_pipeline.transform(text)
print(tokenized_text)


# Esempio di tokenizzazione BPE al dataset
def apply_bpe_to_series(series):
    return series.apply(bpe_pipeline.transform)

# Applicare la funzione alla colonna desiderata del DataFrame
train_set['Tokenized'] = apply_bpe_to_series(train_set['Description'])
train_set['Tokenized'].head()
"""
'BPE PREADDESTRATO'

from transformers import RobertaTokenizer


class BpeRobertaPipeline:
    def __init__(self, model_name='roberta-base'):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def process_text(self, text):
        # Tokenizza il testo
        tokens = self.tokenizer.tokenize(text)
        # Converte i token in una stringa
        return ' '.join(tokens)

    def apply_pipeline(self, df, column_name):
        # Applica la pipeline di processamento del testo alla colonna specificata
        return df[column_name].apply(self.process_text)
   
# Esempio di Utilizzo BpeRobertaPipeline
"""
pipeline = BpeRobertaPipeline()
train_set['Processed_Text'] = pipeline.apply_pipeline(train_set, 'Description')

print(train_set['Processed_Text'])"""