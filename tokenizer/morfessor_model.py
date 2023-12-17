import math
import morfessor
import pandas as pd

def log_func(x):
    return int(round(math.log(x + 1, 2)))

# Caricare il dataset
train_set = pd.read_csv('../dataset/train.csv')

# Estrazione del testo
titoli = train_set['Description']

# Salvare i titoli in un nuovo file di testo
path_del_file_titoli = '../model/titoli_per_morfessor.txt'
with open(path_del_file_titoli, 'w', encoding='utf-8') as file:
    for titolo in titoli:
        file.write(titolo + '\n')


# Creazione di un'istanza per I/O e del modello Morfessor
infile = path_del_file_titoli
io = morfessor.MorfessorIO()
train_data = list(io.read_corpus_file(infile))

# Creare un'istanza del modello Morfessor e caricare i dati
model = morfessor.BaselineModel()
model.load_data(train_data, count_modifier=log_func)

# Addestrare il modello
model.train_batch()

# Salvare il modello addestrato
io.write_binary_model_file("../model/model.bin", model)