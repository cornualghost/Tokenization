import math
import morfessor
import pandas as pd

# Caricare il dataset
train_set = pd.read_csv('train.csv')

# Estrazione solo dei titoli (o modifica questa linea per estrarre la parte desiderata)
titoli = train_set['Title']

# Salvare i titoli in un nuovo file di testo
path_del_file_titoli = 'titoli_per_morfessor.txt'
with open(path_del_file_titoli, 'w', encoding='utf-8') as file:
    for titolo in titoli:
        file.write(titolo + '\n')

# Ora puoi usare 'titoli_per_morfessor.txt' come infile per Morfessor
infile = path_del_file_titoli

# Funzione per aggiustare i conteggi di ogni composto
def log_func(x):
    return int(round(math.log(x + 1, 2)))

# Creazione di un'istanza per I/O e del modello Morfessor
io = morfessor.MorfessorIO()
train_data = list(io.read_corpus_file(infile))

# Creare un'istanza del modello Morfessor e caricare i dati
model = morfessor.BaselineModel()
model.load_data(train_data, count_modifier=log_func)

# Addestrare il modello
model.train_batch()

# Salvare il modello addestrato
io.write_binary_model_file("model.bin", model)

# Caricare il dataset
df_test = pd.read_csv("C:/Users/keita/OneDrive/Documenti/universita/data_processing/test.csv")

# Estrarre i titoli
titles = df_test['Title']

# Caricare il modello Morfessor
io = morfessor.MorfessorIO()
model = io.read_binary_model_file("model.bin")

# Applicare il modello Morfessor a ogni titolo
tokenized_titles = []
for title in titles:
    segmentation = model.viterbi_segment(title)[0]
    tokenized_titles.append(segmentation)

# Aggiungere i titoli tokenizzati al dataframe
df_test['Tokenized_Title'] = tokenized_titles

# Visualizzare i risultati
print(df_test[['Title', 'Tokenized_Title']])