import datetime
import pandas as pd
dict_classes_folha = {
    'poder':"Poder e Política no Brasil",
    'mercado':"Mercado",
    'mundo':"Notícias de fora do Brasil",
    'esporte':"Esporte",
    'tec':"Tecnologia",
    'ambiente':"Meio Ambiente",
    'equilibrioesaude':"Equilíbrio e Saúde",
    'educacao':"Educação",
    'tv':"TV, Televisão e Entretenimento",
    'ciencia':"Ciência",
    'turismo':"Turismo",
    'comida':"Comida" }

dict_classes_ag_news = {
    0:"world",
    1:"sports",
    2:"business",
    3:"science and technology"
   }

def getAgora():
    return str(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))

from datasets import load_dataset

def getDataset(which_dataset):
    implemented_datasets = ["ag_news"]
    if which_dataset == "ag_news":
        dataset = load_dataset("ag_news")
        dataset_df = pd.concat([pd.DataFrame(dataset['train']),pd.DataFrame(dataset['test'])]).reset_index()
        dataset_df['label_text'] = dataset_df['label'].map(dict_classes_ag_news)
        # print(dataset)
        return dataset_df, 'text', 'label_text'
    print ("No dataset chosen. Options are {}.".format(implemented_datasets))
    return None


