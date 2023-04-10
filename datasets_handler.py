import datetime
import pandas as pd
import evaluation_metrics
import datasets
from datasets import load_dataset
from datasets import Dataset
import re
import numpy as np

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

dict_classes_imdb = {
    0 : "positive",
    1 : "negative"
    }

def getAgora():
    return str(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))


def getDataset(which_dataset,path=None,labelEncoder=None):
    implemented_datasets = ["ag_news","bbcnews","folhauol"]
    if which_dataset == "ag_news":
        dataset = load_dataset("ag_news")
        dataset_df = pd.concat([pd.DataFrame(dataset['train']),pd.DataFrame(dataset['test'])]).reset_index()
        class_col = 'class'
        data_col = 'text'
        dataset_df[class_col] = dataset_df['label'].map(dict_classes_ag_news)
        dict_cols = {data_col: 'text', class_col: 'class'}
        dataset_df =dataset_df.rename(columns=dict_cols) 
        return dataset_df, 'text', 'class'
    
    if which_dataset == "imdb":
        dataset = load_dataset("imdb")
        dataset_df = pd.concat([pd.DataFrame(dataset['train']),pd.DataFrame(dataset['test'])]).reset_index()
        class_col = 'class'
        data_col = 'text'
        dataset_df[class_col] = dataset_df['label'].map(dict_classes_imdb)
        dict_cols = {data_col: 'text', class_col: 'class'}
        dataset_df =dataset_df.rename(columns=dict_cols) 
        return dataset_df, 'text', 'class'
    
    if which_dataset == "bbcnews":
        if path == None:
            path = '/Users/alealcoforado/Documents/Projetos/Datasets/bbc-news/BBC News Train.csv'
        dataset_df = pd.read_csv(path, sep = ',')
        data_col = 'Text'
        class_col = 'Category'
        dict_cols = {data_col: 'text', class_col: 'class'}
        dataset_df = dataset_df.rename(columns=dict_cols) 
        return dataset_df,  'text', 'class'
    
    if which_dataset=='folhauol':
        if path == None:
            path = '/Users/alealcoforado/Documents/Projetos/Datasets/folhauol/folhauol_clean_df_articles.csv'
        dataset_df = pd.read_csv(path)
        dataset_df['full_text'] = dataset_df['title'].astype(str)+"."+dataset_df['text'].astype(str)
        dataset_df['len'] = dataset_df['full_text'].apply(len)
        dataset_df = dataset_df[dataset_df['len']>=300]
        data_col = 'full_text'
        class_col = 'category'
        dataset_df[class_col] = dataset_df[class_col].map(dict_classes_folha)

        dict_cols = {data_col: 'text', class_col: 'class'}
        dataset_df = dataset_df.drop(columns='text',errors='ignore').rename(columns=dict_cols) 
        # dataset_df = evaluation_metrics.Encoder(dataset_df,labelEncoder=labelEncoder,columnsToEncode=['class'])
        return dataset_df, 'text', 'class'

    if which_dataset=='ml':
        if path == None:
            path = '/Users/alealcoforado/Documents/Projetos/Datasets/ml/joao_rubinato - base_raw.csv'
        dataset_df = pd.read_csv(path)
        data_col = 'pista_raw'
        class_col = 'macro_raw'
        dataset_df =dataset_df[(dataset_df['tem_explicação?']==True) & (dataset_df['tem_texto?']==True) & (dataset_df['tem_macro?']==True)]
        dataset_df = dataset_df.astype(str)

        return dataset_df

    print ("No dataset chosen. Options are {}.".format(implemented_datasets))
    return None

def getZeroshotPreviousData(which_dataset,class_col,top_n = 8,exec_time=None,zeroshot_data_local_path=None):
    if zeroshot_data_local_path==None:
        zeroshot_data_local_path = '/Users/alealcoforado/Documents/Projetos/Datasets/{which_dataset}/'.format(which_dataset=which_dataset)

    zeroshot_preds_and_probs_file = 'predictions_and_probabilities_test_{exec_time}.csv'.format(exec_time=exec_time)
    preds_probs_df = pd.read_csv(zeroshot_data_local_path+zeroshot_preds_and_probs_file)

    # zeroshot_config_file = 'zeroshot_config_test_{exec_time}.csv'.format(exec_time=exec_time)
    # config_df = pd.read_csv(zeroshot_data_local_path+zeroshot_config_file)
    preds_probs_df.index = preds_probs_df['Unnamed: 0.1'] ### recover original indexes for dataset
    df_top_n = preds_probs_df.sort_values(['top_probability','prediction_code'], ascending=False).groupby('prediction_code').head(top_n)
    df_top_n = df_top_n.drop(columns=["Unnamed: 0.1",class_col,class_col+"_code"],errors='ignore')
    return df_top_n

def mergeLabelingToDataset(raw_data,previous_data,class_col):
    raw_data_final = raw_data.join(previous_data)
#     ## overwrite true labels with predictions from zeroshot
    new_class_col = 'new_'+class_col
    raw_data_final.loc[~raw_data_final['prediction'].isna(),new_class_col] = raw_data_final['prediction'] 
    raw_data_final.loc[raw_data_final['prediction'].isna(),new_class_col] = raw_data_final[class_col]

#     ## keep true labels of the rest, for testing
    raw_data_final = evaluation_metrics.Encoder(raw_data_final,columnsToEncode=[new_class_col])
    return raw_data_final, new_class_col

def splitDataset(raw_data, config,zeroshot_data_local_path=None):
    # Get configuration values
    data_col = config['data_col']
    # new_class_col = config['new_class_col']
    test_dataset_sample_size = config['max_inferences']
    random_state = config['random_state']
    how = config['split']
   
    if how == "zeroshot":
        zeroshot_previous_data = getZeroshotPreviousData( ### load results from labeling step and create training data for contrastive learning
            config['dataset'],config['class_col'],top_n=config['top_n'],
            exec_time=config['exec_time'],zeroshot_data_local_path=zeroshot_data_local_path)
        
        # Split data into train set only
        # train_data = raw_data[~raw_data['prediction_code'].isna()].groupby("prediction_code", group_keys=True)\
        #     .apply(lambda s: s.sample(min(len(s), config['training_examples']), random_state=random_state))
        # print(train_data.sum())
        train_data = zeroshot_previous_data.groupby("prediction_code", group_keys=True)\
            .apply(lambda s: s.sample(min(len(s), config['training_examples']), random_state=random_state))
        print(train_data.sum())

        # Rename columns and convert data types
        
        train_data = train_data.drop(columns='class_code',errors='ignore').rename(columns={'prediction_code': 'class_code'})
        train_data[data_col] = train_data[data_col].apply(str)
        # train_data['class_code'] = train_data['class_code'].apply(int)
        test_data = raw_data
        test_data[data_col] = test_data[data_col].apply(str)
        test_data['class_code'] = test_data['class_code'].apply(int)


    if how == 'fewshot': 
        # Split data into train and test sets by selecting a random subset of each class
        # for the training set and the remaining data for the test set
        train_data = raw_data.groupby('class_code').head(config['training_examples'])[[data_col,'class_code']]
        keys = list(train_data.columns.values)
        i1 = raw_data.set_index(keys).index
        i2 = train_data.set_index(keys).index
        test_data = raw_data[~i1.isin(i2)]
        train_data[data_col] = train_data[data_col].apply(str)
        train_data['class_code'] = train_data['class_code'].apply(int)
        test_data[data_col] = test_data[data_col].apply(str)
        test_data['class_code'] = test_data['class_code'].apply(int)

       
    return train_data[[data_col,'class_code']],test_data[[data_col,'class_code']]


def buildDatasetDict(df_train,df_test):
    test_dataset = Dataset.from_dict(df_test[['text','class_code']].to_dict('list'))
    train_dataset = Dataset.from_dict(df_train[['text','class_code']].to_dict('list'))
    # dataset_dict = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})
    return train_dataset,test_dataset

def splitDocuments(docs: pd.Series()) -> list():
    all_sentences = np.array(docs.apply(lambda x : re.sub(r'(\d+)(\.)(\d+)',r"\1"+r"\3", x)).str.split("."))
    flat_sentences = [s for sentences in all_sentences for s in sentences]
    train_sentences = dropEmptyStrings(flat_sentences)
    return train_sentences

def dropEmptyStrings(strings_list):
    for s in strings_list:
        if s=="" or bool(re.search('^\ +$',s)):
            strings_list.remove(s)
    return strings_list

# def downloadPortugueseWikipedia():
#     # download Portuguese Wikipedia
#     get_wiki(path_data,lang)
#     # create one text file by article
#     dest = split_wiki(path_data,lang)
#     # get all articles in one text file and one csv file
#     get_one_clean_file(dest,lang)
#     get_one_clean_csv_file(dest,lang)