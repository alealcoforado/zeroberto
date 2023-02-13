import datetime
import pandas as pd
import evaluation_metrics
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

def getDataset(which_dataset,path=None):
    implemented_datasets = ["ag_news"]
    if which_dataset == "ag_news":
        dataset = load_dataset("ag_news")
        dataset_df = pd.concat([pd.DataFrame(dataset['train']),pd.DataFrame(dataset['test'])]).reset_index()
        dataset_df['label_text'] = dataset_df['label'].map(dict_classes_ag_news)
        # print(dataset)
        return dataset_df, 'text', 'label_text'
    if which_dataset == "bbcnews":
        if path == None:
            path = '/Users/alealcoforado/Documents/Projetos/Datasets/bbc-news/BBC News Train.csv'
        dataset_df = pd.read_csv(path, sep = ',')
        return dataset_df, 'Text', 'Category'
    if which_dataset=='folhauol':
        if path == None:
            path = '/Users/alealcoforado/Documents/Projetos/Datasets/folhauol/folhauol_clean_df_articles.csv'
        # arq = '/content/drive/MyDrive/folhauol_clean_df_articles.csv'
        dataset_df = pd.read_csv(path)
        dataset_df['full_text'] = dataset_df['title'].astype(str)+dataset_df['text'].astype(str)
        data_col = 'full_text'
        class_col = 'category'
        dataset_df[class_col] = dataset_df[class_col].map(dict_classes_folha)

        return dataset_df, data_col, class_col


    print ("No dataset chosen. Options are {}.".format(implemented_datasets))
    return None


def getZeroshotPreviousData(which_dataset,class_col,top_n = 8,exec_time=None,zeroshot_data_local_path=None):
    if zeroshot_data_local_path==None:
        zeroshot_data_local_path = '/Users/alealcoforado/Documents/Projetos/Datasets/{which_dataset}/'.format(which_dataset=which_dataset)
    zeroshot_preds_and_probs_file = 'predictions_and_probabilities_test_{exec_time}.csv'.format(exec_time=exec_time)
    preds_probs_df = pd.read_csv(zeroshot_data_local_path+zeroshot_preds_and_probs_file)

    # zeroshot_config_file = 'zeroshot_config_test_{exec_time}.csv'.format(exec_time=exec_time)
    # config_df = pd.read_csv(zeroshot_data_local_path+zeroshot_config_file)
    preds_probs_df.index = preds_probs_df['Unnamed: 0'] ### recover original indexes for dataset
    df_top_n = preds_probs_df.sort_values(['top_probability','prediction'], ascending=False).groupby('prediction').head(top_n)
    df_top_n = df_top_n.drop(columns=["Unnamed: 0",class_col,class_col+"_code"])
    return df_top_n

def mergeLabelingToDataset(raw_data,previous_data,class_col):
    raw_data_final = raw_data.join(previous_data)
#     ## overwrite true labels with predictions from zeroshot
    new_class_col = 'new_'+class_col
    raw_data_final.loc[~raw_data_final['prediction'].isna(),new_class_col] = raw_data_final['prediction'] 
    raw_data_final.loc[raw_data_final['prediction'].isna(),new_class_col] = raw_data_final[class_col]

#     ## keep true labels of the rest, for testing
    raw_data_final = evaluation_metrics.Encoder(raw_data_final,[new_class_col])
    return raw_data_final

def splitDataset(dataframe,config):
    if (split == "zeroshot"):
        df_train = dataframe[~dataframe['prediction'].isna()].groupby(new_class_col+"_code")[[data_col,new_class_col+"_code"]].apply(lambda s: s.sample(min(len(s),top_n),random_state=random_state))

        keys = list(df_train.columns.values)

        i1 = dataframe.set_index(keys).index
        i2 = df_train.set_index(keys).index

        df_test = dataframe[~i1.isin(i2)]

        df_test = df_test.groupby(new_class_col+"_code")[[data_col,new_class_col+"_code"]].apply(lambda x:x.sample(int(len(x)*test_dataset_sample_size),random_state=random_state))

        df_train = df_train.astype(str)
        df_test = df_test.astype(str)

        ### transforma dataframes em datasetdict

        train_dataset = Dataset.from_dict(df_train)
        test_dataset = Dataset.from_dict(df_test)
        dataset_dict = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})
        dataset = dataset_dict
    return dataset