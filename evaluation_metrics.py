import evaluate
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import datasets_handler
import os
import pathlib
from datasets import Dataset

def Encoder(df,columnsToEncode=[]):
          # columnsToEncode = list(df.select_dtypes(include=['category']))
          # columnsToEncode = ['Classe_1']
          if columnsToEncode == []:
              columnsToEncode = df.columns
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature+"_code"] = le.fit_transform(df[feature])
                  df[feature+"_code"] = df[feature+"_code"].apply(int)
              except:
                  print('Error encoding '+feature)
          return df

def get_metric(y_pred,y_ref,metric="accuracy",average="binary"):
  # y_test = eval_dataset["label"]

    if (metric=='accuracy'):
      metric_fn = evaluate.load(metric)
      metric_value = metric_fn.compute(predictions=y_pred, references=y_ref)
      return metric_value
    else:
      metric_fn = evaluate.load(metric)
      metric_value = metric_fn.compute(predictions=y_pred, references=y_ref,average=average)

    return metric_value

def get_metrics(y_pred,y_ref,metrics=["accuracy","precision","recall","f1"]):

   all_metrics = {}
  #  df_encoded = Encoder(pd.DataFrame({"pred":y_pred,"ref":y_ref}))
  #  y_pred = df_encoded['pred_code']
  #  y_ref = df_encoded['ref_code']

   averages = ['weighted','macro']
   for average in averages:
      these_metrics = []
      for metric in metrics:
         these_metrics.append(get_metric(y_pred,y_ref,metric,average))
      all_metrics[average] = these_metrics
  
   return all_metrics

# print(get_metrics([ 1,2],[1,2]))
# print(get_metrics([ "boa", "alo"],[1,"alo"]))

# y_pred = [ "politics", "business","tech","tech"]

# y_ref = ["politics","business","politics","politics"]

# encoded = (Encoder(pd.DataFrame({"pred":y_pred,"ref":y_ref})))

# print(get_metrics(encoded['pred_code'],encoded['ref_code']))


def saveResults(setfit_config,metrics,local_path):
    agora = datasets_handler.getAgora()
    if local_path == None:
        dataset_path = '/Users/alealcoforado/Documents/Projetos/Datasets/{which_dataset}/'.format(which_dataset=setfit_config['dataset'])
    else:
        dataset_path = local_path
    folder = pathlib.Path(dataset_path)
    folder.mkdir(parents=True,exist_ok=True)
    metrics_filename = "metrics_setfit_{agora}.csv".format(agora=agora)
    setfit_config_filename = "config_setfit_{agora}.csv".format(agora=agora)
    print(metrics_filename)
    print(setfit_config_filename)
    pd.DataFrame(metrics).to_csv(dataset_path+metrics_filename)    
    pd.DataFrame([setfit_config]).to_csv(dataset_path+setfit_config_filename)
    return agora  



def saveZeroshotResults(zeroberto_config,results,local_path):
    agora = datasets_handler.getAgora()
    if local_path == None:
        dataset_path = '/Users/alealcoforado/Documents/Projetos/Datasets/{which_dataset}/'.format(which_dataset=zeroberto_config['dataset'])
    else:
        dataset_path = local_path
    folder = pathlib.Path(dataset_path)
    folder.mkdir(parents=True,exist_ok=True)
    results_filename = "predictions_and_probabilities_test_{agora}.csv".format(agora=agora)
    zeroberto_config_filename = "zeroshot_config_test_{agora}.csv".format(agora=agora)
    print(results_filename)
    print(zeroberto_config_filename)
    pd.DataFrame(results).to_csv(dataset_path+results_filename)    
    pd.DataFrame([zeroberto_config]).to_csv(dataset_path+zeroberto_config_filename)
    return agora  