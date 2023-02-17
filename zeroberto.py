import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import evaluation_metrics
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
# -*- coding: utf-8 -*-
# !pip install sentence_transformers
def getDevice():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return device

# print(torch.device)

class ZeroBERTo(nn.Module):
  def __init__(self, classes_list, embeddingModel=None, hypothesis_template="{}.",
              clusterModel = None, random_state = 422, train_dataset = None, labeling_method='dotproduct'):
    super(ZeroBERTo, self).__init__()
#    self.embeddingModel = SentenceTransformer('sentence-transformers/nli-roberta-base-v2')
    if embeddingModel == None:
       self.embeddingModel = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual',device=getDevice())
    else:
       self.embeddingModel = embeddingModel
   # self.embeddingModel = SentenceTransformer("ricardo-filho/bert-base-portuguese-cased-nli-assin-2")
    self.labeling_method = labeling_method
    self.hypothesis_template = hypothesis_template
    self.classes = classes_list
    self.queries = self.create_queries(self.classes,self.hypothesis_template)
    self.random_state = random_state
    # self.initial_centroids = initial_centroids
    self.train_dataset = train_dataset
    # self.classes_emb = self.encode(classes)
    self.initial_centroids = np.array(self.queries,dtype=np.double)

    if clusterModel == None:
      self.clusterModel = KMeans(n_clusters=len(self.classes), n_init=1,
                                  init=self.initial_centroids,max_iter = 600, random_state=self.random_state)
    else: self.clusterModel = clusterModel

    self.softmax = nn.Softmax(dim=1)
  
  def create_queries(self, classes_list, hypothesis):

    queries = []
    for c in classes_list:
        queries.append(hypothesis.format(c))
    # print(queries)
    emb_queries = self.embeddingModel.encode(sentences=queries, convert_to_tensor=True, normalize_embeddings=True,device=getDevice())
    # print(emb_queries)
    return emb_queries

  def textSplitter(self, x: str):
    # if not isinstance(x, str):
    #       return  x
    if isinstance(x, str):
       return x.split(".")
    x_set = []
    for paragraph in x:
      x_set.append(paragraph.split("."))
    return x_set

  def encode(self, x):
    # splitted_doc = self.textSplitter(x)
    splitted_doc = x
    doc_emb = self.embeddingModel.encode(splitted_doc,convert_to_tensor=True, normalize_embeddings=True)
    return doc_emb

  def forward(self, x):
    # splitted_doc = self.textSplitter(x)
    splitted_doc = x
    doc_emb = self.embeddingModel.encode(splitted_doc,convert_to_tensor=True, normalize_embeddings=True)
    logits = torch.sum(doc_emb*self.queries, axis=-1)
    z = self.softmax(logits.unsqueeze(0))
    return z

  def evaluateLabeling(self, y_pred_probs,ascending=False):
    df_probs = pd.DataFrame(y_pred_probs,columns=self.classes,index=self.train_dataset.index)

    if self.labeling_method == "kmeans":
      label_results = df_probs.apply(lambda row : row.idxmin(),axis=1)
      prob_results = df_probs.apply(lambda row : row.min(),axis=1)
      ascending = True
    if self.labeling_method == "dotproduct":
      label_results = df_probs.apply(lambda row : row.idxmax(),axis=1)
      prob_results = df_probs.apply(lambda row : row.max(),axis=1)
      ascending = False
    # label_results_df = pd.Series(label_results,name='prediction').sort_index()
    # true_labels_df = self.train_dataset['class'].sort_index()
    label_results_df = pd.Series(label_results,name='prediction')
    true_labels_df = self.train_dataset['class'].sort_index()


    final_result_df = pd.concat([true_labels_df,label_results_df],axis=1)
    final_result_df_encoded = evaluation_metrics.Encoder(final_result_df,['prediction','class'])

    df_predictions_probs = pd.concat([final_result_df_encoded,
                                      pd.Series(prob_results,name='top_probability')],axis=1)
    for i in range(16):
       self.get_top_n_results(df_predictions_probs,ascending=ascending,top_n=i+1)
    self.get_top_n_results(df_predictions_probs,ascending=ascending,top_n=len(self.train_dataset))

    return df_predictions_probs
  
  def get_top_n_results(self,dataframe_results,ascending=False,top_n=1):
      df_top_n = dataframe_results.sort_values(['top_probability','prediction'], ascending=ascending).groupby('prediction').head(top_n)
      # print(df_top_n['top_probability'][:1])
      accuracy_top_n = evaluation_metrics.get_metric(df_top_n['prediction_code'].to_list(),df_top_n['class_code'].to_list())
      acc = "top {}: ".format(top_n) + str(accuracy_top_n)
      print(acc)
      return 
  


def runZeroberto(model,data,config):
    preds = []
    t0 = time.time()
    for text in data:
        pred = (model(text).numpy()[0])
        preds.append(pred)
        # print(pred)
        if len(preds) % 100 == 0:
            t1 = time.time()-t0
            eta = ((t1)/len(preds))*len(data)/60
            print("Preds:",len(preds)," - Total time:",round(t1,2),"seconds"+" - ETA:",round( eta ,1),"minutes")
    return preds   

def getPredictions(setfit_trainer):
  setfit_trainer._validate_column_mapping(setfit_trainer.eval_dataset)
  eval_dataset = setfit_trainer.eval_dataset

  if setfit_trainer.column_mapping is not None:
      eval_dataset = setfit_trainer._apply_column_mapping(setfit_trainer.eval_dataset, setfit_trainer.column_mapping)

  x_test = eval_dataset["text"]
  print("Running predictions on {} sentences.".format(len(x_test)))
  y_pred = setfit_trainer.model.predict(x_test)
  return y_pred

def getProbabilities(setfit_trainer):
    setfit_trainer._validate_column_mapping(setfit_trainer.eval_dataset)
    eval_dataset = setfit_trainer.eval_dataset

    if setfit_trainer.column_mapping is not None:
        eval_dataset = setfit_trainer._apply_column_mapping(setfit_trainer.eval_dataset, setfit_trainer.column_mapping)
    x_test = (eval_dataset["text"])
    print("Running predictions (with probabilities) on {} sentences.".format(len(x_test)))
    
    y_pred = setfit_trainer.model.predict_proba(x_test)
    return y_pred

def mapper(item_tuple):
  index, item = item_tuple
  return dict(zip(item["labels"], item["scores"]))

def runZeroShotPipeline(classifier,data,config):
    # labels =  list(dict_classes_folha.values())
    goal_count = dict(zip(config['labels'],np.zeros(len(config['labels']))))
    print("# data:",len(data))
    print(config)
    preds = []
    indexes = []
    t0 = time.time()
    for text in data:
        pred = classifier(text, candidate_labels=config['labels'], 
                            hypothesis_template=config['template'], multi_label=False)
        preds.append(pred)

        if config['method'] == "probability_threshold":
            top_prob = pred['scores'][0]
            top_label = pred['labels'][0]

            if (top_prob>=config['prob_goal']):
                goal_count[top_label] += 1
            if len(preds) % 50 == 0:
                print("Preds:",len(preds)," - Total time:",round(time.time()-t0,2),"seconds")
                print(goal_count)
            if all(count >= config['top_n_goal'] for count in list(goal_count.values())):
                break  ### stop loop if goal is met

        if config['method'] == "top_n_goal":
            top_prob = pred['scores'][0]
            top_label = pred['labels'][0]
            if len(preds) % 50 == 0:
                t1 = time.time()-t0
                eta = ((t1)/len(preds))*len(data)/60
                print("Preds:",len(preds)," - Total time:",round(t1,2),"seconds"+" - ETA:",round( eta ,1),"minutes")

    print("Total Predictions:",len(preds))
    return preds

def formatZeroShotResults(results):
  df_results = pd.DataFrame(pd.Series(results).to_dict())
  label_probabilities = list(map(mapper,enumerate(results)))
  label_probabilities_df = pd.DataFrame(label_probabilities,index=df_results.columns)
  
  return label_probabilities_df