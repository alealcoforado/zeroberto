import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
# -*- coding: utf-8 -*-
# !pip install sentence_transformers

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ZeroBERTo(nn.Module):
  def __init__(self, classes_list, hypothesis_template):
    super(ZeroBERTo, self).__init__()
    #self.text_splitter

    self.embedding = SentenceTransformer('sentence-transformers/nli-roberta-base-v2', device=device)
    self.queries = self.create_queries(classes_list,hypothesis_template)
    #self.cluster
    self.softmax = nn.Softmax(dim=1)
  
  def create_queries(self, classes_list, hypothesis):

    classes = []
    for c in classes_list:
        classes.append(hypothesis.format(c))

    return self.embedding.encode(sentences=classes, convert_to_tensor=True, normalize_embeddings=True)

  def forward(self, x):
    #splitted_doc = self.text_splitter(x)
    splitted_doc = x
    doc_emb = self.embedding.encode(splitted_doc,convert_to_tensor=True, normalize_embeddings=True)
    logits = torch.sum(doc_emb*self.queries, axis=-1)
    z = self.softmax(logits.unsqueeze(0))
    return z

def runZeroberto(model,data,config):
    preds = []
    t0 = time.time()
    for text in data:
        preds.append(model(text).numpy()[0])

        if len(preds) % 50 == 0:
            t1 = time.time()-t0
            eta = ((t1)/len(preds))*len(data)/60
            print("Preds:",len(preds)," - Total time:",round(t1,2),"seconds"+" - ETA:",round( eta ,1),"minutes")
    return preds

# def predsToDataframe(preds):
   

def getPredictions(setfit_trainer):
  setfit_trainer._validate_column_mapping(setfit_trainer.eval_dataset)
  eval_dataset = setfit_trainer.eval_dataset

  if setfit_trainer.column_mapping is not None:
      eval_dataset = setfit_trainer._apply_column_mapping(setfit_trainer.eval_dataset, setfit_trainer.column_mapping)

  x_test = eval_dataset["text"]
  print("Running predictions on {} sentences.".format(len(x_test)))
  y_pred = setfit_trainer.model.predict(x_test)
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