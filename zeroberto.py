import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import evaluation_metrics
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
import tqdm
tqdm.tqdm()
import nltk
import datasets_handler
# nltk.download('punkt')
from sklearn.neighbors import KNeighborsClassifier
from datasets import Dataset
# -*- coding: utf-8 -*-
# !pip install sentence_transformers

def getDevice():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return device

device = getDevice()
# print(torch.device)

class ZeroBERTo(nn.Module):
  def __init__(self, classes_list, embeddingModel = None, contrastiveModel = None, hypothesis_template = "{}.",
              clusterModel = None, random_state = 422, labeling_dataset = None, 
              labeling_method='dotproduct',config=None):
    
    super(ZeroBERTo, self).__init__()
#    self.embeddingModel = SentenceTransformer('sentence-transformers/nli-roberta-base-v2')
    if embeddingModel == None:
       self.embeddingModel = SentenceTransformer(config['similarity_model'],device=getDevice())
    else: self.embeddingModel = SentenceTransformer(embeddingModel,device=device)
   
    if contrastiveModel == None:
       self.contrastiveModel = SetFitModel.from_pretrained(config['setfit_model'], 
                                                          #  use_differentiable_head=True,
                                        # head_params={"out_features":len(classes_list)},
                                        )
    else: self.contrastiveModel = SetFitModel.from_pretrained(contrastiveModel) 
                                                              # use_differentiable_head=True,
                                        # head_params={"out_features":len(classes_list)})
   
    self.classes = classes_list
   # self.embeddingModel = SentenceTransformer("ricardo-filho/bert-base-portuguese-cased-nli-assin-2")
    self.hypothesis_template = config['template']
    print(1)
    self.queries = self.create_queries(self.classes,self.hypothesis_template)
    self.labeling_method = config['labeling_method']
    self.random_state = config['random_state']
    # self.initial_centroids = initial_centroids
    self.labeling_dataset = labeling_dataset
    print(3)
    # self.initial_centroids = np.array(self.queries,dtype=np.double)

    # if clusterModel == None:
    #   self.clusterModel = KMeans(n_clusters=len(self.classes), n_init=1,
    #                               init=self.initial_centroids,max_iter = 600, random_state=self.random_state)
    # else: self.clusterModel = clusterModel
    self.config = config
    self.softmax = nn.Softmax(dim=1).to(device)
    print(4)

  def buildTrainer(self,train_dataset):
      
      self.column_mapping = {self.config['data_col']: "text", 'class_code': "label"}
      eval_data = self.labeling_dataset[['text','class_code']].to_dict('list')

      # self.trainer = SetFitTrainer(
      #   model=self.contrastiveModel,
      #   train_dataset=train_dataset,
      #   # eval_dataset=Dataset.from_dict(self.labeling_dataset[self.column_mapping.keys()].to_dict()),
      #   eval_dataset = Dataset.from_dict(eval_data),

      #   loss_class=CosineSimilarityLoss,
      #   num_iterations=self.config["num_pairs"], # Number of text pairs to generate for contrastive learning
      #   num_epochs=self.config["num_epochs"], # Number of epochs to use for contrastive learning
      #   column_mapping = self.column_mapping, # NÃO mudar 
      #   batch_size=self.config["batch_size"],
      #   )
      
      self.trainer = SetFitTrainer(
        model= self.contrastiveModel,
        train_dataset=train_dataset,
        eval_dataset = Dataset.from_dict(self.labeling_dataset[['text','class_code']].to_dict('list')),
        loss_class=CosineSimilarityLoss,
        num_iterations=5,
        column_mapping={"text": "text", "class_code": "label"},
        batch_size = 8,
        )
      
  def contrastive_train(self):
    if self.config['keep_body_frozen_setfit']:
      self.trainer.freeze() # Freeze the head
      # self.trainer.train() # Train only the body
      self.trainer.unfreeze(keep_body_frozen=self.config['keep_body_frozen_setfit'])
      print("freeze")

    self.trainer.train(
        # body_learning_rate=1e-5, # The body's learning rate
        # learning_rate=1e-2, # The head's learning rate
        # l2_weight=0.1, # Weight decay on **both** the body and head. If `None`, will use 0.01.
        )
  
  def fit (self, sentences, batch_size = 8, epochs = 10):
    ##### Implementation of TSDAE - Unsupervised Learning for Transformers
    # https://www.sbert.net/examples/unsupervised_learning/TSDAE/README.html

    # Create the special denoising dataset that adds noise on-the-fly
    train_data = datasets.DenoisingAutoEncoderDataset(sentences)

    # DataLoader to batch your data
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(self.embeddingModel,decoder_name_or_path='sentence-transformers/stsb-xlm-r-multilingual',tie_encoder_decoder=True) # decoder_name_or_path=model_name

    # Call the fit method
    self.embeddingModel.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True )

  def create_queries(self, classes_list, hypothesis):

    queries = []
    for c in classes_list:
        queries.append(hypothesis.format(c))
    # print(queries)
    emb_queries = self.embeddingModel.encode(sentences=queries, convert_to_tensor=True, normalize_embeddings=True,device=device)
    # print(emb_queries)
    print("queries")
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
    doc_emb = self.embeddingModel.encode(splitted_doc,convert_to_tensor=True, normalize_embeddings=True,device=device)
    return doc_emb

  def forward(self, x):
    # splitted_doc = self.textSplitter(x)
    splitted_doc = x
    doc_emb = self.embeddingModel.encode(splitted_doc,convert_to_tensor=True, normalize_embeddings=True,device=device)
    logits = torch.sum(doc_emb*self.queries, axis=-1)
    z = self.softmax(logits.unsqueeze(0))
    return z

  def get_top_n_results(self,dataframe_results,ascending=False,top_n=1):
      df_top_n = dataframe_results.sort_values(['top_probability','prediction'], ascending=ascending).groupby('prediction').head(top_n)
      # print(df_top_n['top_probability'][:1])
      accuracy_top_n = evaluation_metrics.get_metric(df_top_n['prediction_code'].to_list(),df_top_n['class_code'].to_list())
      acc = "top {}: ".format(top_n) + str(accuracy_top_n)
      print(acc)
      return 
  
  def runLabeling(self):
    preds = []
    t0 = time.time()
    for text in self.labeling_dataset[self.config['data_col']].to_list():
        pred = (self(text).cpu().numpy()[0])
        preds.append(pred)
        # print(pred)
        if len(preds) % 500 == 0:
            t1 = time.time()-t0
            eta = ((t1)/len(preds))*len(self.labeling_dataset)/60
            print("Preds:",len(preds)," - Total time:",round(t1,2),"seconds"+" - ETA:",round( eta ,1),"minutes")
    self.labeling_results = preds
    return preds

  def evaluateLabeling(self,ascending=False,top_n=16):
    df_probs = pd.DataFrame(self.labeling_results,columns=self.classes,index=self.labeling_dataset.index)

    if self.labeling_method == "kmeans":
      label_results = df_probs.apply(lambda row : row.idxmin(),axis=1)
      prob_results = df_probs.apply(lambda row : row.min(),axis=1)
      ascending = True
    if self.labeling_method == "dotproduct":
      label_results = df_probs.apply(lambda row : row.idxmax(),axis=1)
      prob_results = df_probs.apply(lambda row : row.max(),axis=1)
      ascending = False ### 
    # label_results_df = pd.Series(label_results,name='prediction').sort_index()
    # true_labels_df = self.labeling_dataset['class'].sort_index()
    label_results_df = pd.Series(label_results,name='prediction')
    true_labels_df = self.labeling_dataset['class'].sort_index()

    final_result_df = pd.concat([true_labels_df,label_results_df],axis=1)
    final_result_df_encoded = evaluation_metrics.Encoder(final_result_df,['prediction','class'])

    df_predictions_probs = pd.concat([final_result_df_encoded,
                                      pd.Series(prob_results,name='top_probability')],axis=1)
    for i in range(top_n):
       self.get_top_n_results(df_predictions_probs,ascending=ascending,top_n=i+1)
    self.get_top_n_results(df_predictions_probs,ascending=ascending,top_n=len(self.labeling_dataset))

    cols_to_add = ['prediction_code','top_probability']
    self.labeling_dataset =  pd.concat([self.labeling_dataset.drop(columns=cols_to_add,errors='ignore'),
                                        df_predictions_probs[cols_to_add]],axis=1)
    
  def getLabelingMetrics(self):
     labeling_metrics = evaluation_metrics.get_metrics(self.labeling_dataset['prediction_code'].to_list()
                                                       ,self.labeling_dataset['class_code'].to_list())
     self.labeling_metrics = labeling_metrics

  def saveLabelingResults(self,local_path = None):
     self.config['exec_time'] = evaluation_metrics.saveZeroshotResults(self.config,self.labeling_dataset,local_path=local_path)

  # def loadLabelingResults(self):
  #   zeroshot_previous_data = datasets_handler.getZeroshotPreviousData(
  #      self.config['which_dataset'], self.config['class_col'],top_n= self.config['top_n'],
  #      exec_time=self.config['exec_time'])
    
  #   raw_data_final, self.config['new_class_col'] = datasets_handler.mergeLabelingToDataset(
  #      raw_data,zeroshot_previous_data,self.config['class_col'])

  def getPredictions(self):
    setfit_trainer = self.trainer
    setfit_trainer._validate_column_mapping(setfit_trainer.eval_dataset)
    eval_dataset = setfit_trainer.eval_dataset

    if setfit_trainer.column_mapping is not None:
        eval_dataset = setfit_trainer._apply_column_mapping(setfit_trainer.eval_dataset, setfit_trainer.column_mapping)

    x_test = eval_dataset["text"]
    print("Running predictions on {} sentences.".format(len(x_test)))
    y_pred = setfit_trainer.model.predict(x_test).cpu()
    self.y_pred = y_pred #### xxx remover depois
    # return y_pred 
  



############################################################
####################### FUNCTIONS ##########################
############################################################

def runZeroberto(model,data,config):
    preds = []
    t0 = time.time()
    for text in data:
        pred = (model(text).cpu().numpy()[0])
        preds.append(pred)
        # print(pred)
        if len(preds) % 100 == 0:
            t1 = time.time()-t0
            eta = ((t1)/len(preds))*len(data)/60
            print("Preds:",len(preds)," - Total time:",round(t1,2),"seconds"+" - ETA:",round( eta ,1),"minutes")
    return preds   



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