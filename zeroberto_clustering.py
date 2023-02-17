import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

print(torch.device)
class ZeroBERTo_clustering(nn.Module):
  def __init__(self, classes_list, hypothesis_template):
    super(ZeroBERTo, self).__init__()
    #self.text_splitter

    self.embedding = SentenceTransformer('sentence-transformers/nli-roberta-base-v2')
    # self.embedding = SentenceTransformer("ricardo-filho/bert-base-portuguese-cased-nli-assin-2")
    self.queries = self.create_queries(classes_list,hypothesis_template)
    #self.cluster
    self.clusterizer = KMeans(n_clusters=len(classes_list), n_init=25, max_iter = 600, random_state=422)

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

