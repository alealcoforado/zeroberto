import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union


# Google Colab runs on Python 3.7, so we need this to be compatible
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import joblib
import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

import pandas as pd
import numpy as np


from setfit import logging
from setfit import SetFitModel, SetFitHead

import hdbscan

if TYPE_CHECKING:
    from numpy import ndarray


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MODEL_HEAD_NAME = "model_head.pkl"

class FirstShotModel(nn.Module):
    def __init__(
            self,
            embedding_model: SentenceTransformer,
            classes_list: List[str],
            hypothesis_template: Optional[str] = "This is {}.",
            normalize_embeddings: Optional[bool] = True,
            device: Optional[str] = None,
    ) -> None:

        super(FirstShotModel, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_model = embedding_model
        self.normalize_embeddings = normalize_embeddings
        self.classes_list = classes_list
        self.hypothesis_template = hypothesis_template
        self.queries = self._create_queries(self.classes_list, self.hypothesis_template)
        self.softmax = nn.Softmax(dim=1).to(self.device)

    def forward(self, x, return_embeddings=False, temperature=1.0):
        doc_emb = self.embedding_model.encode(x, convert_to_tensor=True, normalize_embeddings=self.normalize_embeddings,
                                             device=self.device)
        stacked_tensors=[]
        for i in range(self.queries.shape[0]):
            stacked_tensors.append(torch.sum(doc_emb * self.queries[i], axis=-1)) # Hadamard product (element-wise)
        logits = torch.stack(stacked_tensors, dim=-1)
        z = self.softmax(logits/temperature)
        return (z, doc_emb) if return_embeddings else z

    def _create_queries(self, classes_list, hypothesis):
        queries = []
        for c in classes_list:
            queries.append(hypothesis.format(c))
        emb_queries = self.embedding_model.encode(sentences=queries, convert_to_tensor=True, normalize_embeddings=self.normalize_embeddings,
                                                 device=self.device)
        return emb_queries

    # def _text_splitter(self, x: str):
    #     # if not isinstance(x, str):
    #     #       return  x
    #     if isinstance(x, str):
    #         return x.split(".")
    #     x_set = []
    #     for paragraph in x:
    #         x_set.append(paragraph.split("."))
    #     return x_set

    # def encode(self, x):
    #     if (self.average_sentence_embeddings == True):
    #         if '.' in x:
    #             sentences = self._text_splitter(x)
    #             embeddings = []
    #             for sentence in sentences:
    #                 encoded_sentence = self.embedding_model.encode(sentence, convert_to_tensor=True,
    #                                                               normalize_embeddings=True)
    #                 embeddings.append(encoded_sentence)
    #             return torch.mean(torch.stack(embeddings), dim=0)
    #         else:
    #             doc_emb = self.embedding_model.encode(x, convert_to_tensor=True, normalize_embeddings=True)
    #             return doc_emb
    #     else:
    #         # splitted_doc = self.textSplitter(x)
    #         splitted_doc = x
    #         doc_emb = self.embedding_model.encode(splitted_doc, convert_to_tensor=True, normalize_embeddings=True,
    #                                              device=device)
    #     return doc_emb

class ZeroBERToDataSelector:
    def __init__(self, selection_strategy="top_n"):
        self.selection_strategy = selection_strategy

    def __call__(self, text_list, probabilities, embeddings, labels=None, n=8, discard_indices = [], selection_strategy=None):
        if not selection_strategy:
            selection_strategy = self.selection_strategy
        if selection_strategy == "top_n":
            return self._get_top_n_data(text_list, probabilities, labels, n, discard_indices)
        if selection_strategy == "intraclass_clustering":
            return self._get_intraclass_clustering_data(text_list, probabilities, labels, embeddings, n, discard_indices)

    def _get_top_n_data(self, text_list, probs,labels,n,discard_indices = []):
        # QUESTION: está certo ou deveria pegar os top n de cada classe? faz diferença?
        # Aqui permite que o mesmo exemplo entre para duas classes
        # probs = probabilities.detach().clone()
        if len(discard_indices) > 0: ## TO DO: melhorar essa parte
          if type(discard_indices[0]) != type(int(1)):
            discard_indices = [tensor.item() for tensor in discard_indices]
            probs = probs.float()
       
            # Set discard item probabilities to -1.0
            probs[discard_indices] = -1.0*torch.ones(len(discard_indices), probs.shape[-1])
        top_prob, index = torch.topk(probs, k=n, dim=0)
        top_prob, index = top_prob.T, index.T
        x_train = []
        y_train = []
        labels_train = []
        training_indices = []
        probs_train =[]
        for i in range(len(index)):
            for ind in index[i]:
                y_train.append(i)
                x_train.append(text_list[ind])
                probs_train.append(probs[i])
                if labels:
                    labels_train.append(labels[ind])
                training_indices.append(ind)
        return x_train, y_train, labels_train, training_indices, probs_train
    
    def _get_intraclass_clustering_data(self, text_list, probabilities, true_labels, embeddings, n, discard_indices = [],
                                         clusterer='hdbscan', leaf_size=20, min_cluster_size=10):

        discard_indices = set(discard_indices)
        prob_results, label_results = torch.max(probabilities, axis=-1)

        selected_data = []
        for label in range(probabilities.shape[-1]):
            label_selected_data = []

            # Retrieve indices for label
            label_indices = (label_results == label).nonzero().squeeze()
            if len(label_indices) < n:
                print(f"Not enough data to sample for label {label}: {n} samples expected, but only got {len(label_indexes)}")
                # Throws error
                break

            label_embeddings = embeddings[label_indices]


            print("Clustering class {}.".format(label))

            label_clusters = self._clusterer_fit_predict(clusterer, label_embeddings, leaf_size, min_cluster_size) 

            unique_clusters = list(set(label_clusters))
            unique_clusters.sort() # Here should be sorted by density, no? - TO DO

            clustered_docs = {}

            # sort docs by probabilities for each cluster found
            for cluster in unique_clusters:
                cluster_indices = label_indices[(label_clusters == cluster).nonzero().squeeze()]
                cluster_probs = prob_results[cluster_indices]
                cluster_probs, cluster_probs_sorted_ind = torch.sort(cluster_probs, descending=True)
                cluster_indices = cluster_indices[cluster_probs_sorted_ind]

                clustered_docs[cluster] = [item for item in cluster_indices.tolist() if item not in discard_indices]

            # selects data iteratively, 1 from each cluster from biggest to smallest cluster, 
            # following highest probability order inside each cluster
            while len(label_selected_data) < n:
                for cluster in unique_clusters:
                    if len(clustered_docs[cluster]) > 0:
                        selected_element = clustered_docs[cluster].pop(0)
                        label_selected_data.append(selected_element)
                        if len(label_selected_data) == n:
                            break

            selected_data.append(label_selected_data)

        selected_data = [item for sublist in selected_data for item in sublist]

        x_train = [text_list[i] for i in selected_data]
        y_train = label_results[selected_data].tolist()
        labels_train = [true_labels[i] for i in selected_data]
        probs_train = [probabilities[i] for i in selected_data]

        return x_train, y_train, labels_train, selected_data, probs_train

    def _clusterer_fit_predict(self,clusterer,embeddings,leaf_size,min_cluster_size):
        if clusterer=='hdbscan':
            clusterer_model = hdbscan.HDBSCAN(leaf_size=leaf_size, min_cluster_size=min_cluster_size)
        embeds = torch.Tensor(embeddings).cpu()
        clusters = clusterer_model.fit_predict(embeds)
        # logger.info("Found {} clusters.".format(len(list(set(clusters)))))
        print(f"Found {len(list(set(clusters)))} clusters.")
        return torch.IntTensor(clusters)

    # def _get_intraclass_clustering_data(self,text_list,probabilities,labels,n)
        

class ZeroBERToModel(SetFitModel):
    """A ZeroBERTo model with integration to the Hugging Face Hub. """

    def __init__(
            self,
            model_body: Optional[SentenceTransformer] = None,
            first_shot_model: Optional[FirstShotModel] = None,
            model_head: Optional[Union[SetFitHead, LogisticRegression]] = None,
            multi_target_strategy: Optional[str] = None,
            l2_weight: float = 1e-2,
            normalize_embeddings: bool = False,
    ) -> None:
        super(ZeroBERToModel, self).__init__(model_body,model_head,multi_target_strategy,l2_weight,normalize_embeddings)

        # If you don't give a first shot model, we use the body - TO REVIEW
        #if not first_shot_model:
            #first_shot_model = model_body

        self.first_shot_model = first_shot_model

    @classmethod
    def _from_pretrained( # Done
            cls,
            model_id: str,
            first_shot_model_id: Optional[str] = None,
            use_first_shot: bool = True,
            classes_list: Optional[List[str]] = None,
            hypothesis_template: Optional[str] = "{}.",
            revision: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: Optional[bool] = None,
            proxies: Optional[Dict] = None,
            resume_download: Optional[bool] = None,
            local_files_only: Optional[bool] = None,
            use_auth_token: Optional[Union[bool, str]] = None,
            multi_target_strategy: Optional[str] = None,
            use_differentiable_head: bool = False,
            normalize_embeddings: bool = False,
            **model_kwargs,
    ) -> "ZeroBERToModel":
        model_body = SentenceTransformer(model_id, cache_folder=cache_dir, use_auth_token=use_auth_token)
        target_device = model_body._target_device
        model_body.to(target_device)  # put `model_body` on the target device

        if os.path.isdir(model_id):
            if MODEL_HEAD_NAME in os.listdir(model_id):
                model_head_file = os.path.join(model_id, MODEL_HEAD_NAME)
            else:
                logger.info(
                    f"{MODEL_HEAD_NAME} not found in {Path(model_id).resolve()},"
                    " initialising classification head with random weights."
                    " You should TRAIN this model on a downstream task to use it for predictions and inference."
                )
                model_head_file = None
        else:
            try:
                model_head_file = hf_hub_download(
                    repo_id=model_id,
                    filename=MODEL_HEAD_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    use_auth_token=use_auth_token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                logger.info(
                    f"{MODEL_HEAD_NAME} not found on HuggingFace Hub, initialising classification head with random weights."
                    " You should TRAIN this model on a downstream task to use it for predictions and inference."
                )
                model_head_file = None

        if model_head_file is not None:
            model_head = joblib.load(model_head_file)
            use_first_shot = False # Not use first shot model if there is a trained head
        else:
            head_params = model_kwargs.get("head_params", {})
            if use_differentiable_head:
                if multi_target_strategy is None:
                    use_multitarget = False
                else:
                    if multi_target_strategy in ["one-vs-rest", "multi-output"]:
                        use_multitarget = True
                    else:
                        raise ValueError(
                            f"multi_target_strategy '{multi_target_strategy}' is not supported for differentiable head"
                        )
                # Base `model_head` parameters
                # - get the sentence embedding dimension from the `model_body`
                # - follow the `model_body`, put `model_head` on the target device
                base_head_params = {
                    "in_features": model_body.get_sentence_embedding_dimension(),
                    "device": target_device,
                    "multitarget": use_multitarget,
                }
                model_head = SetFitHead(**{**head_params, **base_head_params})
            else:
                clf = LogisticRegression(**head_params)
                if multi_target_strategy is not None:
                    if multi_target_strategy == "one-vs-rest":
                        multilabel_classifier = OneVsRestClassifier(clf)
                    elif multi_target_strategy == "multi-output":
                        multilabel_classifier = MultiOutputClassifier(clf)
                    elif multi_target_strategy == "classifier-chain":
                        multilabel_classifier = ClassifierChain(clf)
                    else:
                        raise ValueError(f"multi_target_strategy {multi_target_strategy} is not supported.")

                    model_head = multilabel_classifier
                else:
                    model_head = clf

        # Create First Shot model
        if use_first_shot:
            if not classes_list:
                # Throws Error
                pass
            if first_shot_model_id:
                s_transf_first_shot = SentenceTransformer(first_shot_model_id,
                                                        cache_folder=cache_dir,
                                                        use_auth_token=use_auth_token)
                target_device = s_transf_first_shot._target_device
                s_transf_first_shot.to(target_device)  # put `first_shot_model` on the target device
            else:
                s_transf_first_shot = model_body
            first_shot_model = FirstShotModel(embedding_model=s_transf_first_shot,
                                              classes_list=classes_list,
                                              hypothesis_template=hypothesis_template
                                              )
        else:
            first_shot_model = None

  

        return cls(
            model_body=model_body,
            first_shot_model=first_shot_model,
            model_head=model_head,
            multi_target_strategy=multi_target_strategy,
            normalize_embeddings=normalize_embeddings,
        )

    def reset_model_head(self, **model_kwargs):
        head_params = model_kwargs.get("head_params", {})
        target_device = self.model_body._target_device

        if type(self.model_head) is SetFitHead: #use_differentiable_head
            if self.multi_target_strategy is None:
                use_multitarget = False
            else:
                if self.multi_target_strategy in ["one-vs-rest", "multi-output"]:
                    use_multitarget = True
                else:
                    raise ValueError(
                        f"multi_target_strategy '{self.multi_target_strategy}' is not supported for differentiable head"
                    )
            # Base `model_head` parameters
            # - get the sentence embedding dimension from the `model_body`
            # - follow the `model_body`, put `model_head` on the target device
            base_head_params = {
                "in_features": self.model_body.get_sentence_embedding_dimension(),
                "device": target_device,
                "multitarget": use_multitarget,
            }
            model_head = SetFitHead(**{**head_params, **base_head_params})
        else:
            clf = LogisticRegression(**head_params)
            if self.multi_target_strategy is not None:
                if self.multi_target_strategy == "one-vs-rest":
                    multilabel_classifier = OneVsRestClassifier(clf)
                elif self.multi_target_strategy == "multi-output":
                    multilabel_classifier = MultiOutputClassifier(clf)
                elif self.multi_target_strategy == "classifier-chain":
                    multilabel_classifier = ClassifierChain(clf)
                else:
                    raise ValueError(f"multi_target_strategy {self.multi_target_strategy} is not supported.")

                model_head = multilabel_classifier
            else:
                model_head = clf
        self.model_head = model_head

    def reset_model_body(self,model_id="sentence-transformers/paraphrase-mpnet-base-v2"):
        print(f"Reset Model to checkpoint: {model_id}")
        self.model_body = SentenceTransformer(model_id)
        self.model_body.to(("cuda" if torch.cuda.is_available() else "cpu"))
        


    def predict_proba(self, x_test: List[str],
                      as_numpy: bool = False,
                      return_embeddings: bool = False
                      ) -> Union[torch.Tensor, "ndarray"]:

        embeddings = self.model_body.encode(
            x_test,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=self.has_differentiable_head,
        )

        outputs = self.model_head.predict_proba(embeddings)
        outputs = self._output_type_conversion(outputs, as_numpy=as_numpy)
        return (outputs, embeddings) if return_embeddings else outputs