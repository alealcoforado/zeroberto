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

    def forward(self, x, return_embeddings=False):
        doc_emb = self.embedding_model.encode(x, convert_to_tensor=True, normalize_embeddings=self.normalize_embeddings,
                                             device=self.device)
        stacked_tensors=[]
        for i in range(self.queries.shape[0]):
            stacked_tensors.append(torch.sum(doc_emb * self.queries[i], axis=-1)) # Hadamard product (element-wise)
        logits = torch.stack(stacked_tensors, dim=-1)
        z = self.softmax(logits)
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

    def __call__(self, text_list, probabilities, embeddings, labels=None, n=8):
        if self.selection_strategy == "top_n":
            return self._get_top_n_data(text_list, probabilities, labels, n)
        if self.selection_strategy == "intraclass_clustering":
            # print("Len text and embeds:",len(text_list),len(embeddings))
            return self._get_intraclass_clustering_data(text_list, probabilities, labels, embeddings, n)

    def _get_top_n_data(self, text_list, probabilities,labels,n):
        # QUESTION: está certo ou deveria pegar os top n de cada classe? faz diferença?
        # Aqui permite que o mesmo exemplo entre para duas classes
        top_prob, index = torch.topk(probabilities, k=n, dim=0)
        top_prob, index = top_prob.T, index.T
        n_classes = probabilities.shape[-1]
        x_train = []
        y_train = []
        labels_train = []
        for i in range(len(index)):
            for ind in index[i]:
                y_train.append(i)
                x_train.append(text_list[ind])
                if labels:
                    labels_train.append(labels[ind])
        return x_train, y_train, labels_train
    
    def _get_intraclass_clustering_data(self, text_list, probabilities, true_labels, embeddings, n,
                                         clusterer='hdbscan', leaf_size=20, min_cluster_size=10):
        
        label_results = [np.argmax(lista) for lista in (np.array(probabilities.cpu()))]
        prob_results = [np.max(lista) for lista in (np.array(probabilities.cpu()))]

        unique_labels = list(set(label_results))
        unique_labels.sort()
        all_labels_selected_data = []
        for label in unique_labels:
            this_label_selected_data = []
            this_label_indexes = [i for i in range(len(label_results)) if label_results[i] == label]
            # print(this_label_indexes)
            this_label_text_list =  [text_list[i] for i in this_label_indexes]
            this_label_embeddings =  [embeddings[i] for i in this_label_indexes]
            this_label_probs =  [prob_results[i] for i in this_label_indexes]
            this_label_true_labels = [true_labels[i] for i in this_label_indexes]
            this_label_label_results = [label_results[i] for i in this_label_indexes]


            print("Clustering class {}.".format(label))
            # logger.info("Clustering class {}.")

            this_label_clusters = self._clusterer_fit_predict(clusterer, this_label_embeddings, leaf_size, min_cluster_size) 
            # print(len(this_label_clusters),len(this_label_indexes),len(this_label_text_list),len(this_label_embeddings),len(this_label_probs))
            unique_clusters = list(set(this_label_clusters))
            unique_clusters.sort()
            # print(unique_clusters)
            all_clusters_sorted_lists = []

            # organize by sorting and zipping lists, 1 list for each cluster found
            for cluster in unique_clusters:
                this_cluster_indexes = [i for i in range(len(this_label_clusters)) if this_label_clusters[i] == cluster]
                this_cluster_probs =  [this_label_probs[i] for i in this_cluster_indexes]
                this_cluster_texts = [this_label_text_list[i] for i in this_cluster_indexes]
                this_cluster_true_labels = [this_label_true_labels[i] for i in this_cluster_indexes]
                this_cluster_label_results = [this_label_label_results[i] for i in this_cluster_indexes]
                zipped_lists = (list(zip(this_cluster_probs,this_cluster_indexes,this_cluster_true_labels,this_cluster_label_results,this_cluster_texts)))
                zipped_lists.sort(reverse=True)
                print(f"Cluster {cluster}: {len(this_cluster_indexes)} documents assigned")
                # print(zipped_lists)

                all_clusters_sorted_lists.append(zipped_lists)
            # selects data iteratively, 1 from each cluster from biggest to smallest cluster, 
            # following highest probability order inside each cluster
            while len(this_label_selected_data) < n:
                for sorted_list in all_clusters_sorted_lists:
                    if len(sorted_list) > 0:
                        # print(sorted_list)
                        selected_element = sorted_list[0]
                        # print(label,selected_element)
                        this_label_selected_data.append(selected_element)
                        sorted_list.pop(0)
                        # print(sorted_list)
                        if len(this_label_selected_data) == n:
                            break
                if len(all_clusters_sorted_lists) == 0 or all_clusters_sorted_lists == [[]]:
                    print("Not enough data to sample for label {label}: {n} samples expected, but only got {this_label_n}".format(
                        label=label,n=n,this_label_n=len(this_label_selected_data)))
                    break
            all_labels_selected_data.append(this_label_selected_data)

        flat_selected_data = [item for sublist in all_labels_selected_data for item in sublist]

        probs,train_indices,true_labels,predicted_labels,texts = zip(*flat_selected_data)

        x_train = texts
        y_train = predicted_labels 
        labels_train = true_labels

        return x_train, y_train, labels_train

    def _clusterer_fit_predict(self,clusterer,embeddings,leaf_size,min_cluster_size):
        if clusterer=='hdbscan':
            clusterer_model = hdbscan.HDBSCAN(leaf_size=leaf_size, min_cluster_size=min_cluster_size)
        tensor_embeddings = [] # TO DO melhorar
        for emb in embeddings: # TO DO melhorar
          tensor_embeddings.append(torch.Tensor(emb))# TO DO melhorar
        embeddings = np.array(torch.stack((tensor_embeddings)).cpu())
        clusters = clusterer_model.fit_predict(embeddings)
        # logger.info("Found {} clusters.".format(len(list(set(clusters)))))
        print(f"Found {len(list(set(clusters)))} clusters.")
        return clusters

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