import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import numpy as np
import time
from sentence_transformers import InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from torch.utils.data import DataLoader
import torch
import evaluate
from tqdm.auto import trange
from transformers.trainer_utils import set_seed

import pickle

from setfit import SetFitTrainer
from setfit import logging
from setfit.modeling import SupConLoss, sentence_pairs_generation, sentence_pairs_generation_multilabel

if TYPE_CHECKING:
    import optuna
    from datasets import Dataset

    from .modeling_zeroberto import ZeroBERToModel, ZeroBERToDataSelector

from .modeling_zeroberto import UnsupervisedEvaluator


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class ZeroBERToTrainer(SetFitTrainer):
    def __init__(
            self,
            model: Optional["ZeroBERToModel"] = None,
            data_selector: Optional["ZeroBERToDataSelector"] = None,
            train_dataset: Optional["Dataset"] = None,
            eval_dataset: Optional["Dataset"] = None,
            model_init: Optional[Callable[[], "ZeroBERToModel"]] = None,
            metric: Union[str, Callable[["Dataset", "Dataset"], Dict[str, float]]] = "accuracy",
            metric_kwargs: Optional[Dict[str, Any]] = None,
            loss_class=losses.CosineSimilarityLoss,
            num_iterations: int = 20,
            num_setfit_iterations: int = 5,
            num_epochs: int = 1,
            num_body_epochs: int = 1,
            learning_rate: float = 2e-5,
            body_learning_rate: float = 2e-5,
            batch_size: int = 16,
            seed: int = 42,
            column_mapping: Optional[Dict[str, str]] = None,
            use_amp: bool = False,
            warmup_proportion: float = 0.1,
            distance_metric: Callable = BatchHardTripletLossDistanceFunction.cosine_distance,
            margin: float = 0.25,
            samples_per_label: int = 2,
            var_samples_per_label: list = None,
            var_selection_strategy: list = None,
            freeze_head: bool = True,
            freeze_body: bool = False,
            train_first_shot: bool = False,
            allow_resampling: bool = False,
            experiment_name: str = "training_zeroberto",

    ):
        if (warmup_proportion < 0.0) or (warmup_proportion > 1.0):
            raise ValueError(
                f"warmup_proportion must be greater than or equal to 0.0 and less than or equal to 1.0! But it was: {warmup_proportion}"
            )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model_init = model_init
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.loss_class = loss_class
        self.num_iterations = num_iterations
        self.num_setfit_iterations = num_setfit_iterations
        self.num_epochs = num_epochs
        self.num_body_epochs = num_body_epochs
        self.learning_rate = learning_rate
        self.body_learning_rate = body_learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.column_mapping = column_mapping
        self.use_amp = use_amp
        self.warmup_proportion = warmup_proportion
        self.distance_metric = distance_metric
        self.margin = margin
        self.samples_per_label = samples_per_label
        self.var_samples_per_label = var_samples_per_label
        self.var_selection_strategy = var_selection_strategy
        self.train_first_shot = train_first_shot
        self.allow_resampling = allow_resampling
        self.experiment_name = experiment_name

        if self.var_samples_per_label is not None:
            assert len(var_samples_per_label) == num_setfit_iterations, "num_setfit_iterations and length of var_samples_per_label must match"
            # print("Asserting: len(var_samples) = ",len(var_samples_per_label))
        if model is None:
            if model_init is not None:
                model = self.call_model_init()
            else:
                raise RuntimeError("`SetFitTrainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                raise RuntimeError("`SetFitTrainer` requires either a `model` or `model_init` argument, but not both")

        self.model = model
        self.data_selector = data_selector
        self.hp_search_backend = None
        self._freeze = freeze_head  # If True, will train the body only; otherwise, train the body and head
        self.freeze_head = freeze_head
        self.freeze_body = freeze_body
        self.unsup_evaluator = UnsupervisedEvaluator()


    def train(
            self,
            num_epochs: Optional[int] = None,
            num_body_epochs: Optional[int] = None,
            num_setfit_iterations: Optional[int] = None,
            batch_size: Optional[int] = None,
            learning_rate: Optional[float] = None,
            body_learning_rate: Optional[float] = None,
            l2_weight: Optional[float] = None,
            max_length: Optional[int] = None,
            trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
            show_progress_bar: bool = False,
            reset_model_head: bool = True,
            return_history: bool = False,
            var_samples_per_label: list = None,
            var_selection_strategy: list = None,
            allow_resampling: bool = False,
            update_embeddings: bool = False,
            train_first_shot: bool = False,
    ):
        """
        Main training entry point.
        Args:
            num_epochs (`int`, *optional*):
                Temporary change the number of epochs to train the Sentence Transformer body/head for.
                If ignore, will use the value given in initialization.
            batch_size (`int`, *optional*):
                Temporary change the batch size to use for contrastive training or logistic regression.
                If ignore, will use the value given in initialization.
            learning_rate (`float`, *optional*):
                Temporary change the learning rate to use for contrastive training or SetFitModel's head in logistic regression.
                If ignore, will use the value given in initialization.
            body_learning_rate (`float`, *optional*):
                Temporary change the learning rate to use for SetFitModel's body in logistic regression only.
                If ignore, will be the same as `learning_rate`.
            l2_weight (`float`, *optional*):
                Temporary change the weight of L2 regularization for SetFitModel's differentiable head in logistic regression.
            max_length (int, *optional*, defaults to `None`):
                The maximum number of tokens for one data sample. Currently only for training the differentiable head.
                If `None`, will use the maximum number of tokens the model body can accept.
                If `max_length` is greater than the maximum number of acceptable tokens the model body can accept, it will be set to the maximum number of acceptable tokens.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Whether to show a bar that indicates training progress.
            var_samples_per_label (`list`, *optional*, defaults to `None`):
                A list of integers containing the roadmap for the Data Selector. Needs to have length = num_setfit_iterations. Overrides samples_per_label.
                If None, will use samples_per_label.
        """
        set_seed(self.seed)  # Seed must be set before instantiating the model when using model_init.

        if trial:  # Trial and model initialization
            self._hp_search_setup(trial)  # sets trainer parameters and initializes model

        if self.train_dataset is None:
            raise ValueError("Training requires a `train_dataset` given to the `SetFitTrainer` initialization.")

        self._validate_column_mapping(self.train_dataset)
        train_dataset = self.train_dataset
        eval_dataset = self.eval_dataset

        if self.column_mapping is not None:
            # logger.info("Applying column mapping to training dataset")
            train_dataset = self._apply_column_mapping(self.train_dataset, self.column_mapping)
            if eval_dataset:
                eval_dataset = self._apply_column_mapping(self.eval_dataset, self.column_mapping)

        #x_train = train_dataset["text"]
        #y_train = train_dataset["label"]
        if self.loss_class is None:
            logger.warning("No `loss_class` detected! Using `CosineSimilarityLoss` as the default.")
            self.loss_class = losses.CosineSimilarityLoss

        num_epochs = num_epochs or self.num_epochs
        num_body_epochs = num_body_epochs or self.num_body_epochs or num_epochs
        train_first_shot = train_first_shot or self.train_first_shot

        num_setfit_iterations = num_setfit_iterations or self.num_setfit_iterations
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate
        body_learning_rate = body_learning_rate or self.body_learning_rate
        def train_setfit_iteration(last_shot_body_epochs=None, last_shot_head_epochs=None,last_shot_body_learning_rate=None):
            setfit_batch_size = batch_size
            if not self.model.has_differentiable_head or self._freeze or not self.freeze_body:
                # sentence-transformers adaptation
                if self.loss_class in [
                    losses.BatchAllTripletLoss,
                    losses.BatchHardTripletLoss,
                    losses.BatchSemiHardTripletLoss,
                    losses.BatchHardSoftMarginTripletLoss,
                    SupConLoss,
                ]:
                    train_examples = [InputExample(texts=[text], label=label) for text, label in zip(x_train, y_train)]
                    train_data_sampler = SentenceLabelDataset(train_examples, samples_per_label=self.samples_per_label)

                    setfit_batch_size = min(batch_size, len(train_data_sampler))
                    train_dataloader = DataLoader(train_data_sampler, batch_size=setfit_batch_size, drop_last=True)

                    if self.loss_class is losses.BatchHardSoftMarginTripletLoss:
                        train_loss = self.loss_class(
                            model=self.model.model_body,
                            distance_metric=self.distance_metric,
                        )
                    elif self.loss_class is SupConLoss:
                        train_loss = self.loss_class(model=self.model.model_body)
                    else:
                        train_loss = self.loss_class(
                            model=self.model.model_body,
                            distance_metric=self.distance_metric,
                            margin=self.margin,
                        )
                else:
                    train_examples = []

                    for _ in trange(self.num_iterations, desc="Generating Training Pairs", disable=not show_progress_bar):
                        if self.model.multi_target_strategy is not None:
                            train_examples = sentence_pairs_generation_multilabel(
                                np.array(x_train), np.array(y_train), train_examples
                            )
                        else:
                            train_examples = sentence_pairs_generation(
                                np.array(x_train), np.array(y_train), train_examples
                            )

                    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=setfit_batch_size)
                    train_loss = self.loss_class(self.model.model_body)
                num_body_epochs = last_shot_body_epochs or num_epochs
                total_train_steps = len(train_dataloader) * (num_body_epochs or num_epochs)
                body_lr = last_shot_body_learning_rate or body_learning_rate
                # print("** Training body **")
                # print(f"Num examples = {len(train_examples)}")
                # print(f"Num body epochs = {num_body_epochs}")
                # print(f"Total optimization steps = {total_train_steps}")
                # print(f"Total train batch size = {setfit_batch_size}")

                warmup_steps = math.ceil(total_train_steps * self.warmup_proportion)
                self.model.model_body.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=num_body_epochs,
                    optimizer_params={"lr": body_lr},
                    warmup_steps=warmup_steps,
                    show_progress_bar=show_progress_bar,
                    use_amp=self.use_amp,
                )

            if not self.model.has_differentiable_head or not self._freeze or not self.freeze_head:
                # Train the final classifier
                num_head_epochs = last_shot_head_epochs or num_epochs
                # print("** Training head **")
                # print(f"Num head epochs = {num_head_epochs}")

                self.model.fit(
                    x_train,
                    y_train,
                    num_epochs=num_head_epochs,
                    batch_size=setfit_batch_size,
                    learning_rate=learning_rate,
                    l2_weight=l2_weight,
                    max_length=max_length,
                    show_progress_bar=show_progress_bar,
                )

        training_history = []

        # Check if there is labels
        labels = train_dataset["label"] if "label" in train_dataset.features else None
        eval_labels = eval_dataset["label"] if "label" in eval_dataset.features else None

        # Run First Shot
        print(f"Running First-Shot on {len(train_dataset['text'])} documents.")
        t0 = time.time()

        if self.model.first_shot_model:
            raw_probs, embeds, original_logits = self.model.first_shot_model(train_dataset["text"], return_embeddings=True, return_logits=True)
            max_probs = ([max(probs) for probs in raw_probs])
            print("mean:",float(torch.mean(torch.stack(max_probs))),"-- std:",float(torch.std((torch.stack(max_probs)))))
            print(f"1st shot - cosine product time: {round(time.time()-t0,2)} seconds")
            # self.data_selector(None, None, embeds,selection_strategy='first_shot')            
            # # TO DO: if demanded and train_dataset["label"], report metrics on the performance of the model
            if return_history and labels:
                y_pred = torch.argmax(raw_probs, axis=-1)
                _, label_embeds = self.model.first_shot_model(self.model.first_shot_model.classes_list, return_embeddings=True)
                current_metric = {"full_train_raw_first_shot":self._predict_metrics(y_pred, labels), "unsup_full_train_raw_first_shot":self.unsup_evaluator(embeds, raw_probs, label_embeds, original_logits)}
                print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                # print(list(current_metric.keys())[1], "----- ",current_metric[list(current_metric.keys())[1]])
                training_history.append(current_metric)
                if eval_dataset and eval_labels:
                    test_probs, test_embeds, test_original_logits = self.model.first_shot_model(eval_dataset["text"], return_embeddings=True, return_logits=True)
                    max_probs = ([max(probs) for probs in test_probs])
                    print("mean:",float(torch.mean(torch.stack(max_probs))),"-- std:",float(torch.std((torch.stack(max_probs)))))

                    # print("raw:",test_probs)
                    y_pred = torch.argmax(test_probs, axis=-1)
                    # print(list(zip(y_pred[0:20],eval_dataset['label'][0:20])))

                    current_metric = {"eval_raw_first_shot": self._predict_metrics(y_pred, eval_dataset["label"]), "unsup_eval_raw_first_shot}":self.unsup_evaluator(test_embeds, test_probs, label_embeds, test_original_logits)}
                    print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                    # print(list(current_metric.keys())[1], "----- ", current_metric[list(current_metric.keys())[1]])
                    training_history.append(current_metric)
                    saving_tuple = (embeds, raw_probs, labels, test_embeds, test_probs, eval_dataset["label"])
                    # with open("dim_" + self.experiment_name + "_" + "full_train_raw_first_shot:" + '.pickle','wb') as handle:
                    #     pickle.dump(saving_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Throws error
            raise RuntimeError("ZeroBERTo training requires a first shot model")
        
        if self.model.first_shot_model and self.train_first_shot:
            x_train, y_train = self._build_first_shot_dataset()
            print(f'1st shot Dataset: {x_train}')
            train_setfit_iteration(last_shot_body_epochs=5)
            trained_probs, fs_trained_embeds = self.model.predict_proba(train_dataset["text"], return_embeddings=True)
            print(f"1st shot - train and prediction time: {round(time.time()-t0,2)} seconds")
            max_probs = ([max(probs) for probs in trained_probs])
            
            print("mean:",float(torch.mean(torch.stack(max_probs))),"-- std:",float(torch.std((torch.stack(max_probs)))))
            # TO DO: if demanded and train_dataset["label"], report metrics on the performance of the model on train set
            if return_history and labels:
                y_pred = torch.argmax(trained_probs, axis=-1)
                current_metric = {f"full_train_trained_first_shot":self._predict_metrics(y_pred, labels)}
                print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                training_history.append(current_metric)
                if eval_dataset and eval_labels:
                    test_probs,test_embeds  = self.model.predict_proba(eval_dataset["text"], return_embeddings=True)
                    # print(test_probs)
                    y_pred = torch.argmax(test_probs, axis=-1)
                    # print(list(zip(y_pred[0:20],eval_dataset['label'][0:20])))
                    current_metric = {f"eval_trained_first_shot": self._predict_metrics(y_pred, eval_dataset["label"])}
                    print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                    training_history.append(current_metric)
                    saving_tuple = (fs_trained_embeds, trained_probs, labels, test_embeds, test_probs, eval_dataset["label"])
                    # with open("dim_" + self.experiment_name + "_" + "full_train_trained_first_shot" + '.pickle', 'wb') as handle:
                    #     pickle.dump(saving_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # probs, embeds = self.model.first_shot_model(train_dataset["text"], return_embeddings=True)




        samples_per_label_roadmap = self.var_samples_per_label if self.var_samples_per_label is not None else list(np.repeat(self.samples_per_label,num_setfit_iterations))
        selection_strategy_roadmap = self.var_selection_strategy if self.var_selection_strategy else num_setfit_iterations*[None]

        training_indices = []
        last_shot_training_data = []
        print(f"Data Selector roadmap: {samples_per_label_roadmap}")
        # Iterations of setfit
        t0_setfit = time.time()
        probs = trained_probs if self.train_first_shot else raw_probs
        for i in range(num_setfit_iterations):
            print(f"********** Running SetFit Iteration {i+1} **********")
            
            ti_setfit = time.time()
            if i!=0:
                last_select_strat = selection_strategy_roadmap[i-1]
                if last_select_strat == 'top_n' and i+1 < num_setfit_iterations:
                    self.model.reset_model_body()
                
            x_train, y_train, labels_train, training_indices, probs_train = self.data_selector(train_dataset["text"], probs, embeds,
                                                                                  labels=labels,
                                                                                  n=samples_per_label_roadmap[i],
                                                                                  selection_strategy=selection_strategy_roadmap[i],
                                                                                  discard_indices=[] if allow_resampling else training_indices)
            # print(type(y_train),y_train)

            # if self.train_first_shot:
            #     x_train_fs, y_train_fs = self._build_first_shot_dataset()
            #     x_train = x_train + x_train_fs
            #     y_train = y_train + list(y_train_fs)
            #     labels_train = labels_train + list(y_train_fs)

            print("Data Selected:", len(x_train))
            # last_shot_training_data.append(list(zip(x_train, y_train, labels_train, training_indices, probs_train)))
             # if demanded and train_dataset["label"], report metrics on the performance of the selection
            if return_history and labels:
                current_metric = {f"data_selector-{i+1}":self._predict_metrics(y_train, labels_train)}
                print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                # Save data_selector_moment
                data_selector_tuple = (probs, embeds, labels, selection_strategy_roadmap[i], [] if allow_resampling else training_indices, original_logits)
                with open(self.experiment_name + "_" + f"data_selector-{i+1}" + '.pickle', 'wb') as handle:
                    pickle.dump(data_selector_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)

                training_history.append(current_metric)
            train_setfit_iteration()
            probs, new_embeds = self.model.predict_proba(train_dataset["text"], return_embeddings=True)

            max_probs = ([max(probs) for probs in probs])
            
            print("mean:",float(torch.mean(torch.stack(max_probs))),"-- std:",float(torch.std((torch.stack(max_probs)))))
            if update_embeddings:
                embeds = new_embeds
            # TO DO: if demanded and train_dataset["label"], report metrics on the performance of the model on train set
            if return_history and labels:
                y_pred = torch.argmax(probs, axis=-1)
                _, label_embeds = self.model.predict_proba(self.model.first_shot_model.classes_list, return_embeddings=True)
                current_metric = {f"full_train_setfit_iteration-{i+1}":self._predict_metrics(y_pred, labels), f"unsup_full_train_setfit_iteration-{i+1}":self.unsup_evaluator(new_embeds, probs, label_embeds, original_logits)}
                print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                training_history.append(current_metric)

                current_probs = self.model.predict_proba(x_train, return_embeddings=False)
                current_pred = torch.argmax(current_probs, axis=-1)
                current_metric = {f"cur_train_setfit_iteration-{i + 1}": self._predict_metrics(current_pred, labels_train)}
                print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                training_history.append(current_metric)
                if eval_dataset and eval_labels:
                    test_probs, test_embeds = self.model.predict_proba(eval_dataset["text"], return_embeddings=True)
                    max_probs = ([max(probs) for probs in test_probs])
                    
                    print("mean:",float(torch.mean(torch.stack(max_probs))),"-- std:",float(torch.std((torch.stack(max_probs)))))
                    y_pred = torch.argmax(test_probs, axis=-1)
                    current_metric = {f"eval_setfit_iteration-{i+1}": self._predict_metrics(y_pred, eval_dataset["label"]), f"unsup_eval_setfit_iteration-{i+1}":self.unsup_evaluator(test_embeds, test_probs, label_embeds, test_original_logits)}
                    print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                    training_history.append(current_metric)
                    saving_tuple = (new_embeds, probs, labels, test_embeds, test_probs, eval_dataset["label"])
                    with open("dim_" + self.experiment_name + "_" + f"full_train_setfit_iteration-{i+1}" + '.pickle','wb') as handle:
                        pickle.dump(saving_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # TO DO: if test_dataset, report metrics on the performance of the model on test set
            if reset_model_head and i+1 < num_setfit_iterations:
                self.model.reset_model_head()
            print(f"Iteration {i+1} time: {round(time.time()-ti_setfit,2)}")

            if not self.data_selector.keep_training:
                print("Training stopped because no clusters were found on last iteration.")
                break



        # print(f"********** Running Last Shot **********") ##########################

        # self.model.reset_model_body(self.model.first_shot_model.embedding_model)

        # t0_lastshot = time.time()

        # x_train, y_train, labels_train, training_indices, probs_train = self.data_selector(train_dataset["text"], probs, embeds,
        #                                                                 labels=labels,
        #                                                                 n=32,
        #                                                                 selection_strategy='top_n',
        #                                                                 discard_indices=[] if allow_resampling else training_indices)
        # print("Data Selected:",len(x_train))

        # # last_shot_training_data.append(list(zip(x_train, y_train, labels_train, training_indices, probs_train)))
        #     # if demanded and train_dataset["label"], report metrics on the performance of the selection
        # if return_history and labels:
        #     current_metric = {f"last_shot_data_selector":self._predict_metrics(y_train, labels_train)}
        #     print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])

        #     training_history.append(current_metric)
        # train_setfit_iteration(last_shot_body_epochs=2,last_shot_head_epochs=2,last_shot_body_learning_rate=1e-4)
        # probs, new_embeds = self.model.predict_proba(train_dataset["text"], return_embeddings=True)
        # if update_embeddings:
        #     embeds = new_embeds
        # print(f"Last Shot time: {round(time.time()-t0_lastshot,2)}")
        # if return_history and labels:
        #     y_pred = torch.argmax(probs, axis=-1)
        #     current_metric = {f"full_train_last_shot":self._predict_metrics(y_pred, labels)}
        #     print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
        #     training_history.append(current_metric)

        #     current_probs = self.model.predict_proba(x_train, return_embeddings=False)
        #     current_pred = torch.argmax(current_probs, axis=-1)
        #     current_metric = {f"cur_train_last_shot": self._predict_metrics(current_pred, labels_train)}
        #     print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
        #     training_history.append(current_metric)
        #     if eval_dataset and eval_labels:
        #         test_probs = self.model.predict_proba(eval_dataset["text"], return_embeddings=False)
        #         y_pred = torch.argmax(test_probs, axis=-1)
        #         current_metric = {f"eval_last_shot": self._predict_metrics(y_pred, eval_dataset["label"])}
        #         print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
        #         training_history.append(current_metric)

        return training_history if return_history else None

    def _predict_metrics(self, predictions, references):
        """
        Computes the metrics for the self.model classifier.
        Returns:
            `Dict[str, float]`: The evaluation metrics.
        """

        if isinstance(self.metric, str):
            metric_config = "multilabel" if self.model.multi_target_strategy is not None else None
            metric_fn = evaluate.load(self.metric, config_name=metric_config)
            metric_kwargs = self.metric_kwargs or {}

            return metric_fn.compute(predictions=predictions, references=references, **metric_kwargs)

        elif callable(self.metric):
            return self.metric(predictions, references)

        else:
            raise ValueError("metric must be a string or a callable")

    def _build_first_shot_dataset(self):
        x_train = [self.model.first_shot_model.hypothesis_template.format(cl) for cl in self.model.first_shot_model.classes_list]
        y_train = np.arange(len(self.model.first_shot_model.classes_list))
        return x_train, y_train

