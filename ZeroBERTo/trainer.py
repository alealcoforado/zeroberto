import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import numpy as np
from sentence_transformers import InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from torch.utils.data import DataLoader
import torch
import evaluate
from tqdm.auto import trange
from transformers.trainer_utils import set_seed

from setfit import SetFitTrainer
from setfit import logging
from setfit.modeling import SupConLoss, sentence_pairs_generation, sentence_pairs_generation_multilabel

if TYPE_CHECKING:
    import optuna
    from datasets import Dataset

    from modeling_zeroberto import ZeroBERToModel, ZeroBERToDataSelector

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
            learning_rate: float = 2e-5,
            batch_size: int = 16,
            seed: int = 42,
            column_mapping: Optional[Dict[str, str]] = None,
            use_amp: bool = False,
            warmup_proportion: float = 0.1,
            distance_metric: Callable = BatchHardTripletLossDistanceFunction.cosine_distance,
            margin: float = 0.25,
            samples_per_label: int = 2,
            var_samples_per_label: list = None,
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
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.column_mapping = column_mapping
        self.use_amp = use_amp
        self.warmup_proportion = warmup_proportion
        self.distance_metric = distance_metric
        self.margin = margin
        self.samples_per_label = samples_per_label
        self.var_samples_per_label = var_samples_per_label
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
        self._freeze = True  # If True, will train the body only; otherwise, train the body and head

    def train(
            self,
            num_epochs: Optional[int] = None,
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
            logger.info("Applying column mapping to training dataset")
            train_dataset = self._apply_column_mapping(self.train_dataset, self.column_mapping)
            if eval_dataset:
                eval_dataset = self._apply_column_mapping(self.eval_dataset, self.column_mapping)

        #x_train = train_dataset["text"]
        #y_train = train_dataset["label"]
        if self.loss_class is None:
            logger.warning("No `loss_class` detected! Using `CosineSimilarityLoss` as the default.")
            self.loss_class = losses.CosineSimilarityLoss

        num_epochs = num_epochs or self.num_epochs
        num_setfit_iterations = num_setfit_iterations or self.num_setfit_iterations
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate

        def train_setfit_iteration():
            setfit_batch_size = batch_size
            if not self.model.has_differentiable_head or self._freeze:
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

                total_train_steps = len(train_dataloader) * num_epochs
                logger.info("***** Running training *****")
                logger.info(f"  Num examples = {len(train_examples)}")
                logger.info(f"  Num epochs = {num_epochs}")
                logger.info(f"  Total optimization steps = {total_train_steps}")
                logger.info(f"  Total train batch size = {setfit_batch_size}")

                warmup_steps = math.ceil(total_train_steps * self.warmup_proportion)
                self.model.model_body.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=num_epochs,
                    optimizer_params={"lr": learning_rate},
                    warmup_steps=warmup_steps,
                    show_progress_bar=show_progress_bar,
                    use_amp=self.use_amp,
                )

            if not self.model.has_differentiable_head or not self._freeze:
                # Train the final classifier
                print("Training head")
                self.model.fit(
                    x_train,
                    y_train,
                    num_epochs=num_epochs,
                    batch_size=setfit_batch_size,
                    learning_rate=learning_rate,
                    body_learning_rate=body_learning_rate,
                    l2_weight=l2_weight,
                    max_length=max_length,
                    show_progress_bar=True,
                )

        training_history = []

        # Check if there is labels
        labels = train_dataset["label"] if "label" in train_dataset.features else None
        eval_labels = eval_dataset["label"] if "label" in eval_dataset.features else None

        # Run First Shot
        print(f"Running First-Shot on {len(train_dataset['text'])} documents.")
        if self.model.first_shot_model:
            probs, embeds = self.model.first_shot_model(train_dataset["text"], return_embeddings=True)
            # TO DO: if demanded and train_dataset["label"], report metrics on the performance of the model
            if return_history and labels:
                y_pred = torch.argmax(probs, axis=-1)
                current_metric = {"first_shot":self._predict_metrics(y_pred, labels)}
                print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                training_history.append(current_metric)

        else:
            # Throws error
            raise RuntimeError("ZeroBERTo training requires a first shot model")
        # print(var_samples_per_label)
        # var_samples_per_label = [el for sublist in var_samples_per_label for el in sublist] if var_samples_per_label is not None else None
        # print(var_samples_per_label)
        samples_per_label_roadmap = self.var_samples_per_label if self.var_samples_per_label is not None else list(np.repeat(self.samples_per_label,num_setfit_iterations))
        
        print(f"Data Selector roadmap: {samples_per_label_roadmap}")
        # Iterations of setfit
        for i in range(num_setfit_iterations):
            print(f"********** Running SetFit Iteration {i+1} **********")
            x_train, y_train, labels_train = self.data_selector(train_dataset["text"], probs, embeds, labels=labels, n=samples_per_label_roadmap[i])
            print("Data Selected:",len(x_train))

            # print(list(zip(x_train,y_train,labels_train)))
            # print(type(x_train),type(y_train),type(labels_train))
            # print(len(x_train),len(y_train),len(labels_train))
            # if demanded and train_dataset["label"], report metrics on the performance of the selection
            if return_history and labels:
                current_metric = {f"data_selector-{i+1}":self._predict_metrics(y_train, labels_train)}
                print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])

                training_history.append(current_metric)
            train_setfit_iteration()
            probs, embeds = self.model.predict_proba(train_dataset["text"], return_embeddings=True)
            # TO DO: if demanded and train_dataset["label"], report metrics on the performance of the model on train set
            if return_history and labels:
                y_pred = torch.argmax(probs, axis=-1)
                current_metric = {f"full_train_setfit_iteration-{i+1}":self._predict_metrics(y_pred, labels)}
                print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                training_history.append(current_metric)

                current_probs = self.model.predict_proba(x_train, return_embeddings=False)
                current_pred = torch.argmax(current_probs, axis=-1)
                current_metric = {f"cur_train_setfit_iteration-{i + 1}": self._predict_metrics(current_pred, labels_train)}
                print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                training_history.append(current_metric)
                if eval_dataset and eval_labels:
                    test_probs = self.model.predict_proba(eval_dataset["text"], return_embeddings=False)
                    y_pred = torch.argmax(test_probs, axis=-1)
                    current_metric = {f"eval_setfit_iteration-{i+1}": self._predict_metrics(y_pred, eval_dataset["label"])}
                    print(list(current_metric.keys())[0], "----- accuracy:",current_metric[list(current_metric.keys())[0]]['weighted']['accuracy'])
                    training_history.append(current_metric)
            # TO DO: if test_dataset, report metrics on the performance of the model on test set
            if reset_model_head and i+1 < num_setfit_iterations:
                self.model.reset_model_head()

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



