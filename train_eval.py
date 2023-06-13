import argparse

from sentence_transformers.losses import CosineSimilarityLoss
from datasets import load_dataset
from functools import partial
import evaluate

from ZeroBERTo.modeling_zeroberto import ZeroBERToModel, ZeroBERToDataSelector
from ZeroBERTo.trainer import ZeroBERToTrainer

import json
from datetime import datetime

def compute_metrics(y_pred, y_test, metrics):
    results = {}
    try:
        for average in ['weighted', 'macro']:
            results[average] = {}
            for metric_name in metrics.keys():
                if metric_name == "accuracy":
                    results[average].update(metrics[metric_name].compute(predictions=y_pred, references=y_test))
                else:
                    results[average].update(metrics[metric_name].compute(predictions=y_pred, references=y_test, average=average))
    except:
        print("Error")
    return results

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="Model name", default="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset name", default="SetFit/sst2"
    )
    parser.add_argument(
        "--dataset_train_split", type=str, help="Training data split", default="train"
    )
    parser.add_argument(
        "--dataset_test_split", type=str, help="Test data split", default="test"
    )
    parser.add_argument(
        "--hypothesis_template", type=str, help="Hypothesis Template for First Shot classification", default="{}"
    )
    parser.add_argument(
        "--multi_target_strategy", type=str, help="Multi Target Strategy", default=None
    )
    parser.add_argument(
        "--use_differentiable_head", action=argparse.BooleanOptionalAction, help="Use Differentiable head", default=False
    )
    parser.add_argument(
        "--num_iterations", type=int, help="Number of pairs to generate on training.", default=20
    )
    parser.add_argument(
        "--num_setfit_iterations", type=int, help="Number of SetFit training iterations to perform", default=2
    )   
    parser.add_argument(
        "--num_epochs", type=int, help="Number of self-training loop iterations to perform", default=1
    )  
    parser.add_argument(
        "--samples_per_label", type=int, help="Number of samples per class to pick for training", default=4
    )   
    parser.add_argument(
        "--normalize_embeddings", type=bool, help="Normalize Embeddings", default=False
    )
    parser.add_argument(
        "--selection_strategy", type=str, help="How Data Selector should pick training data", default='top_n'
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for GPU", default=8
    )
    parser.add_argument(
        "--var_samples_per_label", type=int, nargs="*", default=None
    )
    parser.add_argument(
        "--var_selection_strategy", type=str, nargs="*", default=None
    )
    parser.add_argument(
        "--learning_rate", type=float,  default=2e-5
    )
    parser.add_argument(
        "--body_learning_rate", type=float, default=2e-5
    )
    parser.add_argument(
        "--num_body_epochs", type=int, default=1
    )
    parser.add_argument(
        "--freeze_head",help="If True, will train head.", default=False,action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--freeze_body",help="If True, will not train body.", default=False,action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--train_first_shot",help="If True, will train once before First Shot.", default=False,action=argparse.BooleanOptionalAction
    )    
    parser.add_argument(
        "--allow_resampling",help="If True, will not discard training data on subsequent iterations.", default=False,action=argparse.BooleanOptionalAction
    )              
    parser.add_argument(
        "--random_seed",help="Integer as random seed for dataset sampling.", default=42,type=int
    )
    parser.add_argument(
        "--train_dataset_size",help="number of unlabeled samples to take as train dataset", default=5000,type=int
    ) 
    parser.add_argument(
        "--auto",help="If True, will automatically set hyperparameters.", default=False,action=argparse.BooleanOptionalAction
    ) 

    args = parser.parse_args()
    return args

def main():
    args = arg_parse()

    # TODO: Try config file *.yml

    # Open the dataset
    dataset = load_dataset(args.dataset)
    train_dataset_size = min(len(dataset[args.dataset_train_split]), args.train_dataset_size)
    # print(f"Train dataset size: {train_dataset_size}")
    random_seed = args.random_seed
    train_dataset = dataset[args.dataset_train_split].shuffle(seed=random_seed).select(range(0,train_dataset_size))
    # args.dataset_test_split = "test" # TO DO remove
    test_dataset = dataset[args.dataset_test_split]#.select(range(0,200))

    # Define experiment name
    dataset_name = args.dataset.split("/")[-1]
    current_dateTime = str(datetime.now())
    experiment_name = dataset_name+"_"+ current_dateTime
    experiment_hyperparameters_name = dataset_name+"_hyperparameters_"+current_dateTime


    if args.dataset=='SetFit/sst2':
        classes_list = ["negative", "positive"] # TO DO
        dataset_column_mapping = {"text": "text", "label": "label"}

    elif args.dataset=='ag_news':
        classes_list = ["world","sports","business","science and technology"]
        dataset_column_mapping = {"text": "text", "label": "label"}

    elif args.dataset=='SetFit/ag_news':
        classes_list = ["world","sports","business","science and technology"]
        dataset_column_mapping = {"text": "text", "label": "label"}

    elif args.dataset=='SetFit/sst5':
        classes_list = ["very negative","negative","neutral","positive","very positive"]
        dataset_column_mapping = {"text": "text", "label": "label"}

    elif args.dataset=='SetFit/emotion':
        classes_list = ['sadness','joy','love','anger','fear','surprise']
        dataset_column_mapping = {"text": "text", "label": "label"}

    elif args.dataset=='SetFit/enron_spam':
        classes_list = ['ham','spam']
        dataset_column_mapping = {"text": "text", "label": "label"}

    elif args.dataset=='SetFit/20_newsgroups':
        classes_list = ['atheism', 'computer graphics', 'microsoft windows', 'pc hardware', 'mac hardware','windows x', 'for sale', 'cars'
                        ,'motorcycles','baseball','hockey', 'cryptography', 'electronics','medicine', 'space', 'christianity',
                         'guns', 'middle east', 'politics', 'religion'] 
        dataset_column_mapping = {"text": "text", "label": "label"}
    elif args.dataset=='SetFit/CR':
        classes_list = ['negative','positive']
        dataset_column_mapping = {"text": "text", "label": "label"}
    elif args.dataset=='dbpedia_14':
        classes_list = [ "Company", "Educational Institution", "Artist", "Athlete", "Office Holder", "Mean Of Transportation", 
                        "Building", "Natural Place", "Village", "Animal", "Plant", "Album", "Film", "Written Work" ]
        dataset_column_mapping = {"content": "text", "label": "label"}

    elif args.dataset=='yahoo_answers_topics':
        classes_list = [ "society & culture", "science & mathematics", 'health', 'education & reference', 'computers & internet',
                        'sports', 'business & finance', 'entertainment & music', 'family & relationships', 'politics & government']
        dataset_column_mapping = {"question_title": "text", "topic": "label"}

    elif args.dataset=='imdb':
        classes_list = ['negative','positive']
        dataset_column_mapping = {"text": "text", "label": "label"}
    
    elif args.dataset=='SetFit/yelp_review_full':
        classes_list = ['1 star','2 stars','3 stars','4 stars', '5 stars']
        dataset_column_mapping = {"text": "text", "label": "label"}
    
    # print(args.body_learning_rate,args.learning_rate)
    # Load the model
    if args.use_differentiable_head:
        model = ZeroBERToModel.from_pretrained(args.model_name_or_path,
                                            hypothesis_template=args.hypothesis_template,
                                            classes_list=classes_list,
                                            multi_target_strategy=args.multi_target_strategy,
                                            use_differentiable_head=args.use_differentiable_head,
                                            normalize_embeddings=args.normalize_embeddings,
                                            head_params={"out_features": len(classes_list)}
                                            )
    else:
        model = ZeroBERToModel.from_pretrained(args.model_name_or_path,
                                            hypothesis_template=args.hypothesis_template,
                                            classes_list=classes_list,
                                            multi_target_strategy=args.multi_target_strategy,
                                            use_differentiable_head=args.use_differentiable_head,
                                            normalize_embeddings=args.normalize_embeddings,
                                            )

    # Compute metrics function
    metrics = {}
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        metrics[metric_name] = evaluate.load(metric_name)
    compute_metrics_fn = partial(compute_metrics, metrics=metrics)

    # Set up Data Selector
    data_selector = ZeroBERToDataSelector(selection_strategy=args.selection_strategy)

    print("Start training")

    if args.auto:
        if len(classes_list) == 2:
            var_samples_per_label = [8, 16]
            var_selection_strategy = ['top_n', 'intraclass_clustering']
            num_setfit_iterations = 2
        else:
            var_samples_per_label = [16, 32, 64, 128, 256]
            var_selection_strategy = ['top_n', 'intraclass_clustering','top_n', 'intraclass_clustering','top_n']
            num_setfit_iterations = 2
        num_iterations = 10
        num_epochs = 1
        train_first_shot = True
    else:
        var_samples_per_label = None
        var_selection_strategy = None
        num_setfit_iterations = None
        num_iterations = None
        num_epochs = None
        train_first_shot = None




    # Build trainer
    trainer = ZeroBERToTrainer(
        model=model,
        data_selector=data_selector,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric=compute_metrics_fn,
        loss_class=CosineSimilarityLoss,
        num_iterations= num_iterations or args.num_iterations,
        num_setfit_iterations=num_setfit_iterations or args.num_setfit_iterations,
        num_epochs=num_epochs or args.num_epochs,
        seed=42,
        column_mapping=dataset_column_mapping,
        samples_per_label=args.samples_per_label,
        batch_size=args.batch_size,
        var_samples_per_label=var_samples_per_label or args.var_samples_per_label,
        var_selection_strategy=var_selection_strategy or args.var_selection_strategy,
        learning_rate=args.learning_rate,
        body_learning_rate=args.body_learning_rate,
        freeze_head=args.freeze_head,
        freeze_body=args.freeze_body,
        train_first_shot = train_first_shot or args.train_first_shot,
        allow_resampling=args.allow_resampling,
        experiment_name=experiment_name

    )

    hyperparameters = {}
    hyperparameters['model_name_or_path'] = args.model_name_or_path
    hyperparameters['dataset'] = args.dataset
    hyperparameters['train_dataset_size'] = train_dataset_size
    hyperparameters['random_seed'] = random_seed
    hyperparameters['dataset_train_split'] = args.dataset_train_split
    hyperparameters['dataset_test_split'] = args.dataset_test_split
    hyperparameters['hypothesis_template'] = args.hypothesis_template
    hyperparameters['multi_target_strategy'] = args.multi_target_strategy
    hyperparameters['use_differentiable_head'] = args.use_differentiable_head
    hyperparameters['num_iterations'] = num_iterations or args.num_iterations
    hyperparameters['num_setfit_iterations'] = num_setfit_iterations or args.num_setfit_iterations
    hyperparameters['num_epochs'] = num_epochs or args.num_epochs
    hyperparameters['samples_per_label'] = args.samples_per_label
    hyperparameters['normalize_embeddings'] = args.normalize_embeddings
    hyperparameters['selection_strategy'] = args.selection_strategy
    hyperparameters['batch_size'] = args.batch_size
    hyperparameters['var_samples_per_label'] = var_samples_per_label or args.var_samples_per_label
    hyperparameters['var_selection_strategy'] = var_selection_strategy or args.var_selection_strategy
    hyperparameters['learning_rate'] = args.learning_rate
    hyperparameters['body_learning_rate'] = args.body_learning_rate
    hyperparameters['num_body_epochs'] = args.num_body_epochs
    hyperparameters['freeze_head'] = args.freeze_head
    hyperparameters['freeze_body'] = args.freeze_body
    hyperparameters['train_first_shot'] = train_first_shot or args.train_first_shot
    hyperparameters['allow_resampling'] = args.allow_resampling
    print(hyperparameters)
    # Body training

    with open(experiment_hyperparameters_name+".json", "w") as final:
        json.dump(hyperparameters, final)

    train_history = trainer.train(return_history=True)
    print(train_history)

    with open(experiment_name+".json", "w") as final:
        json.dump(train_history, final)


    # Evaluate
    final_metrics = trainer.evaluate()
    print(final_metrics)


if __name__ == '__main__':
    main()