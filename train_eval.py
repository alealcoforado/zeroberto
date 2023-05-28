import argparse

from sentence_transformers.losses import CosineSimilarityLoss
from datasets import load_dataset
from functools import partial
import evaluate

from ZeroBERTo.modeling_zeroberto import ZeroBERToModel, ZeroBERToDataSelector
from ZeroBERTo.trainer import ZeroBERToTrainer

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
        "--dataset", type=str, help="Dataset name", default="sst2"
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
         

    args = parser.parse_args()
    return args

def main():
    args = arg_parse()

    # TODO: Try config file *.yml

    # Open the dataset
    dataset = load_dataset(args.dataset)
    train_dataset = dataset[args.dataset_train_split].shuffle(seed=42).select(range(0,min(len(dataset[args.dataset_train_split]), 5000)))
    # args.dataset_test_split = "test" # TO DO remove
    test_dataset = dataset[args.dataset_test_split]#.select(range(0,200))


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

    # Build trainer
    trainer = ZeroBERToTrainer(
        model=model,
        data_selector=data_selector,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric=compute_metrics_fn,
        loss_class=CosineSimilarityLoss,
        num_iterations=args.num_iterations,
        num_setfit_iterations=args.num_setfit_iterations,
        num_epochs=args.num_epochs,
        seed=42,
        column_mapping=dataset_column_mapping,
        samples_per_label=args.samples_per_label,
        batch_size=args.batch_size,
        var_samples_per_label=args.var_samples_per_label,
        var_selection_strategy=args.var_selection_strategy,
        learning_rate=args.learning_rate,
        body_learning_rate=args.body_learning_rate,
        freeze_head=args.freeze_head,
        freeze_body=args.freeze_body,
        train_first_shot = args.train_first_shot,
        allow_resampling=args.allow_resampling

    )

    # Body training

    train_history = trainer.train(return_history=True)
    print(train_history)

    # Evaluate

    final_metrics = trainer.evaluate()
    print(final_metrics)


if __name__ == '__main__':
    main()