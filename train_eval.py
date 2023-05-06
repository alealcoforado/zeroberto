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
        "--use_differentiable_head", type=bool, help="Use Differentiable head", default=False
    )
    parser.add_argument(
        "--normalize_embeddings", type=bool, help="Normalize Embeddings", default=False
    )
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()

    # TODO: Try config file *.yml

    # Open the dataset
    dataset = load_dataset(args.dataset)
    train_dataset = dataset[args.dataset_train_split].select(range(0,100))
    args.dataset_test_split = "validation" # TO DO remove
    test_dataset = dataset[args.dataset_test_split]

    classes_list = ["negative", "positive"] # TO DO

    # Load the model
    model = ZeroBERToModel.from_pretrained(args.model_name_or_path,
                                           hypothesis_template=args.hypothesis_template,
                                           classes_list=classes_list,
                                           multi_target_strategy=args.multi_target_strategy,
                                           use_differentiable_head=args.use_differentiable_head,
                                           normalize_embeddings=args.normalize_embeddings
                                           )

    # Compute metrics function
    metrics = {}
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        metrics[metric_name] = evaluate.load(metric_name)
    compute_metrics_fn = partial(compute_metrics, metrics=metrics)

    # Set up Data Selector
    data_selector = ZeroBERToDataSelector(selection_strategy="top_n")

    print("Start training")

    # Build trainer
    trainer = ZeroBERToTrainer(
        model=model,
        data_selector=data_selector,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric=compute_metrics_fn,
        loss_class=CosineSimilarityLoss,
        num_iterations=20,
        num_setfit_iterations=1,
        num_epochs=2,
        seed=42,
        column_mapping={"sentence": "text", "label": "label"},
        samples_per_label=4,
    )

    # Body training

    train_history = trainer.train(return_history=True)
    print(train_history)

    # Evaluate

    final_metrics = trainer.evaluate()
    print(final_metrics)




if __name__ == '__main__':
    main()