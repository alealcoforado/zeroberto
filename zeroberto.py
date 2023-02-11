import numpy as np
import pandas as pd
import time

def getPredictions(setfit_trainer):
  setfit_trainer._validate_column_mapping(setfit_trainer.eval_dataset)
  eval_dataset = setfit_trainer.eval_dataset

  if setfit_trainer.column_mapping is not None:
      eval_dataset = setfit_trainer._apply_column_mapping(setfit_trainer.eval_dataset, setfit_trainer.column_mapping)

  x_test = eval_dataset["text"]
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

        if config['zeroshot_method'] == "probability_threshold":
            top_prob = pred['scores'][0]
            top_label = pred['labels'][0]

            if (top_prob>=config['prob_goal']):
                goal_count[top_label] += 1
            if len(preds) % 50 == 0:
                print("Preds:",len(preds)," - Total time:",round(time.time()-t0,2),"seconds")
                print(goal_count)
            if all(count >= config['top_n_goal'] for count in list(goal_count.values())):
                break  ### stop loop if goal is met

        if config['zeroshot_method'] == "top_n_goal":
            top_prob = pred['scores'][0]
            top_label = pred['labels'][0]
            if len(preds) % 50 == 0:
                print("Preds:",len(preds)," - Total time:",round(time.time()-t0,2),"seconds")
                print(goal_count)


    print(len(preds))
    return preds

def formatZeroShotResults(results):
  df_results = pd.DataFrame(pd.Series(results).to_dict())
  label_probabilities = list(map(mapper,enumerate(results)))
  label_probabilities_df = pd.DataFrame(label_probabilities,index=df_results.columns)
  
  return label_probabilities_df