import json
import glob
import os
import argparse
import pandas as pd

import numpy as np
import scipy.stats


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", type=str, help="Input path", default="./"
    )
    parser.add_argument(
        "--files_start", type=str, nargs="*", default=["sst2_"]
    )
    parser.add_argument(
        "--supervised_metrics", type=int, nargs="*", default=["accuracy"]
    )
    args = parser.parse_args()
    return args


def get_parsed_data(raw_data, sup_metric_name="accuracy"):
    unsup_vec = []
    sup_vec = []
    for dictx in raw_data:
        for key in dictx.keys():
            if "unsup_" in key:
                unsup_vec.append(dictx[key])
                new_key = key.replace("}", "").replace("unsup_", "")
                sup_vec.append(dictx[new_key]["weighted"][sup_metric_name])

    return sup_vec, unsup_vec


def main():
    args = arg_parse()

    files_list = []
    for file_start in args.files_start:
        files_list = files_list + glob.glob(os.path.join(args.input_dir, file_start + "*.json"))

    vec_sup_metrics, vec_unsup_metrics = [], []

    for file in files_list:
        # Open train history
        with open(file) as f:
            for line in f:
                raw_data = json.loads(line)
        # Get parsed data
        supervised_metrics, unsupervised_metrics = get_parsed_data(raw_data)
        vec_sup_metrics += supervised_metrics
        vec_unsup_metrics += unsupervised_metrics

    # Mount unsup metrics
    df_unsup_metrics = pd.DataFrame(vec_unsup_metrics)
    for unsup_metric in df_unsup_metrics.columns:
        print(unsup_metric)
        print(scipy.stats.pearsonr(vec_sup_metrics, df_unsup_metrics[unsup_metric]))
        print(scipy.stats.spearmanr(vec_sup_metrics, df_unsup_metrics[unsup_metric]))
        print(scipy.stats.kendalltau(vec_sup_metrics, df_unsup_metrics[unsup_metric]))

    # evaluation of a model using all input features
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_unsup_metrics, vec_sup_metrics, test_size=0.1, random_state=42)
    # fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    yhat = model.predict(X_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, yhat)
    r2 = r2_score(y_test, yhat)
    print('Pure regression - MAE: %.3f' % mae)
    print('Pure regression - R2: %.3f' % r2)


    # Using correlation scores

    # evaluation of a model using 10 features chosen with correlation
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression

    # feature selection
    def select_features(X_train, y_train, X_test):
        # configure to select a subset of features
        fs = SelectKBest(score_func=f_regression, k=5)
        # learn relationship from training data
        fs.fit(X_train, y_train)
        # transform train input data
        X_train_fs = fs.transform(X_train)
        # transform test input data
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    # fit the model
    model = LinearRegression()
    model.fit(X_train_fs, y_train)
    # evaluate the model
    yhat = model.predict(X_test_fs)
    # evaluate predictions
    mae = mean_absolute_error(y_test, yhat)
    r2 = r2_score(y_test, yhat)
    per = scipy.stats.pearsonr(y_test, yhat).correlation
    print('Correlation scores - MAE: %.3f' % mae)
    print('Correlation scores - R2: %.3f' % r2)
    print('Correlation scores - Person: %.3f' % per)
    print(fs.get_support(indices=True))

    #import matplotlib.pyplot as plt
    # Plot outputs
    #plt.scatter(X_train_fs, y_train, color="green")
    #plt.scatter(X_test_fs, y_test, color="black")
    #plt.plot(X_test_fs, yhat, color="blue", linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()

    # Mutual Information Features

    # evaluation of a model using 88 features chosen with mutual information
    from sklearn.feature_selection import mutual_info_regression

    # feature selection
    def select_features_mutual(X_train, y_train, X_test):
        # configure to select a subset of features
        fs = SelectKBest(score_func=mutual_info_regression, k=7)
        # learn relationship from training data
        fs.fit(X_train, y_train)
        # transform train input data
        X_train_fs = fs.transform(X_train)
        # transform test input data
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    # feature selection
    X_train_fs, X_test_fs, fs = select_features_mutual(X_train, y_train, X_test)
    # fit the model
    model = LinearRegression()
    model.fit(X_train_fs, y_train)
    # evaluate the model
    yhat = model.predict(X_test_fs)
    # evaluate predictions
    mae = mean_absolute_error(y_test, yhat)
    r2 = r2_score(y_test, yhat)
    per = scipy.stats.pearsonr(y_test, yhat).correlation
    per_train = scipy.stats.pearsonr(model.predict(X_train_fs), y_train).correlation
    print('Mutual Information - MAE: %.3f' % mae)
    print('Mutual Information - R2: %.3f' % r2)
    print('Mutual Information - Person: %.3f' % per)
    print('Mutual Information - Person Train: %.3f' % per_train)
    print(fs.get_support(indices=True))
    print("coef:", model.coef_)
    print("intercept:", model.intercept_)






if __name__ == '__main__':
    main()

