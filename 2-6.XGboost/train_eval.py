# -*- coding: utf-8 -*-
import xgboost as xgb
import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import optuna

def transform_data(file_path):
    text = []
    labels = []
    with open(file_path, 'r', encoding='utf8') as rf:
        r = csv.reader(rf, delimiter='\t')
        next(r)
        for row in r:
            labels.append(int(row[1]))
            text.append(' '.join(w for w in jieba.cut(row[2])))
    return text, np.array(labels)

train_text, train_y = transform_data('./data/train.tsv')
vectorizer = TfidfVectorizer(max_features=5000)
train_x = vectorizer.fit_transform(train_text)
train_x = train_x.toarray()

val_text, val_y = transform_data('./data/dev.tsv')
val_x = vectorizer.transform(val_text)
val_x = val_x.toarray()

dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(val_x, label=val_y)

def objective(trial):
    param = {'silent': 0, 
          'objective': 'binary:logistic', 
          'eval_metric': 'error',
          'eta': trial.suggest_float("eta", 1e-8, 1.0, log=True), 
          'max_depth': trial.suggest_int("max_depth", 3, 9, step=2),
          "gamma" : trial.suggest_float("gamma", 1e-8, 1.0, log=True),
          "min_child_weight" : trial.suggest_int("min_child_weight", 2, 10)}

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = metrics.accuracy_score(val_y, pred_labels)
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))