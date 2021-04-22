# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from sklearn import metrics
from optuna.trial import TrialState
from model import DPCNN
from load_data import train_iter, val_iter, id2vocab

EPOCHS = 10
CLS = 2
device = "cuda" if torch.cuda.is_available() else 'cpu'

def objective(trial):

    model = DPCNN(trial, len(id2vocab), CLS)
    model.to(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss= []
        for batch in train_iter:           
            text_idx_batch, label_idx_batch = batch.text.t_().to(device), batch.label.to(device)
            model.zero_grad()
            out = model(text_idx_batch)
            loss = criterion(out, label_idx_batch)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()   
        #print(f'Epoch[{epoch}] - Loss:{sum(epoch_loss)/len(epoch_loss)}')

        model.eval()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():        
            for batch in val_iter:
                text_idx_batch, label_idx_batch = batch.text.t_().to(device), batch.label
                pred = model(text_idx_batch)
                pred = torch.max(pred.data, 1)[1].cpu().numpy()
                predict_all = np.append(predict_all, pred)
                
                truth = label_idx_batch.cpu().numpy()
                labels_all = np.append(labels_all, truth)            
            
        acc = metrics.accuracy_score(labels_all, predict_all)
        
        trial.report(acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return acc


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))    
