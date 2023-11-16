import json
import shutil
import argparse

import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from libs.rich_dataset import RICHDataset
from libs.model import Generator
from utils import split_data, conf_mat_report


def run_training(json_name):
    params = json.load(open(json_name))

    kin_cols = params['input_cols']
    hit_cols = params['hits_cols']
    pred_cols = params['out_cols']
    particles = params['particle_names']
    split_ratio = params['split_ratio']
    batch_size = params['batch_size']
    iter_nums = int(params['iterations'])
    check_iter = params['printout_iter']
    lr = params['learning_rate']
    model_name = params['model_name']
    output_folder = Path(params['output_folder'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    df_total = pd.read_csv(params['file_path'],index_col=False)
    df_train, df_test = split_data(df_total, split_ratio)
    
    data_mean, data_std =  df_train[kin_cols].mean(), df_train[kin_cols].std()

    df_train[kin_cols] = (df_train[kin_cols] - data_mean)/data_std
    df_test[kin_cols] = (df_test[kin_cols] - data_mean)/data_std 

    train_set = RICHDataset(df_train, kin_cols, pred_cols, hit_cols, device)
    test_set = RICHDataset(df_test, kin_cols, pred_cols, hit_cols, device)

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_set, batch_size=int(len(test_set)/8), shuffle=False)


    rich_model = Generator(len(kin_cols), len(particles))

    if torch.cuda.is_available():
        rich_model = rich_model.cuda()

    best_model = None
    best_acc = 0

    optimizer = torch.optim.Adam(rich_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_v = 0
    iter_train_data = iter(train_data)


    print('*************************')
    print('**** TRAINING STARTS ****')
    print('*************************')
    
    for i in range(iter_nums):
        if iter_nums == int(iter_nums*0.8):
            lr*=0.1
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
        try:
            kin_vec, rich_view, label = next(iter_train_data)
        except StopIteration:
            iter_train_data = iter(train_data)
            kin_vec, rich_view, label = next(iter_train_data)

        rich_model.zero_grad()
        y_preds = rich_model(kin_vec, rich_view)
        loss = loss_fn(y_preds, label)
        loss.backward()
        optimizer.step()
        loss_v += float(loss)
        
        if (i+1)%check_iter == 0 or (i+1)==iter_nums:
            with torch.no_grad():
                acc = 0
                test_loss = 0
                for test_kin_vec, test_rich_view, test_label in test_data:
                    y_test_preds = rich_model(test_kin_vec, test_rich_view).argmax(axis=1)
                    acc += sum(test_label == y_test_preds)
                acc = acc/len(test_set)*100
            print(f'Iter {i+1} of {iter_nums} || Train Loss : {loss_v/check_iter:.4} || Test Acc : {acc:.4}%')
            if acc > best_acc:
                best_acc = acc
                best_model = deepcopy(rich_model)
            loss_v = 0

    cm = np.zeros((len(particles),len(particles)),dtype=np.uint64)
    with torch.no_grad():
        for test_kin_vec, test_rich_view, test_label in test_data:
            y_test_preds = best_model(test_kin_vec, test_rich_view).argmax(axis=1)
            y_p = y_test_preds.cpu().detach().numpy()
            y_l = test_label.cpu().numpy()
            for i,j in zip(y_l, y_p):
                cm[i,j] += 1

    cm_r = conf_mat_report(cm,particles)
    acc = (cm[0][0]+cm[1][1])/cm.sum()


    print('****************************')
    print('**** TRAINING COMPLETED ****')
    print('****************************')

    print(f'Accuracy on testset : {acc*100:.4} %')
    
    dt = datetime.now()
    params['date'] = f'{dt.day}:{dt.month}:{dt.year}'
    params['time'] = f'{dt.hour}:{dt.minute}:{dt.second}'
    params['data_mean'] = list(data_mean)
    params['data_std'] = list(data_std)
    params['test_acc'] = acc
    params['conf_mat_report'] = cm_r

    output_folder.mkdir(exist_ok=True, parents=True)
    torch.save(best_model.state_dict(), output_folder/model_name)
    with open(output_folder/json_name,'w') as fw:
        json.dump(params,fw,indent=2)
    
    print('*********************')
    print('**** MODEL SAVED ****')
    print('*********************')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input JSON')

    args = parser.parse_args()
    run_training(args.input)
