import ROOT
import json
import numpy as np
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pathlib import Path
from array import array

from libs.rich_dataset import RICHDataset
from libs.model import Generator

import warnings
warnings.filterwarnings('ignore')

def run_inference(json_name, chunk_size, csv_path):
    json_data = json.load(open(json_name))

    exp_name_to_load = json_name.resolve().parent

    root_name = str(exp_name_to_load / json_data['root_name'])
    model_name = str(exp_name_to_load / json_data['model_name'])
    data_mean = torch.tensor(json_data['data_mean'])
    data_std = torch.tensor(json_data['data_std'])
    kin_cols = json_data['track_cols']
    pred_cols = json_data['out_cols']
    particles = json_data['particle_names']
    hit_cols = json_data['hits_cols']

    model_NN = Generator(len(kin_cols), len(particles))
    model_NN.load_state_dict(torch.load(model_name))
    model_NN = model_NN.cpu()

    event_id = array('i',[0])
    hits_num = array('i', [0])
    pred_nn = array('i', [0])
    conf_nn = array('d', [0])

    pred_mod_50 = array('i',[0])
    pred_mod_85 = array('i',[0])

    tree_vals = [event_id]
    tree_keys = ['eventID']

    res_file = ROOT.TFile(root_name, "RECREATE")
    res_tree = ROOT.TTree("resTree", "resTree")

    res_tree.Branch('event_id',event_id,'event_id/I')
    res_tree.Branch("pred_nn", pred_nn,"pred_nn/I")
    res_tree.Branch("conf_nn", conf_nn,"conf_nn/D")
    res_tree.Branch("pred_mod_50", pred_mod_50,"pred_mod_50/I")
    res_tree.Branch("pred_mod_85", pred_mod_85,"pred_mod_85/I")

    chunk_num=0

    main_df = pd.read_csv(csv_path, chunksize=chunk_size, index_col=False)
    df_residual = pd.DataFrame()
    iter_start = True

    print('***********************')
    print('*** START INFERENCE ***')
    print('***********************')
    while iter_start:
        try:
            df_chunk = next(main_df)
            df_chunk = pd.concat([df_residual, df_chunk])
            df_residual = df_chunk[df_chunk['eventID'] == df_chunk['eventID'].iloc[-1]]
            df_process = df_chunk[df_chunk['eventID'] != df_chunk['eventID'].iloc[-1]]
        except StopIteration:
            df_process = df_residual
            iter_start = False

        df_process[kin_cols] = (df_process[kin_cols] - data_mean)/data_std
        main_set = RICHDataset(df_process, kin_cols, pred_cols, hit_cols)
        main_data = DataLoader(main_set, batch_size=10, shuffle=False)
        
        idx=0
        for kin_vec, rich_view, _ in main_data:
            if (idx+chunk_num*chunk_size) % 1000 == 0:
                print(f'[INFERENCE] : Processed events : {idx+chunk_num*chunk_size}')
            with torch.no_grad():
                probs = model_NN(kin_vec, rich_view)
            probs = probs.detach().numpy()
            for prob in probs:
                for v,k in zip(tree_vals,tree_keys):
                    v[0] = df_process[k].iloc[idx]
                idx+=1
                pred_nn[0] = 211 if prob.argmax()==0 else 321
                conf_nn[0] = float(1 - prob.min()/prob.max())
                pred_mod_50[0] = 211 if np.random.random()<0.5 else 321
                pred_mod_85[0] = 211 if np.random.random()<0.85 else 321
                res_tree.Fill()
        chunk_num+=1

    print(f'[INFERENCE] : Processed events : {idx+chunk_num*chunk_size+1}')
    res_file.Write()
    print('***********************')
    print('*** ROOT FILE SAVED ***')
    print('***********************')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--json_path', help='Path to JSON file from training results')
    parser.add_argument('-c','--chunk_size', help='Number of lines to load from csv')
    parser.add_argument('-i','--input', help='Path to inference csv file')

    args = parser.parse_args()
    run_inference(Path(args.json_path), int(args.chunk_size), args.input)
