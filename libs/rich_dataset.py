import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class RICHDataset(Dataset):
    def __init__(self, df, kinematic_colums, label_cols, device='cpu'):
        self.df = df.reset_index().drop(columns=('index'))
        self.kin_cols = kinematic_colums
        self.device = device
        self.label_cols = label_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        track_df = self.df.iloc[idx]
        kin_vector = torch.tensor(np.array(track_df[self.kin_cols], dtype=np.float32))
        
        rich_view = torch.zeros((1,23*8,28*8))
        x = track_df['x'].split(';')[:-1]
        y = track_df['y'].split(';')[:-1]
        time = track_df['time'].split(';')[:-1]
        for i,j,t in zip(x,y,time):
            rich_view[0,int(i),int(j)] = float(t)

        label = torch.tensor(track_df[self.label_cols].values[0], dtype=torch.int64)

        return kin_vector.to(self.device), rich_view.to(self.device), label.to(self.device)
