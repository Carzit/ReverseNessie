import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from safetensors.torch import save_file, load_file

class NessieDataset(Dataset):

    def __init__(self, json_path, pad=True):
        super(NessieDataset, self).__init__()
        self.pad = pad

        with open(json_path, 'r') as file:
            data = json.load(file)
            self.Xs = data['data_X']
            self.Ys = data["data_Y"]
        self.seq_len = max([len(seq) for seq in self.Ys])

    def __len__(self):
        return min(len(self.Xs),len(self.Ys))

    def __getitem__(self, index):
        cme_params = torch.tensor(self.Xs[index])
        xp = torch.tensor(self.Ys[index])
        x_seq, p_seq = xp.transpose(0,1)
        if self.pad:
            x_seq = torch.linspace(0, self.seq_len-1, self.seq_len)
            p_seq = F.pad(p_seq, (0., self.seq_len-len(p_seq)))
        return cme_params, x_seq, p_seq
    
    def info(self, index=0)->tuple:
        return len(self), self[index][0].shape, self[index][1].shape
    
    def split_dataset(self, ratios:list=None):
        if ratios is None:
            ratios = [0.7, 0.2, 0.1]
        total_length = len(self)
        split_lengths = list(map(lambda x:round(x / sum(ratios) * total_length), ratios))
        split_lengths[0] = total_length - sum(split_lengths[1:])
        return random_split(self, split_lengths)
    

def split_dataset(dataset:Dataset, ratios:list=None):
    if ratios is None:
        ratios = [0.7, 0.2, 0.1]
    total_length = len(dataset)
    split_lengths = list(map(lambda x:round(x / sum(ratios) * total_length), ratios))
    split_lengths[0] = total_length - sum(split_lengths[1:])
    return random_split(dataset, split_lengths)


if __name__ == "__main__":
    dataset = NessieDataset(json_path="data\\data_ssa")
    train_set, val_set, test_set = split_dataset(dataset)
    save_sets = {"train_set":train_set,
                 "val_set":val_set,
                 "test_set":test_set}
    torch.save(save_sets, "splited_dataset.pt")
