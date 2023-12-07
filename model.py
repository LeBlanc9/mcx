import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class FCNet(nn.Module): 
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 15),
            nn.ReLU(),
            nn.Linear(15, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )

    def forward(self, x):
        return self.net(x)


class MyDataset(Dataset):
    def __init__(self) -> None:
        self.data = pd.read_csv('50_test.csv')

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index):
        all_reflectance =  [self.data.iloc[index,i] for i in range(3,13)]

        selected_reflectance = torch.tensor(all_reflectance[1::2], dtype=torch.float)
        optical_parameter = torch.tensor([self.data.iloc[index, 1], self.data.iloc[index, 2]], dtype=torch.float)
        return  selected_reflectance, optical_parameter


if __name__ == "__main__":
    model = FCNet()
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False) 
    for batch_index, (reflectance, optical_parameter) in enumerate(dataloader):
        print(batch_index)
        print(type(reflectance))
        print(reflectance, reflectance.shape)
        print(type(optical_parameter))
        print(optical_parameter, optical_parameter.shape)
        break