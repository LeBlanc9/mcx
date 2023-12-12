##-- 获取训练数据，以及进行训练和测试 --##
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from simulator import Simulator
from model import FCNet, MyDataset
import numpy as np
import pandas as pd
import pmcx


class Trainer():
    def __init__(self, print_info=True) -> None:
        ## 训练配置
        self.print_info = print_info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if print_info: print(f"当前设备: {self.device}")


    def tain(self, epoch_num=100):
        model = FCNet().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        dataset = MyDataset()
        train_set, test_set = random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

        writer = SummaryWriter()
        step = 0

        ##--- Training ---##
        for epoch in range(epoch_num):
            if self.print_info: print(f"--- Epoch {epoch} ---")
            # r -> reflectance
            # op -> optical parameter
            for batch_id, (r, op) in enumerate(train_loader):
                r, op = r.to(self.device), op.to(self.device)

                pred = model(r) 
                loss = criterion(pred, op)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if self.print_info: print(f"{batch_id}: loss-{loss}")
                writer.add_scalar('Loss', loss, step)
                step += 1

        writer.close()


    def test(self):
        pass



class Database:
    single_layer = "one"
    two_layers = "two"
    three_layers = "three"

    def __init__(self, mode="one") -> None:
        self.mode = mode 


    ## 获取需要仿真光学参数们到optical_parameters列表
    #   return [[op1, op2, ...], 
    #           [op1, op2, ...], 
    #                 ...      ]
    def __get_optical_parameters(self) -> list:
        if self.mode == self.single_layer:
            for i in range(len(self.ua_group)):
                self.optical_parameters.append([self.ua_group[i], self.us_group[i]])

        elif self.mode == self.two_layers:
            for i in range(len(self.ua1_group)):
                self.optical_parameters.append([self.ua1_group[i], self.ua1_group[i],self.ua1_group[i], self.us2_group[i]])


    ## 对optical_parameters列表中的参数进行逐一仿真，获得漫反射值
    #   return [[r1,r2,r3,...],
    #           [r1,r2,r3,...],
    #               ...       ]
    def __get_reflectances(self) -> list:
        simulator = Simulator()

        if self.mode == self.single_layer:
            for ua,us in self.optical_parameters:
                simulator.cfg['prop'] = [[0,0,1,1],[ua,us, 0.8,1.37]]
                simulator.simulate() 
                reflectance = simulator.get_detect_r()
                self.reflectances.append(reflectance)

        elif self.mode == self.two_layers:
            for ua1,ua2,us1,us2 in self.optical_parameters:
                simulator.cfg['vol'][:,:,:5] = 2
                simulator.cfg['prop'] = [[0,0,1,1],[ua1,us1, 0.8,1.37],[ua2,us2, 0.8,1.37]]
                simulator.simulate() 
                reflectance = simulator.get_detect_r()
                self.reflectances.append(reflectance)

        return self.reflectances


    ## 保存数据为CSV文件 
    def __save_simulated_data(self, filename='test.csv') -> None:
        if self.mode == self.single_layer:
            df = pd.DataFrame(columns=['ua','us']+[f"r{i}" for i in range(1,11)])
        elif self.mode == self.two_layers:
            df = pd.DataFrame(columns=['ua1','ua2','us1','us2']+[f"r{i}" for i in range(1,11)])
        for i in range(len(self.reflectances)):
            df.loc[i] = list(self.optical_parameters[i]) + list(self.reflectances[i])

        df.to_csv('dataset/' + filename)


    ## 仿真获取训练数据并保存为csv
    def get_csv_dataset(self, number=10000, filename='test.csv') -> None:
        self.optical_parameters = []
        self.reflectances = []

        if self.mode == self.single_layer:
            ## 设置需要进行仿真的数据集
            self.ua_group = np.random.uniform(0.05, 0.3, number)
            self.us_group = np.random.uniform(0.5, 3.5, number)

        elif self.mode == self.two_layers:
            self.ua1_group = np.random.uniform(0.05, 0.3, number)
            self.ua2_group = np.random.uniform(0.05, 0.3, number)
            self.us1_group = np.random.uniform(0.5, 3.5, number)
            self.us2_group = np.random.uniform(0.5, 3.5, number)

        self.__get_optical_parameters()
        self.__get_reflectances()
        self.__save_simulated_data(filename=filename)


if __name__ == "__main__":
    #trainer = Trainer()
    #trainer.tain()
    database = Database()
    database.mode = Database.two_layers
    database.get_csv_dataset(number=10000, filename='2layers_10000_random_uniform.csv')