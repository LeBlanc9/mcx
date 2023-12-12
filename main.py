import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pmcx
from utils import *
from simulator import Simulator
from InverseMC import InverseMC
from trainer import *
import logging

# 获取模块的日志记录器
logger = logging.getLogger('module_name')

# 设置日志记录级别为CRITICAL，这将禁用除了CRITICAL级别以外的所有日志记录
logger.setLevel(logging.CRITICAL)


if __name__ == "__main__":
    inverseMC = InverseMC()
    sim_df = pd.read_csv("dataset/10_test.csv")
    iMC_df = pd.DataFrame(columns=['ua','us']+[f"r{i}" for i in range(1,11)])

    for i in range(len(sim_df)):
        sim_ua = sim_df['ua'][i] 
        sim_us = sim_df['us'][i] 

        reflectence = sim_df.iloc[i, 3:]   
        res = inverseMC.optimize(target_r=reflectence)

        iMC_df.loc[i] = list(res["op"]) + list(res['reflectence'])
        iMC_df.to_csv("dataset/10_test_iMC.csv")


        print("_"*45)
        print("True Value: ")
        print(f"ua: {sim_ua}")
        print(f"us: {sim_us}")