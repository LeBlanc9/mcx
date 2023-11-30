import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pmcx
from utils import *

class Simulator:
    cfg = {}
    res = {}

    def __init__(self):
        self.default_configure()
        
    @classmethod
    def default_configure(cls) -> dict:
        det_radius = 0.25 
        cls.cfg = {
            'nphoton': 1e7,
            'vol':np.ones([100,100,100],dtype='uint8'),
            'tstart':0,
            'tend':5e-9,
            'tstep':5e-9,
            'srcpos': [50,10,0],
            'srcdir':[0,0,1],
            'prop':[[0,0,1,1],[7,1,0.1,1.37]],
            'detpos':[[50,12,0, det_radius], [50,14,0,det_radius], [50,16,0,det_radius], [50,18,0,det_radius]],
            'unitinmm': 0.1,
            'issavedet':1,
            'issaveseed':1,
            'issrcfrom0':1,
            'savedetflag':'dpx',
        }
        #cls.cfg['vol'][:,:10,:] = 0
        return cls.cfg

    @classmethod
    def simulate(cls, cfg="default") -> dict:
        if cfg == "default":
            cfg = cls.default_configure()

        cls.res = pmcx.mcxlab(cfg)
        cls.flux = np.squeeze(cls.res['flux'])
        return cls.res

    @classmethod
    def slice_visualize(cls, slice_idx=50):
        reflectence = cls.flux[slice_idx,:,:]
        plt.imshow(np.log10(reflectence))
        plt.colorbar()
        plt.show()

def get_optical_parameters() -> list:
    optical_parameters = []
    number = 500 
    ua_group = np.linspace(0.05, 0.3, number)
    us_group = np.linspace(0.5, 3.5, number)
    for i in range(len(ua_group)):
        for j in range(len(us_group)):
            optical_parameters.append((ua_group[i], us_group[j])) 

    return optical_parameters

def test_Simulator():
    simulator = Simulator()
    simulator.simulate()
    simulator.slice_visualize()
  
    plt.scatter(simulator.res['detp']['p'][simulator.res['detp']['detid']==1,0], simulator.res['detp']['p'][simulator.res['detp']['detid']==1,1], marker='.',color='red');
    plt.show()

    reflectance = pmcx.cwdref(simulator.res['detp'], simulator.cfg)
    print("-----------------Reflectance Below----------------------")
    plt.scatter(reflectance)
    plt.show()


def get_reflectances(optical_parameters):
    reflectances = []
    for ua,us in optical_parameters:
        simulator = Simulator()
        simulator.cfg['prop'] = [[0,0,1,1],[ua,us, 0.8,1.37]],
        simulator.simulate() 

        reflectance = pmcx.cwdref(simulator.res['detp'], simulator.cfg)
        reflectances.append(reflectance) 
    return reflectances


def save_simulated_data(optical_parameters, reflectances):
    df = pd.DataFrame(columns=['ua','us','r1','r2','r3','r4'])
    print(len(reflectances))
    print(len(optical_parameters))
    print(reflectances)
    print(optical_parameters)

    for i in range(len(reflectances)):
        df.loc[i] = [optical_parameters[i][0], optical_parameters[i][1], reflectances[i][0], reflectances[i][1], reflectances[i][2], reflectances[i][3]]
    df.to_csv('test.csv')

if __name__ == "__main__":
    optical_parameters = get_optical_parameters()
    reflectances = get_reflectances(optical_parameters)
    save_simulated_data(optical_parameters, reflectances) 