import matplotlib.pylab as plt
import sys
import numpy as np
import pmcx

class Simulator:
    cfg = {}
    res = {}

    def __init__(self):
        self.default_configure()
        
    @classmethod
    def default_configure(cls) -> dict:
        cls.cfg = {
            'nphoton': 1e7,
            'vol':np.ones([100,100,100],dtype='uint8'),
            'tstart':0,
            'tend':5e-9,
            'tstep':5e-9,
            'srcpos': [50,10,0],
            'srcdir':[0,0,1],
            'prop':[[0,0,1,1],[7,1,0.8,1.37]],
            'unitinmm': 0.1,
            'issavedet':1,
            'issaveseed':1,
            'issrcfrom0':1,
            'savedetflag':'dpx',
        }
        cls.__set_detpos()
        cls.cfg['vol'][:,:,:5] = 2
        cls.cfg['prop'] = [[0,0,1,1],[7,1,0.8,1.37], [0.05,0.5,0.8,1.37]] 
        
        return cls.cfg

    ## 根据cls.cfg参数进行仿真
    # return {'flux', ...}
    @classmethod
    def simulate(cls) -> dict:
        cls.res = pmcx.mcxlab(cls.cfg)
        cls.flux = np.squeeze(cls.res['flux'])
        return cls.res

    ## 获取探测器采集到的漫反射率
    # return [r1, r2, r3, ...]
    @classmethod 
    def get_detect_r(cls) -> list:
        if cls.res == {}:
            raise Exception("Please make sure simulated with detpos settled first")
        reflectance = pmcx.cwdref(Simulator.res['detp'], Simulator.cfg)
        return reflectance

    ## 调试使用，获取某个切片部位的flux图
    @classmethod
    def slice_visualize(cls, slice_idx=50):
        reflectence = cls.flux[slice_idx,:,:]
        plt.imshow(np.log10(reflectence))
        plt.colorbar()
        plt.show()


    # 设置探测器们的位置
    @classmethod
    def __set_detpos(cls) -> list:
        detpos = []
        det_radius = 0.25  #对100um一格来说，为半径25um
        for i in range(11,21, 1):
            sublist = [50, i, 0, det_radius]
            detpos.append(sublist)

        cls.cfg['detpos'] = detpos
        return detpos

if __name__ == "__main__":
    simulator = Simulator()
    simulator.simulate()
    simulator.slice_visualize()
    #reflectance = pmcx.cwdref(simulator.res['detp'], simulator.cfg)
    reflectance = simulator.get_detect_r()
    print(reflectance)