import matplotlib.pylab as plt
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
            'prop':[[0,0,1,1],[7,1,0.1,1.37]],
            'unitinmm': 0.1,
            'issavedet':1,
            'issaveseed':1,
            'issrcfrom0':1,
            'savedetflag':'dpx',
        }
        cls.__set_detpos()
        #cls.cfg['vol'][:,:10,:] = 0
        return cls.cfg

    @classmethod
    def simulate(cls) -> dict:
        cls.res = pmcx.mcxlab(cls.cfg)
        cls.flux = np.squeeze(cls.res['flux'])
        return cls.res

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
    reflectance = pmcx.cwdref(simulator.res['detp'], simulator.cfg)
    print(reflectance)