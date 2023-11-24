import matplotlib.pyplot as plt
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
            'vol':np.ones([60,60,60],dtype='uint8'),
            'tstart':0,
            'tend':5e-9,
            'tstep':5e-9,
            'srcpos': [30,30,0],
            'srcdir':[0,0,1],
            'prop':[[0,0,1,1],[0.005,0.1,0.01,1.37]]
        }
        return cls.cfg

    @classmethod
    def simulate(cls, cfg="default") -> dict:
        if cfg == "default":
            cfg = cls.default_configure()

        cls.res = pmcx.mcxlab(cfg)
        return(cls.res)

    @classmethod
    def visualize(cls):
        reflectence = cls.res['flux'][30,:,30]
        x = np.arange(60)
        plt.plot(x, reflectence)
        plt.savefig("temp.png")


if __name__ == "__main__":
    Simulator.simulate()
    Simulator.visualize()