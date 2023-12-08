import numpy as np
from simulator import Simulator
from scipy.optimize import least_squares, leastsq
import pmcx

##
# op -> Optical Parameter
#  

class InverseMC(): 

    def __init__(self, target_r):
        Simulator.default_configure()

        self.init_op = [0.1, 0.5]
        self.target_r = target_r 

    def run(self):
        for i in range(100):
            res, flag = leastsq(self.loss, self.init_op, args=(self.target_r,), full_output=True)

            # 输出目标函数的值
            print('iteration:', i+1, 'loss:', flag['fvec'])
            # 判断是否收敛，如果收敛则跳出循环
            if flag['ierr'] == 1:
                break
            # 更新初始光学参数
            u0 = res              

    @staticmethod
    def loss(op, target_r):
        ua, ub = op[0], op[1]
        Simulator.cfg['prop'] = [[0,0,1,1], [ua,ub,0.8,1.37]]
        Simulator.simulate()
        r = Simulator.get_detect_r()[1]

        loss = r - target_r
        return loss


if __name__ == "__main__":
    inverseMC = InverseMC(target_r=0.0001)
    inverseMC.run()