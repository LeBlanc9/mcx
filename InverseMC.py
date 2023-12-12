import numpy as np
from scipy.optimize import least_squares
from simulator import Simulator
from utils import log


class InverseMC:
    def __init__(self):
        self.target_r = [0.017224096536046313,0.007471655626211807,0.0060548643240016875,0.004308402758773683,0.002837477484125674,0.0020984815783706458,0.0023184437171650347,0.0018082351173437487,0.00142540206117369,0.001463391218167464]
        self.initial_params = [0.01,0.8]  # 你需要根据你的问题提供一个合理的初始估计


    def optimize(self):
        result = least_squares(self.cost_function, self.initial_params, args=(self.target_r,), bounds=[0.001,1.5], f_scale=1)
        # 输出结果
        optimized_op = result.x

        simulator = Simulator()
        simulator.cfg['prop'] = [[0,0,1,1],[optimized_op[0],optimized_op[1], 0.8,1.37]]
        simulator.simulate()
        optimized_r = simulator.get_detect_r()

        log("Optimized ua:", optimized_op[0])
        log("Optimized us:", optimized_op[1])
        log("Optimized Reflectance (r):", list(optimized_r))
        log("Target Reflectance (r):", self.target_r)

        result = {
            "op": optimized_op,
            "reflectence": optimized_r
        }
        return result


    @staticmethod
    def cost_function(op, target_r):
        ua, us = op

        log("-"*50)
        log(op)

        simulator = Simulator()
        simulator.cfg['prop'] = [[0,0,1,1],[ua,us, 0.8,1.37]]
        simulator.simulate()
        reflectence = simulator.get_detect_r()
        cost = np.sqrt((target_r-reflectence)**2)
        log("cost: ", cost.sum())

        return cost


if __name__ == "__main__":
    inverseMC = InverseMC()
    inverseMC.optimize()