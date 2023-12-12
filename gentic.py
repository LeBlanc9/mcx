import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import differential_evolution
from simulator import Simulator
from utils import log


simulator = Simulator()
def simulateReflection(op):
    ua,us = op 
    simulator.cfg['prop'] = [[0,0,1,1],[ua,us,0.8,1.37]]
    simulator.simulate()
    r0 = simulator.get_detect_r()
    return r0


def objectiveFunction(op):
    r_target = [0.017224096536046313,0.007471655626211807,0.0060548643240016875,0.004308402758773683,0.002837477484125674,0.0020984815783706458,0.0023184437171650347,0.0018082351173437487,0.00142540206117369,0.001463391218167464]
    r0 = simulateReflection(op)
    diff = abs(r0 - r_target).sum()  # 使用绝对差作为目标函数

    log("___"*30)
    log("Optimized ua:", op[0])
    log("Optimized us:", op[1])
    log("Optimized Reflectance (r):", list(r0))
    log("Target Reflectance (r):", r_target)

    return diff

# 定义参数边界
bounds = Bounds([0, 0], [1, 1])


# 调用differential_evolution函数进行优化
result = differential_evolution(objectiveFunction, bounds)

# 输出优化结果
print('优化得到的参数ua, ub分别为:')
print(result.x)
print('对应的反射率r0为:')
print(simulateReflection(result.x))