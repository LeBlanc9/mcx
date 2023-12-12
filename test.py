import matlab.engine
import matlab

eng = matlab.engine.start_matlab()

x = 9
y = eng.sqrt(matlab.double(x))

print(y)
