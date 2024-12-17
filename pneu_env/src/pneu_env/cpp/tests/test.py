import numpy as np
import pandas as pd
from collections import deque

from ctypes import CDLL, c_double

lib = CDLL("./libMain.so")

lib.solenoid_valve.argtypes = [c_double, c_double, c_double] # pressure inlet [kPa], pressure outlet [kPa], current [A]
lib.solenoid_valve.restype = c_double
solenoid_valve = lambda press_inlet, press_outlet, current: lib.solenoid_valve(press_inlet, press_outlet, current)


current = 0.001*np.arange(0, 165, 0.1)

dPs = [700, 500, 350, 200]
ATM = 101.325
R = 287 # J/kg/K
T = 293.15 # K

logger = dict()
logger["current"] = current
for dP in dPs:

    press_inlet = ATM + dP
    press_outlet = ATM

    density = 1000*press_inlet/R/T

    dV = deque()
    for I in current:
        dV.append(solenoid_valve(press_inlet, press_outlet, I)*60000/density)
    logger[f"dP{dP}"] = np.array(dV)

print(logger)

df = pd.DataFrame(logger)
df.to_csv("output.csv", index=False, header=True)

        