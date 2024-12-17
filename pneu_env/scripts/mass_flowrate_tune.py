import numpy as np

from pneu_env.tuner import PneuSimTuner

kwargs = dict(
    data_names = [
        "241002_14_54_53_Flowrate_pump_pos",
        "241002_14_56_56_Flowrate_pump_neg",
        "241002_14_59_07_Flowrate_095_pump_pos",
        "241002_15_01_12_Flowrate_095_pump_neg",
        "241002_15_03_41_Flowrate_090_pump_pos",
        "241002_15_05_44_Flowrate_090_pump_neg"
    ]
)


tuner = PneuSimTuner(**kwargs)

print(list(tuner.datas[kwargs["data_names"][0]].keys()))
# print(tuner.datas[kwargs["data_names"][0]]["flowrate1"]*1.2041/60000)
tuner.objective_function(np.array([1.5, 1]))
tuner.objective_function(np.array([1.2, 1]))

