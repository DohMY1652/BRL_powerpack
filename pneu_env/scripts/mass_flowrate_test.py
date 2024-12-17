import numpy as np

from pneu_env.sim import PneuSim


sim = PneuSim(
    freq = 1/0.02
)

sim.set_discharge_coeff(1e-6, 1e-6)

dummy_goal = np.array([101.325, 101.325])

print(sim.observe(
    ctrl = np.array([1.0, 1.0]),
    goal = dummy_goal
))

print(sim.get_mean_mass_flowrate())

print(sim.observe(
    ctrl = np.array([1.0, 1.0]),
    goal = dummy_goal
))

# print(np.array(sim.get_mean_mass_flowrate().values())*60000/1.2041)
print(sim.get_mean_mass_flowrate())

print(sim.get_mass_flowrate())
