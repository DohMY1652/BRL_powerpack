from pneu_env.sim import PneuSim

sim = PneuSim()

print(sim.solenoid_valve(
    Pin = 141.325,
    Pout = 101.325,
    ctrl = 0.98,
    type = 1,
    num = 3
))
