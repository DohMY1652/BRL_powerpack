import yaml

file = "../exp/240708_17_43_01_PID_Real/cfg.yaml"

with open(file, "r") as f:
    data = yaml.safe_load(f)

print(data)