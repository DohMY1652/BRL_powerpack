import json

file_names = [
    "ctrl.json",
    "obs.json",
]

data = dict(
    time = 0,
    sen_pos = 0,
    sen_neg = 0,
    ref_pos = 0,
    ref_neg = 0,
    ctrl_pos = 0,
    ctrl_neg = 0
)

for fn in file_names:
    with open(fn, 'w') as f:
        json.dump(data, f)
