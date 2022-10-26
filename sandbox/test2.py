import json
import numpy as np


# Data to be written
reg_model = {
    "name": "regression model",
    "x1_p1": 1,
    "x1_p2": 2,
    "x2_p1": 1,
    "x2_p2": 2,
}

# Serializing json
json_object = json.dumps(reg_model, indent=4)

# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)