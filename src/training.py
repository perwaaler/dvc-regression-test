import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import io
import sys
from sklearn.metrics import mean_squared_error
import math

# get the parameter for the training stage:
params = yaml.safe_load(open("parameters/params.yaml"))['training']
# get the path to the data. Note: sys.argv contains a list of the command line arguments used when running a commands
data_path = sys.argv[1]+"/out.csv"
# read output from the filter stage:
df = pd.read_csv(data_path)
# get the number of rows specified by the num_rows parameter:
df = df.head(params['num_rows'])
print("taking the mean of:")

# get the mean of the y-column
mean = df['y'].mean()
print(mean)

# create directory where output of this stage is stored:
os.makedirs(os.path.join("output","training"), exist_ok=True)
# create name (with path) of output:
output_training = os.path.join("output","training","out.csv")

### linear regression ###
p = 3
print(p)
x = df.x1
A = np.vstack([x, np.abs(x)**p, np.ones(len(x))]).T
y = df.y
c = np.linalg.lstsq(A, y)[0]
y_pred = A @ c.T
e = y_pred - y

MSE = mean_squared_error(y, y_pred)
RMSE = math.sqrt(MSE)

print("the RMSE of the fit is:")
print(RMSE)

with io.open(output_training,"w",encoding="utf8") as fd_out_training:
    fd_out_training.write(str(mean))




