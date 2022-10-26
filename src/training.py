import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import io
import json
import sys
from sklearn.linear_model import LinearRegression
import pickle

### linear regression ###
def prepare_reg_input(df,params):
    """"prepare input for sklearn linear regression fit"""
    x1 = df.x1.values
    x2 = df.x2.values
    x1_p = x1**params['p1']
    x2_p = x2**params['p2']

    X = np.vstack((x1, x2, x1_p, x2_p)).T
    y = df.y.values.reshape(-1,1)
    return X,y

# get the parameter for the training stage:
params = yaml.safe_load(open("parameters/params.yaml"))['training']
# get the path to the data. Note: sys.argv contains a list of the command line arguments used when running a commands
data_path = sys.argv[1]+"/out.csv"
# read output from the filter stage:
df = pd.read_csv(data_path)
# get the number of rows specified by the num_rows parameter:
df = df.head(params['num_rows'])


# get the mean of the y-column
mean = df['y'].mean()
print("taking the mean of:")
print(mean)

# create directory where output of this stage is stored:
os.makedirs(os.path.join("output","training"), exist_ok=True)
# create name (with path) of output:
output_training = os.path.join("output","training","out.csv")

# prepare input of model fitting:
X,y = prepare_reg_input(df,params)
# fit model:
model = LinearRegression().fit(X,y)

# save the model in the model folder
filename = 'output/model/trained_model.sav'
pickle.dump(model, open(filename, 'wb'))


with io.open(output_training,"w",encoding="utf8") as fd_out_training:
    fd_out_training.write(str(mean))




