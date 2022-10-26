from joblib import load
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import pickle
import yaml

# from training import prepare_reg_input

def prepare_reg_input(df,params):
    """"prepare input for sklearn linear regression fit"""
    x1 = df.x1.values
    x2 = df.x2.values
    x1_p = x1**params['p1']
    x2_p = x2**params['p2']

    X = np.vstack((x1, x2, x1_p, x2_p)).T
    y = df.y.values.reshape(-1,1)
    return X,y


# import the filtered dataframe from the filter stage:
df = pd.read_csv("output/filter/out.csv")

# get model path
path_model = "output/model/trained_model.sav"
# load model from training stage:
trained_model = pickle.load(open(path_model, 'rb'))

# *** predict ***
# get the model parameters:
params = yaml.safe_load(open("parameters/params.yaml"))['training']
X,y = prepare_reg_input(df,params)
y_pred = trained_model.predict(X)
# compute root-mean-square-error
rmse = np.sqrt(mean_squared_error(y_pred,y))
R_sqr = trained_model.score(X,y)
print("RMSE is {}".format(rmse))

print(type(R_sqr))
results = {
    "RMSE": rmse,
    "R_sqr": R_sqr
}

# Serializing json
json_object = json.dumps(results, indent=4)
 
# Writing to sample.json
with open("output/metrics/sample.json", "w") as outfile:
    outfile.write(json_object)

