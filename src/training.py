import pandas as pd
import yaml
import os
import io
import sys

# get the parameter for the training stage:
params = yaml.safe_load(open("parameters/params.yaml"))['training'] # training is the name of the workspace variable
# get the path to the data. Note: sys.argv contains a list of the command line arguments used when running a commands
data_path = sys.argv[1]+"/out.csv"
# read output from the filter stage:
df = pd.read_csv(data_path)
# get the number of rows specified by the num_rows parameter:
df = df.head(params['num_rows'])
print("taking the mean of:")
print(df)
# get the mean of the y-column
mean = df['y'].mean()
print(mean)

# create directory where output of this stage is stored:
os.makedirs(os.path.join("output","training"), exist_ok=True)
# create name (with path) of output:
output_training = os.path.join("output","training","out.csv")

with io.open(output_training,"w",encoding="utf8") as fd_out_training:
    fd_out_training.write(str(mean))




