import pandas as pd
import yaml
import sys
import os
import io

# load filtration stage parameters:
params = yaml.safe_load(open("params.yaml"))['filter']
# load dataset:
df = pd.read_csv("regression-data/regression_data.csv")
# get all rows where y is greater than the filter parameter:
df = df[df['y']>params['value']]
print(df)
# create the directory where the training results are stored
dir_name = os.path.join("output","filter")
os.makedirs(dir_name, exist_ok=True)
# write filtered dataframe to csv file named "out" in the directory just created:
df.to_csv("output/filter/out.csv")



