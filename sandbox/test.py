
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pickle
import os
current_directory = os.getcwd()
print(current_directory)
df = pd.read_csv("../data/data.csv")

print(df)

# p = 2

# x = df.x1
# one_column = np.ones( (len(x),1) )
# A = np.vstack([one_column.T, x, np.abs(x)**p]).T
# y = df.y
# c = np.linalg.lstsq(A, y)[0]
# y_pred = A @ c.T
# e = y_pred - y

# MSE = mean_squared_error(y, y_pred)
# RMSE = math.sqrt(MSE)

# using sklearn

def prepare_reg_input(df,p1,p2):
    """"prepare input for sklearn linear regression fit"""
    x1 = df.x1.values
    x2 = df.x2.values
    x1_p = x1**p1
    x2_p = x2**p2

    X = np.vstack((x1, x2, x1_p, x2_p)).T
    y = df.y.values.reshape(-1,1)
    return X,y


p1 = 2
p2 = 2
X,y = prepare_reg_input(df,p1,p2)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y_pred,y))
print(rmse)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
R_sqr = loaded_model.score(X, y)
print(R_sqr)






