# Databricks notebook source
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model
from sklearn.feature_selection import RFE
#from sklearn.svm import SVR, LinearSVC

%matplotlib inline

# COMMAND ----------

#dbutils.library APIs are deprecated and will be removed in a future DBR release. You can use %pip and %conda commands to install notebook scoped python libraries. For more information see https://docs.microsoft.com/azure/databricks/libraries/notebooks-python-libraries.
#dbutils.library.installPyPI("mlflow", extras="extras")
#dbutils.library.installPyPI("joblib")

# COMMAND ----------

df = spark.sql("SELECT * FROM tab_nycitibike")

# COMMAND ----------

dataset = df.select("*").sample(0.01).toPandas()
dataset.count()

# COMMAND ----------

dataset.isnull().sum()

# COMMAND ----------

dataset = dataset.fillna(method='ffill')
dataset.describe()

# COMMAND ----------

dataset['gender'] = pd.to_numeric(dataset['gender'])

# COMMAND ----------

X = dataset[[x for x in dataset.columns if x!='trip_duration']]
y = dataset['trip_duration']

# COMMAND ----------

X = dataset[['start_station_id','end_station_id','bike_id','birth_year','gender']]
y = dataset['trip_duration']

# COMMAND ----------

# split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC Linear Regression

# COMMAND ----------

# fit (train)
from sklearn import linear_model

linmodel = linear_model.LinearRegression(fit_intercept=True, normalize=True)
linmodel.fit(X_train, y_train)

# COMMAND ----------

# predict
y_predict = linmodel.predict(X_test)
y_predict

# COMMAND ----------

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error

print("MSE : ", mean_squared_error(y_test, y_predict))
print("R2 : ", r2_score(y_test, y_predict))
print("RMSE : ", np.sqrt(mean_squared_error(y_test, y_predict)))
print("MAE : ", median_absolute_error(y_test, y_predict))

# COMMAND ----------

# cross validation 
from sklearn.model_selection import cross_validate

scores = cross_validate(linmodel, X_train, y_train, cv=3, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
print(scores['test_neg_mean_squared_error']) 

# COMMAND ----------

# model export (pickle)
import joblib

joblib.dump(linmodel, open('linmodel.pkl','wb'))

# COMMAND ----------

# scatter plot des valeurs réelles vs prédites

# Display results
fig = plt.figure(1)

plt.scatter(X_test['gender'], y_test,  color='black')
plt.plot(X_test['gender'], y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

#plt.show()

#display(fig)

# Save figure
fig.savefig("obs_vs_predict.png")

# Close plot
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC Ridge Regression

# COMMAND ----------

dbutils.fs.rm('/mnt/nycitibike/MODELS/', recurse=True)

# COMMAND ----------

dbutils.fs.mkdirs('/mnt/nycitibike/MODELS/')

# COMMAND ----------

# MAGIC %fs ls /mnt/nycitibike/MODELS/

# COMMAND ----------

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

alpha = 1.0
normalize = True

with mlflow.start_run():
  
  ridgemodel = Ridge(alpha=1.0, normalize=True)
  ridgemodel.fit(X_train, y_train)
  y_predict = ridgemodel.predict(X_test)
  
  mse = mean_squared_error(y_test,y_predict)
  r2 = r2_score(y_test,y_predict)
  rmse = np.sqrt(mean_squared_error(y_test,y_predict))
  mae = median_absolute_error(y_test,y_predict)

  # Log mlflow attributes for mlflow UI
  mlflow.log_param("alpha", alpha)
  mlflow.log_param("normalize", normalize)
  mlflow.log_metric("mse", mse)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)

  registered_model_name = "ridge-registered-model"
  mlflow.sklearn.log_model(ridgemodel, "ridge-model", registered_model_name = registered_model_name)
  
  #model_path = "/mnt/nycitibike/MODELS/ridge-model-%f-%f" % (alpha, normalize)
  #mlflow.sklearn.save_model(ridgemodel, model_path)
  
  # Log artifacts (output files)
  #mlflow.log_artifact("obs_vs_predict.png")
  
  run = mlflow.active_run()
  print("Active run_id: {}".format(run.info.run_id))


# COMMAND ----------

# Register model
## NE PAS EXECUTER (fait par la cellule ci-dessus)
#https://www.mlflow.org/docs/latest/model-registry.html#registering-a-model

client = MlflowClient()
result = client.create_model_version( \
                                     name="ridge-registered-model", \
                                     source="dbfs:/databricks/mlflow-tracking/1051022099699898/4a4cd9fb4a034066a4a635eedd8b5a65/artifacts/ridge-model", \
                                     run_id="2ca9694883284c55948d49e7254f6a23" \
                                    )


# COMMAND ----------

# hyperparameters tuning (grid search)
from sklearn.model_selection import GridSearchCV

dico_param = {'alpha': [1e-3, 1e-2, 1e-1, 1]}
search_hyperp_ridge = GridSearchCV(Ridge(), dico_param, scoring='neg_mean_squared_error', cv = 5)
search_hyperp_ridge.fit(X_train, X_train)
search_hyperp_ridge.predict(X_test)

print(search_hyperp_ridge.best_params_)
print(search_hyperp_ridge.best_score_)

# COMMAND ----------

# model export (pickle)
import joblib

joblib.dump(ridgemodel, open('tripduration_ridge_model.pkl','wb'))
#joblib.dump(search_hyperp_ridge, open('tripduration_hyper_model.pkl','wb'))

#Par défaut, l'enregistrement se fait sur le driver

# COMMAND ----------

# unpickle and test
my_pickle_model = joblib.load('tripduration_ridge_model.pkl')
my_pickle_model.predict(X_test)

# COMMAND ----------

# MAGIC %fs ls file:/databricks/driver/tripduration_ridge_model.pkl

# COMMAND ----------

# MAGIC %fs cp file:/databricks/driver/tripduration_ridge_model.pkl /mnt/nycitibike/MODELS/

# COMMAND ----------

# MAGIC %fs ls /mnt/nycitibike/MODELS/

# COMMAND ----------

import mlflow
logged_model = 'runs:/2ca9694883284c55948d49e7254f6a23/ridge-model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(X_test)

# COMMAND ----------


