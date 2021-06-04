# Databricks notebook source
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, LinearSVC

%matplotlib inline

# COMMAND ----------

# MAGIC %fs ls file:/databricks/driver/duration_ridge_model.pkl

# COMMAND ----------

# MAGIC %md Déploiement avec Azure Machine Learning Service

# COMMAND ----------

# MAGIC %sh pip install azureml-sdk[databricks]

# COMMAND ----------

from azureml.core import Workspace

# COMMAND ----------

# voir le contenu du fichier config.json
subscription_id = 'f80606e5-788f-4dc3-a9ea-2eb9a7836082'
resource_group = 'rg-synapse-training'
workspace_name = 'mlworkspace-training'

# COMMAND ----------

# se connecter à la ressource Azure Machine Learning Service
try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()
    print("Workspace configuration succeeded. Skip the workspace creation steps below")
except:
    print("Workspace not accessible. Change your parameters or create a new workspace below")

# COMMAND ----------

ws.get_details()

# COMMAND ----------

from azureml.core.model import Model

model = Model.register(model_path = "linmodel.pkl",
                       model_name = "linmodel",
                       tags = {"key": "0.1"},
                       description = "linear regression",
                       workspace = ws)

# COMMAND ----------

# recharger un modèle déjà enregistré
from azureml.core.model import Model

model = Model(ws,name = "linmodel" )

# COMMAND ----------

from azureml.core import Run

# COMMAND ----------

# MAGIC %%writefile score.py
# MAGIC 
# MAGIC import pickle
# MAGIC import json
# MAGIC import numpy as np
# MAGIC import joblib
# MAGIC #from sklearn.externals import joblib
# MAGIC from sklearn.linear_model import Ridge
# MAGIC from azureml.core.model import Model
# MAGIC 
# MAGIC 
# MAGIC def init():
# MAGIC     global model
# MAGIC     # note here "best_model" is the name of the model registered under the workspace
# MAGIC     # this call should return the path to the model.pkl file on the local disk.
# MAGIC     model_path = Model.get_model_path(model_name='linmodel')
# MAGIC     # deserialize the model file back into a sklearn model
# MAGIC     model = joblib.load(model_path)
# MAGIC 
# MAGIC 
# MAGIC # note you can pass in multiple rows for scoring
# MAGIC def run(raw_data):
# MAGIC     try:
# MAGIC         data = json.loads(raw_data)['data']
# MAGIC         data = np.array(data)
# MAGIC         result = model.predict(data)
# MAGIC 
# MAGIC         # you can return any data type as long as it is JSON-serializable
# MAGIC         return result.tolist()
# MAGIC     except Exception as e:
# MAGIC         result = str(e)
# MAGIC         return result

# COMMAND ----------

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

myenv = Environment('CitibikeNY-deployment-env')
myenv.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'pip==20.1.1'
    'azureml-defaults',
    'inference-schema[numpy-support]',
    'joblib',
    'numpy',
    'sklearn'
])

with open('mydbxenv.yml','w') as f:
  f.write(myenv.python.conda_dependencies.serialize_to_string())

# COMMAND ----------

from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(entry_script='score.py', environment=myenv)

# COMMAND ----------

from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(
          cpu_cores = 1,     
          memory_gb = 1,
          auth_enabled=False, #passer à True pour enclencher la sécurité par clés
          tags = {"data": "New York citibike", "method": "sklearn regression"},
          description = "Trip duration regression")

# COMMAND ----------

from azureml.core import Webservice
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException

service_name = 'dbx-tripduration-service'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aciconfig)

service.wait_for_deployment(show_output=True)

# COMMAND ----------

from azureml.core.image import ContainerImage

# Image configuration 
# Ref : https://docs.microsoft.com/fr-fr/python/api/azureml-core/azureml.core.image.containerimage?view=azure-ml-py
image_config = ContainerImage.image_configuration(
                   execution_script = "score.py", 
                   runtime = "python",
                   conda_file = "mydbxenv.yml",
                   tags = {"data": "New York citibike", "method": "sklearn regression"},
                   description = "New York citibike trip duration")

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # Register the image from the image configuration
# MAGIC image = ContainerImage.create(name = "dbx-tripduration-image", 
# MAGIC                               models = [model], # the model object
# MAGIC                               image_config = image_config,
# MAGIC                               workspace = ws)
# MAGIC image.wait_for_creation(show_output=True)

# COMMAND ----------

from azureml.core import Webservice
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException

service_name = 'dbx-tripduration-service'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass
  
service = Webservice.deploy_from_image(
              deployment_config = aciconfig,
              image = image,
              name = service_name,
              workspace = ws)

service.wait_for_deployment(True)

# COMMAND ----------

import time

for i in range(20):
  if(service.state == "Transitioning"):
    print(service.state)
    time.sleep(30)
  else:
    print(service.state)
    break

# COMMAND ----------

print(service.get_logs())

# COMMAND ----------

print(service.location)

# COMMAND ----------

print(service.scoring_uri)

# COMMAND ----------

print(service.swagger_uri)

# COMMAND ----------

print(service.get_keys()) # auth_enabled=True

# COMMAND ----------

# recharger le service
from azureml.core import Webservice

service_name = 'dbx-tripduration-service'
service = Webservice(ws, service_name)

# COMMAND ----------

import json

input_payload = json.dumps({ 
    'data': [
        [1.129146933555603, 1.0, 1.0, 0.0],
        [1.129146933555603, 1.0, 1.0, 0.0]
    ],
    'method': 'predict'  # If you have a classification model, you can get probabilities by changing this to 'predict_proba'.
})

output = service.run(input_payload)

print(output)

# COMMAND ----------

# send a random row from the test set to score
random_index = np.random.randint(0, len(X_test)-1)
print(random_index)
input_data = "{\"data\": [" + str(list(X_test.iloc[random_index])) + "]}"
print(input_data)

# COMMAND ----------

# request the web service

import requests

headers = {'Content-Type': 'application/json'}

# for AKS deployment you'd need to the service key in the header as well
# api_key = service.get_key()
# headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_key)} 

resp = requests.post(service.scoring_uri, input_data, headers=headers)

print("POST to url", service.scoring_uri)
print("input data:", input_data)
#print("label:", y_test[random_index])
print("prediction:", resp.text)
print("prediction:", resp)
