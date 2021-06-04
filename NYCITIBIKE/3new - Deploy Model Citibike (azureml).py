# Databricks notebook source
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

%matplotlib inline

# COMMAND ----------

# MAGIC %fs ls /mnt/nycitibike/MODELS/

# COMMAND ----------

# MAGIC %md Déploiement avec Azure Machine Learning Service

# COMMAND ----------

# MAGIC %sh pip install azureml-sdk[databricks]

# COMMAND ----------

from azureml.core import Workspace

# voir le contenu du fichier config.json
subscription_id = 'f80606e5-788f-4dc3-a9ea-2eb9a7836082'
resource_group = 'rg-synapse-training'
workspace_name = 'mlworkspace-training'

# COMMAND ----------

# se connecter à la ressource Azure Machine Learning Service
# Performing interactive authentication.

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

model = Model.register(model_path = "/dbfs/mnt/nycitibike/MODELS/tripduration_ridge_model.pkl",
                       model_name = "ridge-onazureml-model",
                       tags = {"key": "0.1"},
                       description = "Trip duration ridge regression",
                       workspace = ws)

# COMMAND ----------

# recharger un modèle déjà enregistré sur le portail Azure ML
from azureml.core.model import Model

model = Model(ws, name = "ridge-onazureml-model" )

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
# MAGIC     model_path = Model.get_model_path(model_name='ridge-onazureml-model')
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
# MAGIC       
# MAGIC     except Exception as e:
# MAGIC         result = str(e)
# MAGIC         return result

# COMMAND ----------

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

nycenv = Environment('nycitibike-deployment-env')
nycenv.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'pip==20.1.1',
    'azureml-defaults',
    'inference-schema[numpy-support]',
    'joblib',
    'numpy',
    'sklearn'
])

with open('nycenv.yml','w') as f:
  f.write(nycenv.python.conda_dependencies.serialize_to_string())

# COMMAND ----------

from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(entry_script='score.py', environment=nycenv)

# COMMAND ----------

from azureml.core.webservice import AciWebservice

aci_config = AciWebservice.deploy_configuration(
          cpu_cores = 1,     
          memory_gb = 1,
          auth_enabled=False, #passer à True pour enclencher la sécurité par clés
          tags = {"data": "New York citibike", "method": "sklearn ridge regression"},
          description = "Trip duration regression")

# COMMAND ----------

from azureml.core import Webservice
from azureml.exceptions import WebserviceException


service_name = 'tripduration-service'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config)

service.wait_for_deployment(show_output=True)

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

service_name = 'tripduration-service'
service = Webservice(ws, service_name)

# COMMAND ----------

import json

input_payload = json.dumps({ 
    'data': [
        [3621, 3094, 37793, 1991, 1],
        [3081, 3048, 26396, 1969, 0]
    ],
    'method': 'predict'  # If you have a classification model, you can get probabilities by changing this to 'predict_proba'.
})

output = service.run(input_payload)

print(output)

# COMMAND ----------

# recharger un modèle déjà enregistré sur le portail Azure ML
from azureml.core.model import Model

model = Model(ws, name = "ridge-onazureml-model" )
# deserialize the model file back into a sklearn model
#model = joblib.load(model_path)

raw_data = json.dumps({ 
    'data': [
        [3621, 3094, 37793, 1991, 1],
        [3081, 3048, 26396, 1969, 0]
    ]
})

# COMMAND ----------

X_test = spark.sql("SELECT start_station_id, end_station_id, bike_id, birth_year, CAST(gender AS int) FROM tab_nycitibike LIMIT 100").toPandas()

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

# COMMAND ----------

import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

#Request data goes here
data = { 
    'data': [
        [3621, 3094, 37793, 1991, 1],
        [3081, 3048, 26396, 1969, 0]
    ]
}

body = str.encode(json.dumps(data))

url = 'http://90743893-ed37-4029-a4fc-ea7680a9a3f2.westeurope.azurecontainer.io/score'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))


# COMMAND ----------


