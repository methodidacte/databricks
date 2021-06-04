# Databricks notebook source
from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

model_name = "spark-lr-model"

# COMMAND ----------

# Delete a registered model along with all its versions
client.delete_registered_model(name=model_name)

# COMMAND ----------

versions=[1, 2, 3]
for version in versions:
    client.delete_model_version(name=model_name, version=version)

# COMMAND ----------


