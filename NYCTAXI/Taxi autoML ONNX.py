# Databricks notebook source
import azureml.core

from azureml.core import Experiment, Workspace, Dataset, Datastore
from azureml.train.automl import AutoMLConfig
from azureml.data.dataset_factory import TabularDatasetFactory

# COMMAND ----------

subscription_id = "f80606e5-788f-4dc3-a9ea-2eb9a7836082"
resource_group = "rg-synapse-training"
workspace_name = "mlworkspace-training"
experiment_name = "satraining-nyc_taxi-20210525085218"

ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
experiment = Experiment(ws, experiment_name)

# COMMAND ----------

df = spark.sql("SELECT * FROM default.nyc_taxi")

datastore = Datastore.get_default(ws)
dataset = TabularDatasetFactory.register_spark_dataframe(df, datastore, name = experiment_name + "-dataset")

# COMMAND ----------

automl_config = AutoMLConfig(spark_context = sc,
                             task = "regression",
                             training_data = dataset,
                             label_column_name = "tripDistance",
                             primary_metric = "normalized_root_mean_squared_error",
                             experiment_timeout_hours = 1,
                             max_concurrent_iterations = 4,
                             enable_onnx_compatible_models = True)

# COMMAND ----------

run = experiment.submit(automl_config)

# COMMAND ----------

displayHTML("<a href={} target='_blank'>Your experiment in Azure Machine Learning portal: {}</a>".format(run.get_portal_url(), run.id))

# COMMAND ----------

run.wait_for_completion()

import onnxruntime
import mlflow
import mlflow.onnx

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema

# Get best model from automl run
best_run, onnx_model = run.get_output(return_onnx_model=True)

# Define utility functions to infer the schema of ONNX model
def _infer_schema(data):
    res = []
    for _, col in enumerate(data):
        t = col.type.replace("tensor(", "").replace(")", "")
        if t in ["bool"]:
            dt = DataType.boolean
        elif t in ["int8", "uint8", "int16", "uint16", "int32"]:
            dt = DateType.integer
        elif t in ["uint32", "int64"]:
            dt = DataType.long
        elif t in ["float16", "bfloat16", "float"]:
            dt = DataType.float
        elif t in ["double"]:
            dt = DataType.double
        elif t in ["string"]:
            dt = DataType.string
        else:
            raise Exception("Unsupported type: " + t)
        res.append(ColSpec(type=dt, name=col.name))
    return Schema(res)

def _infer_signature(onnx_model):
    onnx_model_bytes = onnx_model.SerializeToString()
    onnx_runtime = onnxruntime.InferenceSession(onnx_model_bytes)
    inputs = _infer_schema(onnx_runtime.get_inputs())
    outputs = _infer_schema(onnx_runtime.get_outputs())
    return ModelSignature(inputs, outputs)

# Infer signature of ONNX model
signature = _infer_signature(onnx_model)

artifact_path = experiment_name + "_artifact"
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    # Save the model to the outputs directory for capture
    mlflow.onnx.log_model(onnx_model, artifact_path, signature=signature)

    # Register the model to AML model registry
    mlflow.register_model("runs:/" + run.info.run_id + "/" + artifact_path, "satraining-nyc_taxi-20210525085218-Best")

# COMMAND ----------


