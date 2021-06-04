# Databricks notebook source
# MAGIC %sh pip list | grep azure-mgmt-containerregistry

# COMMAND ----------

import azureml.core

from azureml.core import Experiment, Workspace, Dataset, Datastore
from azureml.train.automl import AutoMLConfig
from azureml.data.dataset_factory import TabularDatasetFactory

# COMMAND ----------

subscription_id = "f80606e5-788f-4dc3-a9ea-2eb9a7836082"
resource_group = "rg-synapse-training"
workspace_name = "mlworkspace-training"
experiment_name = "satraining-nyc_taxi-20210525085738"

ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
experiment = Experiment(ws, experiment_name)

# COMMAND ----------

df = spark.sql("SELECT * FROM tab_nyctaxi")

datastore = Datastore.get_default(ws)
dataset = TabularDatasetFactory.register_spark_dataframe(df, datastore, name = experiment_name + "-dataset")

# COMMAND ----------

dataset =  Dataset.get_by_name(ws, name='satraining-nyc_taxi-20210525085738-dataset')
#dataset.to_pandas_dataframe()

# COMMAND ----------

# MAGIC %md
# MAGIC Class SynapseCompute: This is an experimental class, and may change at any time. Please see https://aka.ms/azuremlexperimental for more information.
# MAGIC 
# MAGIC 'linksyn-spark': id: /subscriptions/f80606e5-788f-4dc3-a9ea-2eb9a7836082/resourceGroups/rg-synapse-training/providers/Microsoft.MachineLearningServices/workspaces/mlworkspace-training/computes/linksyn-spark,
# MAGIC  name: linksyn-spark,
# MAGIC  tags: None,
# MAGIC  location: westeurope,
# MAGIC  properties: {'computeType': 'SynapseSpark', 'computeLocation': 'westeurope', 'description': None, 'resourceId': '/subscriptions/f80606e5-788f-4dc3-a9ea-2eb9a7836082/resourceGroups/rg-synapse-training/providers/Microsoft.Synapse/workspaces/satraining/bigDataPools/sparkpooltrain', 'provisioningErrors': None, 'provisioningState': 'Succeeded', 'properties': {'sparkVersion': '2.4', 'nodeCount': 12, 'nodeSizeFamily': 'MemoryOptimized', 'nodeSize': 'Small', 'autoScaleProperties': {'enabled': False, 'maxNodeCount': 10, 'minNodeCount': 3}, 'autoPauseProperties': {'enabled': True, 'delayInMinutes': 15}C}}

# COMMAND ----------

from azureml.core import LinkedService, SynapseWorkspaceLinkedServiceConfiguration

for service in LinkedService.list(ws) : 
    print(f"Service: {service}")

# Retrieve a known linked service
linked_service = LinkedService.get(ws, 'ls_synapsetraining')

# COMMAND ----------

from azureml.core.compute import SynapseCompute, ComputeTarget

attach_config = SynapseCompute.attach_configuration(
        linked_service = linked_service,
        type="SynapseSpark",
        pool_name="sparkpooltrain") # This name comes from your Synapse workspace

synapse_compute=ComputeTarget.attach(
        workspace=ws,
        name='linksyn-spark',
        attach_configuration=attach_config)

synapse_compute.wait_for_completion()

#'SynapseCompute' object has no attribute 'vm_size'

# COMMAND ----------

# MAGIC %md ### environnement
# MAGIC "CondaDependencies": {
# MAGIC           "name": "project_environment",
# MAGIC           "dependencies": [
# MAGIC             "python=3.6.2",
# MAGIC             {
# MAGIC               "pip": [
# MAGIC                 "azureml-train-automl==1.29.0.*",
# MAGIC                 "inference-schema"
# MAGIC               ]
# MAGIC             },
# MAGIC             "pandas==0.25.1",
# MAGIC             "psutil>5.0.0,<6.0.0",
# MAGIC             "scikit-learn==0.22.1",
# MAGIC             "numpy~=1.18.0",
# MAGIC             "py-xgboost<=0.90",
# MAGIC             "fbprophet==0.5",
# MAGIC             "setuptools-git"
# MAGIC           ],
# MAGIC           "channels": [
# MAGIC             "anaconda",
# MAGIC             "conda-forge",
# MAGIC             "pytorch"
# MAGIC           ]
# MAGIC         }

# COMMAND ----------

from azureml.core.runconfig import RunConfiguration

aml_run_config = RunConfiguration()

# Use just-specified compute target ("cpu-cluster")
aml_run_config.target = synapse_compute

# COMMAND ----------

automl_config = AutoMLConfig(spark_context = sc,
                             task = "regression",
                             training_data = dataset,
                             label_column_name = "tripDistance",
                             primary_metric = "spearman_correlation",
                             #runconfig=aml_run_config,
                             #compute_target = synapse_compute,
                             experiment_timeout_hours = 1,
                             max_concurrent_iterations = 4,
                             enable_onnx_compatible_models = False)

# COMMAND ----------

import azureml.train.automl.runtime

run = experiment.submit(automl_config, show_output=True)
                        
#No module named 'azureml.train.automl.runtime'
#No run_configuration provided, running on local with default configuration
## It is strongly recommended that you pass in a value for the "algorithms" argument when calling decode(). This argument will be mandatory in a future version.


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Message: Module 'pandas.compat' is required in the current environment for running Remote or Local (in-process) runs. Please install this dependency (e.g. `pip install pandas.compat`) or provide a RunConfiguration.
# MAGIC 	InnerException: ImportError: cannot import name 'raise_with_traceback' from 'pandas.compat' (/databricks/python/lib/python3.7/site-packages/pandas/compat/__init__.py)
# MAGIC 	ErrorResponse 
# MAGIC {
# MAGIC     "error": {
# MAGIC         "code": "UserError",
# MAGIC         "message": "Module 'pandas.compat' is required in the current environment for running Remote or Local (in-process) runs. Please install this dependency (e.g. `pip install pandas.compat`) or provide a RunConfiguration.",
# MAGIC         "target": "compute_target",
# MAGIC         "inner_error": {
# MAGIC             "code": "NotSupported",
# MAGIC             "inner_error": {
# MAGIC                 "code": "IncompatibleOrMissingDependency"
# MAGIC             }
# MAGIC         }
# MAGIC     }
# MAGIC }

# COMMAND ----------

# MAGIC %md
# MAGIC {
# MAGIC     "error": {
# MAGIC         "code": "UserError",
# MAGIC         "message": "Install the required versions of packages by setting up the environment as described in documentation.\nRequired version/Installed version\nazure-mgmt-containerregistry<=2.8.0/azure-mgmt-containerregistry 8.0.0",
# MAGIC         "details_uri": "https://docs.microsoft.com/azure/machine-learning/how-to-configure-environment?#sdk-for-databricks-with-automated-machine-learning",
# MAGIC         "inner_error": {
# MAGIC             "code": "NotSupported",
# MAGIC             "inner_error": {
# MAGIC                 "code": "IncompatibleOrMissingDependency"
# MAGIC             }
# MAGIC         },
# MAGIC         "reference_code": "baf60d78-c1c2-4c27-9100-623981408905"
# MAGIC     }
# MAGIC }

# COMMAND ----------

# MAGIC %md
# MAGIC ERROR: {
# MAGIC     "additional_properties": {},
# MAGIC     "error": {
# MAGIC         "additional_properties": {
# MAGIC             "debugInfo": null
# MAGIC         },
# MAGIC         "code": "SystemError",
# MAGIC         "severity": null,
# MAGIC         "message": "Encountered an internal AutoML error. Error Message/Code: ServiceException. Additional Info: ServiceException:\n\tMessage: AzureMLAggregatedException:\n\tMessage: UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/env_dependencies.json already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/conda_env_v_1_0_0.yml already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_1_0_0.py already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_2_0_0.py already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/pipeline_graph.json already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/model.pkl already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/internal_cross_validated_models.pkl already exists.\n\tInnerException None\n\tErrorResponse \n{\n    \"error\": {\n        \"message\": \"UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/env_dependencies.json already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/conda_env_v_1_0_0.yml already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_1_0_0.py already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_2_0_0.py already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/pipeline_graph.json already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/model.pkl already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/internal_cross_validated_models.pkl already exists.\"\n    }\n}\n\tInnerException: None\n\tErrorResponse \n{\n    \"error\": {\n        \"message\": \"AzureMLAggregatedException:\\n\\tMessage: UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/env_dependencies.json already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/conda_env_v_1_0_0.yml already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_1_0_0.py already exists.\\nUserError: Resource Con",
# MAGIC         "message_format": "Encountered an internal AutoML error. Error Message/Code: ServiceException. Additional Info: {error_details}",
# MAGIC         "message_parameters": {
# MAGIC             "error_message": "ServiceException",
# MAGIC             "error_details": "ServiceException:\n\tMessage: AzureMLAggregatedException:\n\tMessage: UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/env_dependencies.json already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/conda_env_v_1_0_0.yml already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_1_0_0.py already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_2_0_0.py already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/pipeline_graph.json already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/model.pkl already exists.\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/internal_cross_validated_models.pkl already exists.\n\tInnerException None\n\tErrorResponse \n{\n    \"error\": {\n        \"message\": \"UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/env_dependencies.json already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/conda_env_v_1_0_0.yml already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_1_0_0.py already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_2_0_0.py already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/pipeline_graph.json already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/model.pkl already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/internal_cross_validated_models.pkl already exists.\"\n    }\n}\n\tInnerException: None\n\tErrorResponse \n{\n    \"error\": {\n        \"message\": \"AzureMLAggregatedException:\\n\\tMessage: UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/env_dependencies.json already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/conda_env_v_1_0_0.yml already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_1_0_0.py already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_2_0_0.py already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/pipeline_graph.json already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/model.pkl already exists.\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/internal_cross_validated_models.pkl already exists.\\n\\tInnerException None\\n\\tErrorResponse \\n{\\n    \\\"error\\\": {\\n        \\\"message\\\": \\\"UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/env_dependencies.json already exists.\\\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/conda_env_v_1_0_0.yml already exists.\\\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_1_0_0.py already exists.\\\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/scoring_file_v_2_0_0.py already exists.\\\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/pipeline_graph.json already exists.\\\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/model.pkl already exists.\\\\nUserError: Resource Conflict: ArtifactId ExperimentRun/dcid.AutoML_aff8eb37-4c77-497b-acd8-97a3afb94dd3_63/outputs/internal_cross_validated_models.pkl already exists.\\\"\\n    }\\n}\",\n        \"target\": \"SaveArtifact\",\n        \"reference_code\": \"932614ee-ceb4-48d2-940f-971ad8e2f3d7\"\n    }\n}"
# MAGIC         },
# MAGIC         "reference_code": "932614ee-ceb4-48d2-940f-971ad8e2f3d7",
# MAGIC         "details_uri": "https://docs.microsoft.com/azure/machine-learning/resource-known-issues#automated-machine-learning",
# MAGIC         "target": "SaveArtifact",
# MAGIC         "details": [],
# MAGIC         "inner_error": {
# MAGIC             "additional_properties": {},
# MAGIC             "code": "ClientError",
# MAGIC             "inner_error": {
# MAGIC                 "additional_properties": {},
# MAGIC                 "code": "AutoMLInternal",
# MAGIC                 "inner_error": null
# MAGIC             }
# MAGIC         },
# MAGIC         "additional_info": null
# MAGIC     },
# MAGIC     "correlation": null,
# MAGIC     "environment": null,
# MAGIC     "location": null,
# MAGIC     "time": {},
# MAGIC     "component_name": null
# MAGIC }

# COMMAND ----------

# MAGIC %md
# MAGIC WARNING:urllib3.connectionpool:Retrying (Retry(total=2, connect=2, read=3, redirect=None, status=None)) after connection broken by 'NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x7fdfc5266a10>: Failed to establish a new connection: [Errno 111] Connection refused')': /metric/v2.0/subscriptions/f80606e5-788f-4dc3-a9ea-2eb9a7836082/resourceGroups/rg-synapse-training/providers/Microsoft.MachineLearningServices/workspaces/mlworkspace-training/runs/AutoML_9e4ea685-9b87-415a-89a4-dd223bc6037d_311/full

# COMMAND ----------

from azureml.widgets import RunDetails

RunDetails(run).show()

# COMMAND ----------

displayHTML("<a href={} target='_blank'>Your experiment in Azure Machine Learning portal: {}</a>".format(run.get_portal_url(), run.id))

# COMMAND ----------

!pip install azureml-mlflow

# COMMAND ----------

run.wait_for_completion()

import mlflow

# Get best model from automl run
best_run, non_onnx_model = run.get_output()

artifact_path = experiment_name + "_artifact"

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    # Save the model to the outputs directory for capture
    mlflow.sklearn.log_model(non_onnx_model, artifact_path)

    # Register the model to AML model registry
    mlflow.register_model("runs:/" + run.info.run_id + "/" + artifact_path, "satraining-nyc_taxi-20210525085738-Best")

# COMMAND ----------

# MAGIC %md
# MAGIC Model registry functionality is unavailable; got unsupported URI 'azureml://westeurope.experiments.azureml.net/mlflow/v1.0/subscriptions/f80606e5-788f-4dc3-a9ea-2eb9a7836082/resourceGroups/rg-synapse-training/providers/Microsoft.MachineLearningServices/workspaces/mlworkspace-training?' for model registry data storage. Supported URI schemes are: ['', 'file', 'databricks', 'http', 'https', 'postgresql', 'mysql', 'sqlite', 'mssql']. See https://www.mlflow.org/docs/latest/tracking.html#storage for how to run an MLflow server against one of the supported backend storage locations.

# COMMAND ----------

# Get best model from automl run
best_run, non_onnx_model = run.get_output()

artifact_path = experiment_name + "_artifact"

# register best model
from azureml.core.model import Model

model = best_run.register_model(model_name='nyc_taxi_tip_best_model', model_path=artifact_path)

print(model.name, model.version, sep='\t')

# COMMAND ----------

# MAGIC %md
# MAGIC {
# MAGIC     "error": {
# MAGIC         "message": "Could not locate the provided model_path satraining-nyc_taxi-20210525085738_artifact in the set of files uploaded to the run: ['automl_driver.py', 'azureml-logs/azureml.log', 'outputs/conda_env_v_1_0_0.yml', 'outputs/env_dependencies.json', 'outputs/internal_cross_validated_models.pkl', 'outputs/model.pkl', 'outputs/pipeline_graph.json', 'outputs/scoring_file_v_1_0_0.py', 'outputs/scoring_file_v_2_0_0.py', 'predicted_true', 'residuals']\n                See https://aka.ms/run-logging for more details."
# MAGIC     }
# MAGIC }

# COMMAND ----------


