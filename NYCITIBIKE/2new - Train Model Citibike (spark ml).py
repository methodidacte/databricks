# Databricks notebook source
df = spark.sql("SELECT * FROM tab_nycitibike")

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.sql.types import IntegerType

df = df.withColumn("gender", df["gender"].cast(IntegerType()))

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
#from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# COMMAND ----------

# create features vector
feature_columns = ['start_station_id', 'end_station_id', 'bike_id', 'gender']

# COMMAND ----------

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembler = assembler.transform(df)

# COMMAND ----------

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 3 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=3).fit(df_assembler)

# COMMAND ----------

trainingData, testData = df_assembler.randomSplit([0.7, 0.3])

trainingData.cache()
testData.cache()

# COMMAND ----------

# Train a LinearRegression model.
lr = LinearRegression(featuresCol="features", labelCol="trip_duration")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, lr])

# COMMAND ----------

grid = ParamGridBuilder() \
  .addGrid(lr.maxIter, [1]) \
  .addGrid(lr.regParam, [1]) \
  .addGrid(lr.elasticNetParam, [1]) \
  .build()

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="rmse")

# COMMAND ----------

tuning = TrainValidationSplit(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=grid)

# COMMAND ----------

import mlflow
#from mlflow import spark
#import mlflow.mleap #fonctionne avec Spark 3.0 ?
#import mlflow.pyfunc
#import mleap.pyspark

# COMMAND ----------

with mlflow.start_run(run_name='TripDuration_lr'):

  tunedModel = tuning.fit(trainingData)

  # We log a custom tag, a custom metric, and the best model to the main run.
  mlflow.set_tag('Citibike_training', 'Data_team')
  
  rmse = evaluator.evaluate(tunedModel.transform(testData), {evaluator.metricName: "rmse"})
  r2 = evaluator.evaluate(tunedModel.transform(testData), {evaluator.metricName: "r2"})
  mae = evaluator.evaluate(tunedModel.transform(testData), {evaluator.metricName: "mae"})
  mse = evaluator.evaluate(tunedModel.transform(testData), {evaluator.metricName: "mse"})
  
  mlflow.log_metric('rmse', rmse)
  mlflow.log_metric('r2', r2)
  mlflow.log_metric('mae', mae)
  mlflow.log_metric('mse', mse)
  
  print("Tuned model r2: {}".format(r2))
  print("Tuned model rmse: {}".format(rmse))
  print("Tuned model mae: {}".format(mae))
  print("Tuned model mse: {}".format(mse))
  
  # Log the model within the MLflow run
  # https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html
  # https://www.mlflow.org/docs/latest/python_api/mlflow.pyspark.ml.html
  
  mlflow.mleap.log_model(spark_model=tunedModel.bestModel, sample_input=trainingData.limit(1), artifact_path="spark-model-mleap")
  
  mlflow.pyfunc.log_model(tunedModel.bestModel, "spark-model-pyfunc")
  
  # We log other artifacts
  
  
  model_path = "/dbfs/mnt/nycitibike/spark-model"
  mlflow.spark.save_model(tunedModel.bestModel, model_path)
  
  run = mlflow.active_run()
  print("Active run_id: {}".format(run.info.run_id))
  
  mlflow.end_run()

# COMMAND ----------

model = tunedModel.bestModel

# COMMAND ----------

# MAGIC %fs ls /mnt/nycitibike/spark-model

# COMMAND ----------

# MAGIC %md https://docs.microsoft.com/fr-fr/azure/databricks/_static/notebooks/mleap-model-export-demo-python.html

# COMMAND ----------

sparkTransformed = model.transform(testData)
display(sparkTransformed)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md Serialize to bundle and deserialize

# COMMAND ----------

import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer

model.serializeToBundle("jar:file:/dbfs/mnt/nycitibike/spark-model/lr-spark-model.zip", sparkTransformed)

# COMMAND ----------

from pyspark.ml import PipelineModel

deserializedPipeline = PipelineModel.deserializeFromBundle("jar:file:/dbfs/mnt/nycitibike/spark-model/lr-spark-model.zip")

# COMMAND ----------

test_df = testData.limit(10)

# COMMAND ----------

exampleResults = deserializedPipeline.transform(test_df)
display(exampleResults)

# COMMAND ----------

# MAGIC %md Register model
# MAGIC 
# MAGIC https://www.mlflow.org/docs/latest/model-registry.html#registering-a-model
# MAGIC https://docs.microsoft.com/fr-fr/azure/databricks/applications/machine-learning/manage-model-lifecycle/

# COMMAND ----------

result=mlflow.register_model("runs:<model-path>", "<model-name>")

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
name = "spark-lr-registered-model"
client.create_registered_model(name)

desc = "A new version of the model"
model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
mv = client.create_model_version(name, model_uri, run.info.run_id, description=desc)

# COMMAND ----------

# MAGIC %md mlflow.pyfunc

# COMMAND ----------

import mlflow.pyfunc

mlflow.pyfunc.log_model(tunedModel.bestModel, "spark-model")
