# Databricks notebook source
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
#from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

import mlflow
from mlflow import spark
import mlflow.mleap #fonctionne avec Spark 3.0 ?
#import mlflow.pyfunc
import mleap.pyspark

# COMMAND ----------

#df = spark.sql("SELECT trip_duration,start_station_id,birth_year,unknown_gender,male_gender,female_gender,Subscriber,Customer,real_distance,((real_distance / trip_duration)* 3.6) as vitesse, DATE(start_time) as date,HOUR(start_time) as hour FROM CitibikeNY NATURAL JOIN citybike_station_distance")

df = spark.sql("SELECT trip_duration,start_station_id,birth_year,unknown_gender,male_gender,female_gender,Subscriber,Customer,distance_bwn_stations,(((distance_bwn_stations * 1000) / trip_duration)* 3.6) as vitesse, DATE(start_time) as date,HOUR(start_time) as hour FROM CitibikeNY2 NATURAL JOIN citybike_station_distance")

# COMMAND ----------

df = df.filter((df.vitesse>13) & (df.vitesse<32))

# COMMAND ----------

df = spark.sql("SELECT * FROM tab_nycitibike")

# COMMAND ----------

display(df)

# COMMAND ----------

df2=spark.sql("SELECT DATE(Date) as date,Day,Day_Name,Day_of_month,Day_of_week,Month,Month_Name,Month_Number,Quarter_Number,Week_of_month,Year_Month,Year FROM calandar_2013_2020")

# COMMAND ----------

df3=df.join(df2,["date"],"left")

# COMMAND ----------

#df = spark.sql("select * from CitibikeNY")
df = spark.sql("SELECT trip_duration,start_station_id,end_station_id,birth_year,unknown_gender,male_gender,female_gender,Subscriber,Customer,distance_bwn_stations,real_distance FROM CitibikeNY NATURAL JOIN citybike_station_distance")

# COMMAND ----------

df= df3.drop("date","Day","Day_Name","Month","Month_Name","Year_Month","vitesse")

# COMMAND ----------

# create features vector
feature_columns = ['start_station_id', 'end_station_id', 'bike_id', 'usertype', 'gender']

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembler = assembler.transform(df)

# COMMAND ----------

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 3 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=3).fit(df_assembler)

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

with mlflow.start_run(run_name='Linear_Regression'):

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
  mlflow.pyfunc.log_model(tunedModel.bestModel, "spark-model")
  #mlflow.mleap.log_model(spark_model=tunedModel.bestModel, sample_input=df.limit(1), artifact_path="model")
  
  # We log other artifacts
  
  
  model_path = "/dbfs/mnt/nycitibike/spark-model"
  mlflow.spark.save_model(tunedModel.bestModel, model_path)
  
  run = mlflow.active_run()
  print("Active run_id: {}".format(run.info.run_id))
  
  mlflow.end_run()

# COMMAND ----------

# MAGIC %md https://docs.databricks.com/applications/mlflow/models.html

# COMMAND ----------

# MAGIC %fs mkdirs /mnt/nycitibike/spark-model/

# COMMAND ----------

#Save model to dbfs
from mlflow import spark

model_path = "/dbfs/mnt/nycitibike/spark-model"
mlflow.spark.save_model(tunedModel.bestModel, "/dbfs/FileStore/spark-model")

# COMMAND ----------

# MAGIC %fs ls /FileStore/spark-model/sparkml

# COMMAND ----------

# MAGIC %md https://adatis.co.uk/mlflow-introduction-to-model-registry/

# COMMAND ----------

#Save model to Model registry

model_name = "lr_trip_duration_model"
run_id = run.info.run_id #"c5218874277e4644a6536affee9b3ba0"
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

#https://www.mlflow.org/docs/latest/model-registry.html
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.create_registered_model("spark-lr-model")
#client.log_artifacts(run.info.run_id, "/FileStore/spark-model", artifact_path=mlflow.get_artifact_uri())

# COMMAND ----------

artifacts = [f.path for f in client.list_artifacts(run.info.run_id, "model")]

print("artifacts: {}".format(artifacts))

# COMMAND ----------

mlflow.get_artifact_uri()

# COMMAND ----------

# MAGIC %fs ls /databricks/mlflow-tracking/

# COMMAND ----------


