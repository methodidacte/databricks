# Databricks notebook source
df = spark.sql("SELECT * FROM tab_nycitibike LIMIT 10")

# COMMAND ----------

from pyspark.sql.types import IntegerType

df = df.withColumn("gender", df["gender"].cast(IntegerType()))

# COMMAND ----------

from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler

# create features vector
feature_columns = ['start_station_id', 'end_station_id', 'bike_id', 'gender']

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembler = assembler.transform(df)

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=3).fit(df_assembler)

# COMMAND ----------

import mlflow

logged_model = 'runs:/c58e635fe1f1465dba38a6e2d14554a4/model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# COMMAND ----------

# Predict on a Spark DataFrame.
df.withColumn('predictions', loaded_model(*feature_columns)).collect()

# COMMAND ----------


