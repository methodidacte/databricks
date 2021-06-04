// Databricks notebook source
// MAGIC %fs ls /mnt/nycitibike/spark-model/sparkml/

// COMMAND ----------

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

// COMMAND ----------

val sameModel = PipelineModel.load("/mnt/nycitibike/spark-model/sparkml/")

// COMMAND ----------

val test = spark.sql("SELECT start_station_id, end_station_id, bike_id, gender FROM tab_nycitibike LIMIT 3")

// COMMAND ----------

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType

// Convert String to Integer Type
val test2 = test.withColumn("gender",col("gender").cast(IntegerType))

// COMMAND ----------

display(test)

// COMMAND ----------

// Prepare test documents, which are unlabeled (id, text) tuples.
val test = spark.createDataFrame(Seq(
  (4L, "spark i j k"),
  (5L, "l m n"),
  (6L, "spark hadoop spark"),
  (7L, "apache hadoop")
)).toDF("id", "text")

// COMMAND ----------

// Make predictions on test documents.
model.transform(test)
  .select("id", "text", "probability", "prediction")
  .collect()
  .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
  }

// COMMAND ----------

// create features vector
val feature_columns = Array("start_station_id", "end_station_id", "bike_id", "gender")

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler().setInputCols(feature_columns).setOutputCol("features")

val df_assembler = assembler.transform(test2)

// COMMAND ----------

import org.apache.spark.ml.feature.VectorIndexer

val indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(3)

val featureIndexer = indexer.fit(df_assembler)

// COMMAND ----------

sameModel.transform(featureIndexer)
