# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC This notebook shows you how to create and query a table or DataFrame loaded from data stored in Azure Blob storage.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 1: Set the data location and type
# MAGIC 
# MAGIC There are two ways to access Azure Blob storage: account keys and shared access signatures (SAS).
# MAGIC 
# MAGIC To get started, we need to set the location and type of the file.

# COMMAND ----------

storage_account_name = "dlsynapsetraining"
storage_account_access_key = "9UpB5Y81V5Jjid1PrH6ck81L20JDLDtDnEBwAG2CCLjTIcnJb+vI4X60lOgh5RKVL95WOe4xGmJutV83pq3lAQ=="
container = "nycitibike"

# COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://"+container+"@"+storage_account_name+".blob.core.windows.net/",
  mount_point = "/mnt/"+container,
  extra_configs = { "fs.azure.account.key."+storage_account_name+".blob.core.windows.net":  storage_account_access_key })

# COMMAND ----------

dbutils.fs.mounts()

# COMMAND ----------

# MAGIC %fs ls /mnt/nycitibike/golden/parquet/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 2: Read the data
# MAGIC 
# MAGIC Now that we have specified our file metadata, we can create a DataFrame. Notice that we use an *option* to specify that we want to infer the schema from the file. We can also explicitly set this to a particular schema if we have one already.
# MAGIC 
# MAGIC First, let's create a DataFrame in Python.

# COMMAND ----------

file_location = "/mnt/nycitibike/golden/parquet/*/*/*"
file_type = "parquet"

# COMMAND ----------

df = spark.read.format(file_type).option("inferSchema", "true").load(file_location)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3: Query the data
# MAGIC 
# MAGIC Now that we have created our DataFrame, we can query it. For instance, you can identify particular columns to select and display.

# COMMAND ----------

display(df.select("trip_duration"))

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 4: (Optional) Create a view or table
# MAGIC 
# MAGIC If you want to query this data as a table, you can simply register it as a *view* or a table.

# COMMAND ----------

df.createOrReplaceTempView("TEMP_NYCITIBIKE")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We can query this view using Spark SQL. For instance, we can perform a simple aggregation. Notice how we can use `%sql` to query the view from SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT gender, ROUND(AVG(trip_duration),2) FROM TEMP_NYCITIBIKE GROUP BY gender

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since this table is registered as a temp view, it will be available only to this notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("TAB_NYCITIBIKE")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This table will persist across cluster restarts and allow various users across different notebooks to query this data.

# COMMAND ----------


