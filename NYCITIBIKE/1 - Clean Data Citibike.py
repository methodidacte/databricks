# Databricks notebook source
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from math import acos, atan, cos, pi, radians, sin, sqrt
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType

# COMMAND ----------

@udf("float")
def Distance(lat1, long1, lat2, long2, unite):
    """
    Calculates the distance between two points with latitude and longitude as input parameters.
    'lat1, long1, lat2, long2,' int in str or float type.
    'unite' str type.
    """
    # Convert in miles or in km by default.
    if unite == 'miles':
        convert = 1.60934
    else:
        convert = 1
    
    # Set latitude and longitude in float().
    lat1 = float(lat1)
    long1 = float(long1)
    lat2 = float(lat2)
    long2 = float(long2)
    
    # Set latitude and longitude in radian.
    lat1rad = lat1 * pi / 180
    long1rad = long1 * pi / 180
    lat2rad = lat2 * pi / 180
    long2rad = long2 * pi / 180
    
    #Radius of the earth
    r = 6372.797
    
    # Radian latitude2 minus latitude
    dlat = lat2rad - lat1rad
    
    # Radian longitude2 minus longitude
    dlng = long2rad - long1rad
    
    a = sin(dlat / 2) * sin(dlat / 2) + cos(lat1) * cos(lat2) * sin(dlng / 2) * sin(dlng / 2)
    try:
        c = 2 * atan(sqrt(a) / sqrt(1 - a))
    except:
        # if a < 0 c funtion break, return 0 and drop line if distance == 0.
        # a < 0 when bike is in repaire center, latitude and longitude is set at 0.
        return 0
        
    km = r * c
    
    return km/convert

# COMMAND ----------

def rename_columns(df, columns):
  """
  Rename all columns present in variable 'columns' in the dataset 'df.
  """
  if isinstance(columns, dict):
      for old_name, new_name in columns.items():
          df = df.withColumnRenamed(old_name, new_name)
      return df
  else:
      raise ValueError("'columns' should be a dict, like {'old_name_1':'new_name_1', 'old_name_2':'new_name_2'}")

# COMMAND ----------

def ColToDummies(df, columns):
  """
  Take dataset in parameter 'df' and column 'columns'.
  For this column look all distinct values.
  Create columns with all distinct values and for each row if 'column' values is equal to the new column name take as values 1 else 0.
  """
  categories = df.select(columns).distinct().rdd.flatMap(lambda x: x).collect()

  exprs = [F.when(F.col(columns) == category, 1).otherwise(0).alias(category)
           for category in categories]
  return exprs

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

def ColToDummiesOHE(df, column):

  stringIndexer = StringIndexer(inputCol=column, outputCol=column+"Index")
  model = stringIndexer.fit(df)
  indexed = model.transform(df)

  encoder = OneHotEncoder(inputCol=column+"Index", outputCol=column+"Vec")
  encoded = encoder.transform(indexed)

  return encoded

# COMMAND ----------

# Set all variables
location_field = "/mnt/nycitibike/CSV/*/*"
location_field_date = "/mnt/nycitibike/CSV/"
format_field = "csv"

# Set schema for csv to dataset.
schema = StructType([
    StructField("trip_duration", IntegerType(), True),
    StructField("start_time", TimestampType(), True),
    StructField("stop_time", TimestampType(), True),
    StructField("start_station_id", IntegerType(), True),
    StructField("start_station_name", StringType(), True),
    StructField("start_station_latitude", DoubleType(), True),
    StructField("start_station_longitude", DoubleType(), True),
    StructField("end_station_id", IntegerType(), True),
    StructField("end_station_name", StringType(), True),
    StructField("end_station_latitude", DoubleType(), True),
    StructField("end_station_longitude", DoubleType(), True),
    StructField("bike_id", IntegerType(), True),
    StructField("usertype", StringType(), True),
    StructField("birth_year", IntegerType(), True),
    StructField("gender", StringType(), True),    
    ])

# COMMAND ----------

# Take all file in path 'location_field' with extention 'format_field'.load(location_field).
# .option("header","true") each files have header, pass this line.
# .option("mode", "DROPMALFORMED") Drop each lines dosen't have  the current schema.
# .schema(schema) override in parameter columns type and column name.

df = spark.read.format(format_field).option("header","true").option("mode", "DROPMALFORMED").schema(schema).load(location_field)

# COMMAND ----------

# Use display() on visualize data on Databricks
display(df)

# COMMAND ----------

df_raw = df

# COMMAND ----------

#https://stackoverflow.com/questions/44627386/how-to-find-count-of-null-and-nan-values-for-each-column-in-a-pyspark-dataframe

from pyspark.sql.functions import isnan, when, count, col

#df_raw.select([count(when(isnan(c), c)).alias(c) for c in df_raw.columns]).show()
#df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_raw.columns]).show()
df_raw.select('usertype').withColumn('isNull_c',F.col('usertype').isNull()).where('isNull_c = True').count() #51780

# COMMAND ----------

# Create dataframe without empty values
# Dropna with parameter how set to any drop all row with empty value.
df = df.dropna(how="any")

# COMMAND ----------

# New dataFrame with only 'trip_duration' values lower than 50 minutes. 
df = df.where((df.trip_duration <= 3000 ))

# COMMAND ----------

# MAGIC %time
# MAGIC df1 = df.select("trip_duration","start_time","stop_time","start_station_id","start_station_name","start_station_latitude","start_station_longitude","end_station_id","end_station_name","end_station_latitude","end_station_longitude","bike_id","usertype","birth_year", *ColToDummies(df,"gender"))

# COMMAND ----------

# MAGIC %time
# MAGIC df2 = ColToDummiesOHE(df,"gender")

# COMMAND ----------

df = df1
df = rename_columns(df, {"0":"unknown_gender","1":"male_gender","2":"female_gender"})

# COMMAND ----------

df = df.select("trip_duration","start_time","stop_time","start_station_id","start_station_name","start_station_latitude","start_station_longitude","end_station_id","end_station_name","end_station_latitude","end_station_longitude","bike_id","birth_year","unknown_gender","male_gender","female_gender", *ColToDummies(df,"usertype"))

# COMMAND ----------

df = df.withColumn('distance_bwn_stations', Distance("start_station_latitude","start_station_longitude","end_station_latitude","end_station_longitude",F.lit("km")))

# COMMAND ----------

# New dataFrame with only 'distance_bwn_stations' values greater than 0.0 km. 
df = df.where((df.distance_bwn_stations > 0))

# COMMAND ----------


