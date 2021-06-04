# Databricks notebook source
# MAGIC %md
# MAGIC ## Load data
# MAGIC Get a sample data of nyc yellow taxi from Azure Open Datasets

# COMMAND ----------

!pip install azureml-opendatasets

# COMMAND ----------

from azureml.opendatasets import NycTlcYellow
from datetime import datetime
from dateutil import parser

start_date = parser.parse('2018-05-01')
end_date = parser.parse('2018-05-07')
nyc_tlc = NycTlcYellow(start_date=start_date, end_date=end_date)
nyc_tlc_df = nyc_tlc.to_pandas_dataframe()
nyc_tlc_df.info()

# COMMAND ----------

#from IPython.display import display

sampled_df = nyc_tlc_df.sample(n=10000, random_state=123)
display(sampled_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare and featurize data
# MAGIC - There are extra dimensions that are not going to be useful in the model. We just take the dimensions that we need and put them into the featurised dataframe. 
# MAGIC - There are also a bunch of outliers in the data so we need to filter them out.

# COMMAND ----------

import numpy
import pandas

def get_pickup_time(df):
    pickupHour = df['pickupHour'];
    if ((pickupHour >= 7) & (pickupHour <= 10)):
        return 'AMRush'
    elif ((pickupHour >= 11) & (pickupHour <= 15)):
        return 'Afternoon'
    elif ((pickupHour >= 16) & (pickupHour <= 19)):
        return 'PMRush'
    else:
        return 'Night'

featurized_df = pandas.DataFrame()
featurized_df['tipped'] = (sampled_df['tipAmount'] > 0).astype('int')
featurized_df['fareAmount'] = sampled_df['fareAmount'].astype('float32')
featurized_df['paymentType'] = sampled_df['paymentType'].astype('int')
featurized_df['passengerCount'] = sampled_df['passengerCount'].astype('int')
featurized_df['tripDistance'] = sampled_df['tripDistance'].astype('float32')
featurized_df['pickupHour'] = sampled_df['tpepPickupDateTime'].dt.hour.astype('int')
featurized_df['tripTimeSecs'] = ((sampled_df['tpepDropoffDateTime'] - sampled_df['tpepPickupDateTime']) / numpy.timedelta64(1, 's')).astype('int')

featurized_df['pickupTimeBin'] = featurized_df.apply(get_pickup_time, axis=1)
featurized_df = featurized_df.drop(columns='pickupHour')

display(featurized_df.head(5))


# COMMAND ----------

filtered_df = featurized_df[(featurized_df.tipped >= 0) & (featurized_df.tipped <= 1)\
    & (featurized_df.fareAmount >= 1) & (featurized_df.fareAmount <= 250)\
    & (featurized_df.paymentType >= 1) & (featurized_df.paymentType <= 2)\
    & (featurized_df.passengerCount > 0) & (featurized_df.passengerCount < 8)\
    & (featurized_df.tripDistance >= 0) & (featurized_df.tripDistance <= 100)\
    & (featurized_df.tripTimeSecs >= 30) & (featurized_df.tripTimeSecs <= 7200)]

filtered_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the data to spark table

# COMMAND ----------

spark_df = spark.createDataFrame(filtered_df)
spark_df.write.mode("overwrite").saveAsTable("tab_nyctaxi")

# COMMAND ----------


