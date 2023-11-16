from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from utils import *

conf=SparkConf()
conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
sc=SparkContext(conf=conf)
sqlContext=SQLContext(sc)

df=sqlContext.read.csv('training.1600000.processed.noemoticon.csv',header=True)
df=df.rdd
df=df.map(lambda x:(x[0],x[5]))
df_processed=df.map(lambda x:(0 if x[0]=='0' else 1,preprocessingText(x[1])))

df_processed=df_processed.toDF(["sentiment", "text"])

df_processed=df_processed.toPandas()