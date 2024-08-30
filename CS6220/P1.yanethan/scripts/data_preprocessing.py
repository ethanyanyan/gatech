import os
import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

import misc

os.environ['SPARK_HOME'] = '/usr/local/spark'

# Start Spark
spark = SparkSession.builder.appName("Million Song Mining").master("local[*]").getOrCreate()

path_to_dataset = "data/raw/kaggle_visible_evaluation_triplets.txt"
subset_dir = "data/subsets/"
processed_dir = "data/processed/"
report_path = "results/reports/data_processing_report.txt"

def create_user_subset(df, num_users, subset_name, spark):
    start_time = time.time()
    user_subset = df.select('user_id').distinct().orderBy("user_id").sample(withReplacement=False, fraction=1.0).limit(num_users)
    df_subset = df.join(user_subset, on='user_id', how='inner')
    
    # Define path, delete if it exists
    subset_path = os.path.join(subset_dir, f"{subset_name}_subset.csv")
    misc.delete_if_exists(subset_path)
    
    df_subset.write.csv(subset_path, header=True)
    end_time = time.time()
    misc.log_spark_job_info(f"Creating {subset_name} subset", start_time, end_time, spark, report_path)
    return df_subset

# Preprocess data (index user_id and song_id)
def preprocess_data(df, subset_name, spark):
    start_time = time.time()
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
    song_indexer = StringIndexer(inputCol="song_id", outputCol="song_index")
    pipeline = Pipeline(stages=[user_indexer, song_indexer])
    df_preprocessed = pipeline.fit(df).transform(df)
    
    # Define path and delete if it exists
    preprocessed_path = os.path.join(processed_dir, f"{subset_name}_preprocessed.csv")
    misc.delete_if_exists(preprocessed_path)
    
    df_preprocessed.write.csv(preprocessed_path, header=True)
    end_time = time.time()
    misc.log_spark_job_info(f"Preprocessing {subset_name} subset", start_time, end_time, spark, report_path)
    return df_preprocessed


# Load dataset
df = spark.read.csv(path_to_dataset, sep='\t', inferSchema=True, header=False)
df = df.withColumnRenamed("_c0", "user_id") \
       .withColumnRenamed("_c1", "song_id") \
       .withColumnRenamed("_c2", "play_count")

misc.delete_if_exists(report_path)

small_subset = create_user_subset(df, 1000, "small", spark)
medium_subset = create_user_subset(df, 10000, "medium", spark)
large_subset = create_user_subset(df, 100000, "large", spark)

small_preprocessed = preprocess_data(small_subset, "small", spark)
medium_preprocessed = preprocess_data(medium_subset, "medium", spark)
large_preprocessed = preprocess_data(large_subset, "large", spark)

spark.stop()
