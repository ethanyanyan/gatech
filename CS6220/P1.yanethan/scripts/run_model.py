import os
import time
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

import misc

os.environ['SPARK_HOME'] = '/usr/local/spark'

# Start Spark session
spark = SparkSession.builder.appName("Million Song Model Training").master("local[*]").getOrCreate()

processed_dir = "data/processed/"
report_path = "results/reports/run_model_report.txt"

# Function to load preprocessed data
def load_preprocessed_data(subset_name):
    preprocessed_path = os.path.join(processed_dir, f"{subset_name}_preprocessed.csv")
    df_preprocessed = spark.read.csv(preprocessed_path, header=True, inferSchema=True)
    return df_preprocessed

# Function to train ALS model
def train_als_model(df_train, subset_name):
    start_time = time.time()
    
    als = ALS(userCol="user_index", itemCol="song_index", ratingCol="play_count",
              maxIter=10, regParam=0.1, coldStartStrategy="drop")
    model = als.fit(df_train)
    
    end_time = time.time()
    misc.log_spark_job_info(f"Training ALS model on {subset_name} subset", start_time, end_time, spark, report_path)
    
    return model

# Function to evaluate ALS model
def evaluate_model(model, df_test, subset_name):
    start_time = time.time()
    
    predictions = model.transform(df_test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="play_count", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    
    end_time = time.time()
    misc.log_spark_job_info(f"Evaluating ALS model on {subset_name} subset", start_time, end_time, spark, report_path)
    
    with open(report_path, 'a') as report_file:
        report_file.write(f"RMSE for {subset_name} subset: {rmse:.4f}\n")
        report_file.write("\n")
    
    return rmse

# Ensure the report file is fresh
misc.delete_if_exists(report_path)

# Load, train, and evaluate on small, medium, and large subsets
for subset_name in ["small", "medium", "large"]:
    df_preprocessed = load_preprocessed_data(subset_name)
    
    # Split data into training and test sets
    df_train, df_test = df_preprocessed.randomSplit([0.8, 0.2], seed=42)
    
    # Train the model
    model = train_als_model(df_train, subset_name)
    
    # Evaluate the model
    evaluate_model(model, df_test, subset_name)

# Stop the Spark session
spark.stop()
