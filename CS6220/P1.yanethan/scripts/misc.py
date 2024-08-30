import os
import shutil
import time

def delete_if_exists(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)  # Remove directory and contents
        else:
            os.remove(path)  # Remove file

def log_spark_job_info(description, start_time, end_time, spark, report_path):
    duration = end_time - start_time
    num_executors = spark.sparkContext.defaultParallelism
    
    with open(report_path, 'a') as report_file:
        report_file.write(f"{description}\n")
        report_file.write(f"Number of Executors: {num_executors}\n")
        report_file.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        report_file.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        report_file.write(f"Duration: {duration:.2f} seconds\n")
        report_file.write("\n")