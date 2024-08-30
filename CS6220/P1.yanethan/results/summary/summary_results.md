# Million Song Dataset Mining Project Summary

## Overview

This document summarizes the key results and metrics obtained during the data pre-processing, model training, and model evaluation of the ALS model algorithm on different subsets of the Million Song Dataset.

### Project Structure

THe reports obtained for data processing and model training and evaluation can be found at the respective locations:

- **Data Processing Report:** `results/reports/data_processing_report.txt`
- **Model Training and Evaluation Report:** `results/reports/run_model_report.txt`

## Data Processing

### Subset Creation

The subsets were created by randomly selecting distinct user IDs and joining them with the original dataset. The data was then saved for further processing.

- **Small Subset:** 1000 users
- **Medium Subset:** 10000 users
- **Large Subset:** 100000 users

#### Subset Creation Metrics

| Subset     | Executors | Start Time          | End Time            | Duration |
| ---------- | --------- | ------------------- | ------------------- | -------- |
| **Small**  | 8         | 2024-08-29 21:25:12 | 2024-08-29 21:25:15 | 3.06 sec |
| **Medium** | 8         | 2024-08-29 21:25:15 | 2024-08-29 21:25:17 | 2.56 sec |
| **Large**  | 8         | 2024-08-29 21:25:17 | 2024-08-29 21:25:20 | 2.73 sec |

We note that the subset creation times do not increase linearly. This is expected as Spark optimizes its operations based on its available resources and the characteristics of the data, which could lead to variations in processing times that may not scale linearly with data size.

### Data Preprocessing

The preprocessing involved converting `user_id` and `song_id` into numeric indices using Spark's `StringIndexer` function. This step was essential for training the ALS model. ALS expects user IDs and song IDs to be represented as numeric values, where the algorithm then performs matrix factorization.

#### Preprocessing Metrics

| Subset     | Executors | Start Time          | End Time            | Duration |
| ---------- | --------- | ------------------- | ------------------- | -------- |
| **Small**  | 8         | 2024-08-29 21:25:20 | 2024-08-29 21:25:25 | 4.82 sec |
| **Medium** | 8         | 2024-08-29 21:25:25 | 2024-08-29 21:25:30 | 5.66 sec |
| **Large**  | 8         | 2024-08-29 21:25:30 | 2024-08-29 21:25:39 | 8.83 sec |

We note that the duration of preprocessing subset times increase. This is expected as larger datasets naturally require more time to process due to the increased volume of data that needs to be indexed, shuffled, and transformed.

## Model Training and Evaluation

### ALS Model Training

The ALS model was trained on the above preprocessed subsets. We note that the training time increased as the subset size increased.

#### Training Metrics

| Subset     | Executors | Start Time          | End Time            | Duration |
| ---------- | --------- | ------------------- | ------------------- | -------- |
| **Small**  | 8         | 2024-08-29 21:40:39 | 2024-08-29 21:40:42 | 2.37 sec |
| **Medium** | 8         | 2024-08-29 21:40:43 | 2024-08-29 21:40:46 | 3.35 sec |
| **Large**  | 8         | 2024-08-29 21:40:48 | 2024-08-29 21:40:56 | 7.52 sec |

We note that the training times increase as the dataset size grows. This is expected as larger datasets require more computational resources and time for the ALS model to process the increased number of user-song interactions.

### Model Evaluation

The model was evaluated using RMSE (Root Mean Square Error), which is a common metric for measuring the accuracy of a recommendation system.

#### Evaluation Metrics

| Subset     | Executors | Start Time          | End Time            | Duration | RMSE   |
| ---------- | --------- | ------------------- | ------------------- | -------- | ------ |
| **Small**  | 8         | 2024-08-29 21:40:42 | 2024-08-29 21:40:42 | 0.81 sec | 7.4697 |
| **Medium** | 8         | 2024-08-29 21:40:46 | 2024-08-29 21:40:47 | 1.13 sec | 8.6925 |
| **Large**  | 8         | 2024-08-29 21:40:56 | 2024-08-29 21:40:58 | 2.48 sec | 8.5858 |

We observe that the RMSE does not decrease with increasing dataset size as might be expected. In fact, the RMSE slightly increases from the small subset to the medium and large subsets. This counterintuitive result suggests several possible factors at play:

- Data Sparsity: As the dataset size increases, the sparsity in the user-song interaction matrix may also increase, which could potentially lead to more difficulties for the ALS model to accurately predict ratings. This can lead to higher RMSE values despite the larger amount of data.
- Overfitting: The ALS model may start to overfit to the larger dataset especially since there is no regularization parameter in this situation, leading to worse generalization on the test set. This could cause the RMSE to increase as the model becomes more complex and starts to capture noise in the data.
- Need for Hyperparameter Tuning: The ALS model's hyperparameters, such as rank, maxIter, and regParam, may need to be tuned specifically for each dataset size to achieve better performance.

## Observations and Lessons Learned

1. **Processing Time Variations:** The processing times did not scale linearly with the subset size potentially due to Spark's internal optimizations.
2. **Model Accuracy:** Contrary to expectations, the RMSE values increased as the dataset size increased. This may indicate issues with data sparsity, overfitting, or the need for hyperparameter tuning as the dataset size grows.
3. **Scalability:** The project demonstrated Spark's ability to handle large-scale data processing and model training effectively, though for better results, other portions of the model training could be better handled, such as hyperparameter tuning.

## Conclusion

This project successfully processed, trained, and evaluated a recommendation system using the ALS model on various subsets of the Million Song Dataset.

For detailed logs and metrics, please refer to the following files:

- `results/reports/data_processing_report.txt`
- `results/reports/run_model_report.txt`
