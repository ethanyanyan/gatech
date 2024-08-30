# Million Song Dataset Mining Project

## Overview

This project is part of an assignment to mine the Kaggle Million Songs dataset using Apache Spark and other machine learning libraries. The objective is to predict the missing half of the listening history for 110K users based on a large dataset of user-song interactions.

## Project Structure

P1.yanethan/
│
├── data/
│ ├── raw/ # Raw dataset files
│ ├── processed/ # Processed datasets used for training/testing
│ └── subsets/ # Different subsets of data (small, medium, large)
│
├── notebooks/
│ ├── 01_data_exploration.ipynb # Jupyter notebook for initial data exploration
│ ├── 02_data_preprocessing.ipynb # Jupyter notebook for data preprocessing
│ ├── 03_model_training.ipynb # Jupyter notebook for training models
│ ├── 04_evaluation.ipynb # Jupyter notebook for evaluating model performance
│ └── 05_experiments.ipynb # Jupyter notebook for running experiments
│
├── scripts/
│ ├── data_preprocessing.py # Script for data preprocessing
│ ├── train_model.py # Script for model training
│ ├── evaluate_model.py # Script for model evaluation
│ └── run_experiments.py # Script for running experiments on different dataset sizes
│ ├── results/
│ ├── models/ # Saved models
│ ├── logs/ # Logs of experiments and training
│ └── reports/ # Evaluation reports and performance metrics
│
├── requirements.txt # List of dependencies
├── README.md # Project documentation
└── .gitignore # Ignore unnecessary files in git

## Dataset

- **URL:** [Million Song Dataset on Kaggle](http://www.kaggle.com/c/msdchallenge)
- **Subset Used:**
  - Initial Subset: 1000 users, 10,000 songs
  - Medium Subset: 5,000 users, 50,000 songs
  - Large Subset: 10,000 users, 100,000 songs

## Tools and Libraries

- **Apache Spark:** Data processing and MLlib for machine learning.
- **Jupyter Notebooks:** Development environment for exploration and analysis.
- **Python:** General programming and scripting.
- **Scikit-learn:** Machine learning library for model development.
- **Pandas:** Data manipulation and processing.
- **Matplotlib/Seaborn:** Visualization tools.

## Open Source Code Used

- **Hadoop MapReduce Examples:** Adapted from [Hadoop Official Documentation](https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html)
- **Spark MLlib Examples:** Adapted from [Spark Official Documentation](https://spark.apache.org/examples.html)

## Model Selection: Considered Alternatives and Why ALS was Chosen

In the process of selecting an appropriate algorithm for predicting the missing listening history in the Million Song Dataset, several models and algorithms were considered. After evaluating the strengths and weaknesses of each, ALS (Alternating Least Squares) was chosen for this project. Below is a brief overview of the alternatives considered and the rationale for ultimately choosing ALS.

### Considered Alternatives

1. **K-Nearest Neighbors (KNN) for Collaborative Filtering**

   - **Overview:** KNN is a straightforward algorithm that identifies the nearest neighbors of a user or item based on a similarity metric (e.g., cosine similarity or Pearson correlation). It is often used in memory-based collaborative filtering methods.
   - **Advantages:**
     - Simple to understand, implement and interpret.
     - Effective for small to medium-sized datasets.
   - **Disadvantages:**
     - **Scalability Issues:** KNN struggles with scalability, especially with large datasets like the Million Song Dataset, due to the high computational cost of calculating similarities across millions of users or items. It would perform fine on smaller datasets though.
     - **Cold Start Problem:** KNN does not handle the cold start problem well, as it relies heavily on the presence of similar users or items in the dataset.

2. **Matrix Factorization with Singular Value Decomposition (SVD)**
   - **Overview:** SVD is a matrix factorization technique that decomposes a user-item interaction matrix into three matrices: one representing users, one representing items, and a diagonal matrix of singular values. It can capture latent factors similar to ALS.
   - **Advantages:**
     - Good in capturing latent factors and reducing dimensionality.
     - Can produce high-quality recommendations by identifying hidden patterns.
   - **Disadvantages:**
     - **Lack of Implicit Feedback Handling:** Standard SVD is not designed for implicit feedback data like play counts. It typically requires explicit ratings, making it less suitable for this dataset.

### Why ALS was Ultimately Chosen

After considering the above alternatives, ALS was selected as the most suitable algorithm for the following reasons:

- **Scalability:** Able to work efficiently with large datasets and is optimized for distributed computing environments like Apache Spark.
- **Handling of Implicit Feedback:** Unlike traditional matrix factorization methods like SVD, ALS can natively handle implicit feedback data, such as play counts, making it an ideal choice for the MSD.
- **Regularization and Flexibility:** ALS includes regularization to prevent overfitting and offers flexibility in tuning hyperparameters to optimize model performance.

## Experiments

### Experiment 1: Small Dataset (1000 users, 10,000 songs)

- **Training Time:** X minutes
- **Validation Accuracy (RMSE):** X

### Experiment 2: Medium Dataset (5,000 users, 50,000 songs)

- **Training Time:** X minutes
- **Validation Accuracy (RMSE):** X

### Experiment 3: Large Dataset (10,000 users, 100,000 songs)

- **Training Time:** X minutes
- **Validation Accuracy (RMSE):** X

## Results

- **Performance Trends:** As the dataset size increased, the model showed a trend of increasing/decreasing accuracy.
- **Generalization:** Larger datasets generally provided better generalization but required more computational resources and time to train.

## Lessons Learned

1. **Data Preprocessing is Key:** Proper handling of missing values and normalization significantly improved model performance.
2. **Scalability Challenges:** As the dataset size increased, managing memory and ensuring efficient data processing became critical to avoid performance bottlenecks.
3. **Model Selection:** The choice of algorithm, as well as careful tuning of hyperparameters, had a significant impact on the model’s predictive accuracy.

## How to Run the Project

1. Clone this repository: `git clone <repository-url>`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the dataset from Kaggle and place it in the `data/raw/` directory.
4. Run the data preprocessing script: `python scripts/data_preprocessing.py`
5. Train the model: `python scripts/train_model.py`
6. Evaluate the model: `python scripts/evaluate_model.py`
7. Run experiments: `python scripts/run_experiments.py`

## References

- [Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## Contact

- **Ethan Yan** - [Email](mailto:eyan38@gatech.edu)
