# Million Song Dataset Mining Project

## Overview

This project is part of an assignment to mine the Kaggle Million Songs dataset using Apache Spark and other machine learning libraries. The objective is to predict the missing half of the listening history for 110K users based on a large dataset of user-song interactions.

## Project Structure

```
P1.yanethan/
│
├── data/
│ ├── raw/ # Raw dataset files
│ ├── processed/ # Processed datasets used for training/testing
│ └── subsets/ # Different subsets of data (small, medium, large)
│
├── notebooks/
│ └── data_exploration.ipynb # Jupyter notebook for initial data exploration
│
├── scripts/
│ ├── data_preprocessing.py # Script for data preprocessing
│ ├── run_model.py # Script for training and evaluating the ALS model
│ └── misc.py # Commonly used function declarations
│
├── results/
│ ├── reports/ # Evaluation reports and performance metrics
│ └── summary/ # Summary report of experiment
│
├── requirements.txt # List of dependencies
├── README.md # Project documentation
└── .gitignore # Ignore unnecessary files in git
```

## Dataset

- **URL:** [Million Song Dataset on Kaggle](http://www.kaggle.com/c/msdchallenge)
- **Subsets Used:**
  - **Small Subset:** 1000 users
  - **Medium Subset:** 10,000 users
  - **Large Subset:** 100,000 users

## Tools and Libraries

- **Apache Spark:** Data processing and MLlib for machine learning.
- **Jupyter Notebooks:** Development environment for exploration and analysis.
- **Python:** General programming and scripting.

## Open Source Code Used

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

### Experiment 1: Small Dataset (1000 users)

- **Training Time:** 2.37 seconds
- **Validation Accuracy (RMSE):** 7.4697

### Experiment 2: Medium Dataset (10,000 users)

- **Training Time:** 3.35 seconds
- **Validation Accuracy (RMSE):** 8.6925

### Experiment 3: Large Dataset (100,000 users)

- **Training Time:** 7.52 seconds
- **Validation Accuracy (RMSE):** 8.5858

## Results

- **Performance Trends:** The RMSE did not decrease with increasing dataset size, suggesting potential issues with data sparsity, model complexity, or the need for hyperparameter tuning.
- **Generalization:** Larger datasets provided more complex models, but did not necessarily lead to better generalization without proper tuning.

## Detailed Analysis (Lessons Learned and Observations)

For a more detailed analysis of the results, including data processing metrics, training metrics, and model evaluation, please refer to the `summary_results.md` file located in the `results/reports/` directory:

- [Detailed Analysis: summary_results.md](results/reports/summary_results.md)

## How to Run the Project

1. Clone this repository: `git clone https://github.com/ethanyanyan/gatech.git`
2. Navigate to folder `cd CS6220/P1.yanethan`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Download the dataset from Kaggle and place it in the `data/raw/` directory.
5. Run the data preprocessing script: `python scripts/data_preprocessing.py`
6. Train and evaluate the model: `python scripts/run_model.py`

## References

- [Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)

## Contact

- **Ethan Yan** - [Email](mailto:eyan38@gatech.edu)
