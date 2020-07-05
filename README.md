# Machine Learning -- Continuous Assessment

## Dataset
:mushroom: :mushroom: :mushroom: [mushroom.csv](https://archive.ics.uci.edu/ml/datasets/Mushroom)

:book: :book: :book: [student.mat.csv](https://archive.ics.uci.edu/ml/datasets/Student+Performance#)

:wine_glass: :wine_glass: :wine_glass: [winequality-red.csv](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

## Structure
```
app
├── data
├── reports
├── src
│   ├── main.py
│   ├── processing.py
│   ├── supervised_learning.py
│   └── unsupervised_learning.py
└── README.md
```

- `main.py` -> Using processing and ML packages to perform testing on the dataset and generate reports
- `processing.py` -> Include preprocessing function for data engineering (options for label encoding, dummy encoding, scaling)
                 Feature selection using pearson correlation, PCA for feature engineering
- `supervised_learning.py` -> Include linear regression, logistic regression, knn, decision tree, neural network 
                          for regression and classification
- `unsupervised_learning.py` -> Include kmeans, hierarchical, dbscan for clustering