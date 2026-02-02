## Title :- Sampling Assignment

## Methodology
- used Over sampling Technique to balance the dataset
- Used 5 sampling techniques - To divide the dataset into 5 different samples 
  1. Random Sampling
  2. Systematic Sampling
  3. Stratified Sampling
  4. Cluster Sampling
  5. Bootstrap Sampling

- used 5 ML models
  1. Logistic Regression
  2. SVM
  3. RandomForestClassifier
  4. DecisionTreeClassifier
  5. KNN

## Description
    Dataset Used - Credit Card fraud detection
    Number of rows: Class 0 - 763 , Class 1 - 9

    Results: 

    0       Random  M1_LogisticRegression  0.970760
    1       Random        M2_RandomForest  1.000000
    2       Random                 M3_SVM  0.654971
    3       Random                 M4_KNN  0.988304
    4       Random        M5_DecisionTree  0.988304
    5   Systematic  M1_LogisticRegression  0.500000
    6   Systematic        M2_RandomForest  0.500000
    7   Systematic                 M3_SVM  0.500000
    8   Systematic                 M4_KNN  0.500000
    9   Systematic        M5_DecisionTree  0.500000
    10  Stratified  M1_LogisticRegression  0.976608
    11  Stratified        M2_RandomForest  0.994152
    12  Stratified                 M3_SVM  0.678363
    13  Stratified                 M4_KNN  0.964912
    14  Stratified        M5_DecisionTree  0.988304
    15     Cluster  M1_LogisticRegression  0.938776
    16     Cluster        M2_RandomForest  1.000000
    17     Cluster                 M3_SVM  0.693878
    18     Cluster                 M4_KNN  0.948980
    19     Cluster        M5_DecisionTree  0.979592
    20   Bootstrap  M1_LogisticRegression  0.987705
    21   Bootstrap        M2_RandomForest  1.000000
    22   Bootstrap                 M3_SVM  0.790984
    23   Bootstrap                 M4_KNN  0.987705
    24   Bootstrap        M5_DecisionTree  1.000000

    Best Model:
        Sample               Random
        Model       M2_RandomForest
        Accuracy                1.0