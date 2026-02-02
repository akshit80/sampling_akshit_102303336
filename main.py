import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# pip install imbalanced-learn
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

## using different ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score

# Task - Converting imbalanced dataset to balanced dataset using Random Over Sampling
# Loading the dataset
df = pd.read_csv('Creditcard_data.csv')

# Checking class Imbalance
# 0 - 763
# 1 - 9
df['Class'].value_counts()

print(df['Class'].value_counts())

X = df.drop(columns=['Class'])
y = df['Class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Applying Random Over Sampling
# random state - used for reproducibility
balance_df = RandomOverSampler(
    sampling_strategy='auto',
    random_state=42
)

X_train_over, y_train_over = balance_df.fit_resample(X_train, y_train)[:2]
print(y_train_over.value_counts())

# Creating 5 samples from the balanced dataset

balanced_trained_df = pd.concat(
    [X_train_over, y_train_over],
    axis=1
)

##Simple Random Sampling - 70% of the balanced dataset
sample_random = balanced_trained_df.sample(
    frac=0.7,
    random_state=42
)

##Systematic Sampling - every 3rd record from the balanced dataset
k = int(len(balanced_trained_df)/10)
sample_systematic = balanced_trained_df.iloc[::k,:]

##Stratified Sampling 
X = balanced_trained_df.drop(columns=['Class'])
y = balanced_trained_df['Class']

X_stratified,_,y_stratified,_ = train_test_split(
    X,y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

sample_stratified = pd.concat([X_stratified, y_stratified], axis=1)

## Cluster Sampling
num_clusters = 5
shuffled_df = balanced_trained_df.sample(frac=1, random_state=42).reset_index(drop=True)

chunk_size = len(shuffled_df) // num_clusters
clusters = [shuffled_df.iloc[i : i + chunk_size] for i in range(0, len(shuffled_df), chunk_size)]

selected_indices = np.random.choice(range(len(clusters)), size=2, replace=False)
sample_cluster = pd.concat([clusters[i] for i in selected_indices]).reset_index(drop=True)


## bootstrap Sampling
sample_bootstrap = balanced_trained_df.sample(
    frac=1.0,
    replace=True,
    random_state=42
)

samples = {
    "Random": sample_random,
    "Systematic": sample_systematic,
    "Stratified": sample_stratified,
    "Cluster": sample_cluster,
    "Bootstrap": sample_bootstrap
}

## Printing 5 samples
# for name, sample in samples.items():
#     print(f"\n{name} sample distribution:")
#     print(sample['Class'].value_counts())

## applying different ML models on each sample
models = {
    "M1_LogisticRegression":LogisticRegression(max_iter=1000),
    "M2_RandomForest":RandomForestClassifier(n_estimators=100, random_state=42),
    "M3_SVM":SVC(random_state=42),
    "M4_KNN":KNeighborsClassifier(n_neighbors=5),
    "M5_DecisionTree":DecisionTreeClassifier(random_state=42)
}

## Split function
def split_sample(df):
    X = df.drop(columns='Class')
    y = df['Class']
    
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

### Evaluating models on each sample
results = []

for sample_name, sample_df in samples.items():
    
    X_train, X_test, y_train, y_test = split_sample(sample_df)
    
    for model_name, model in models.items():
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            "Sample": sample_name,
            "Model": model_name,
            "Accuracy": accuracy
        })



results_df = pd.DataFrame(results)
print(results_df)

best_result = results_df.loc[results_df['Accuracy'].idxmax()]
print(best_result)
