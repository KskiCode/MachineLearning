from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# same dataset as in main.py
x1 = np.array([2.4, 0.9, 1.5, 1.5, 2.5, 1.6, 1.0, 1.0, 1.5, 1.1])
x2 = np.array([3.5, 1.1, 2.2, 1.9, 2.8, 2.7, 1.6, 1.1, 1.9, 2.2])

# some preprocessing for pca - like centraliszing around (0,0)
def create_dataframe(x1, x2):
    df = pd.DataFrame({'x1': x1, 'x2': x2})
    return df

print('Basic Statistic:')
# some basic statistic
x1_mean = round(x1.mean(), 2)
x2_mean = round(x2.mean(), 2)
print('Mean x1:', x1_mean)
print('Mean x2:', x2_mean)
# pearson correlation between x1 and x2
corr_pearson = round(np.corrcoef(x1, x2)[1,0],3)
print('r(pearson):', corr_pearson)
print('*'*40)
print(" ")

# step 1 - transform dataset from mean-centered to the center around 0 
def center_dataset(arr, mean):
    arr = np.asarray(arr)
    arr_centered = np.empty((0, 10), dtype=float)
    for x in arr:
        new_value = x - mean
        arr_centered = np.append(arr_centered, new_value)
    return arr_centered
    
x1_centered = center_dataset(x1, x1_mean)
x2_centered = center_dataset(x2, x2_mean)
#print(x1_centered)
#print(x2_centered)

df_centered = create_dataframe(x1_centered, x2_centered)

# pca with scikit-learn
pca = PCA(n_components=2) # comparison between the two attributes/components
X1D = pca.fit_transform(df_centered)

print("Explained variance ratio with PCA")
print(pca.explained_variance_ratio_)