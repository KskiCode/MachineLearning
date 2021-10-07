import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# some dataset with 10 values and 2 variables - linear
x1 = np.array([2.4, 0.9, 1.5, 1.5, 2.5, 1.6, 1.0, 1.0, 1.5, 1.1])
x2 = np.array([3.5, 1.1, 2.2, 1.9, 2.8, 2.7, 1.6, 1.1, 1.9, 2.2])

# create pandas df for the two arrays
def create_dataframe(x1, x2):
    df = pd.DataFrame({'x1': x1, 'x2': x2})
    return df

df_original = create_dataframe(x1, x2)

# plot original dataset.
df_original.plot('x1', 'x2', kind='scatter')
plt.title('Initial Dataset')
#plt.show()

# some basic statistic that is needed
x1_mean = round(x1.mean(), 2)
x2_mean = round(x2.mean(), 2)
print('Mean x1:', x1_mean)
print('Mean x2:', x2_mean)

# pearson correlation between x1 and x2
corr_pearson = round(np.corrcoef(x1, x2)[1,0],3)
print('r(pearson):', corr_pearson)
print('*'*40)

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

df_centered = create_dataframe(x1_centered, x2_centered)

# plot the centered dataset.
df_centered.plot('x1', 'x2', kind='scatter')
plt.title('Centered Dataset')
#plt.show()

# create matrix X for the centered values. (10x2 matrix)
x_matrix = np.column_stack((x1_centered, x2_centered))

# Step 2 - Create CoVariance Matrix
# matrix C
covariance_matrix = np.cov(x_matrix.transpose())
print('Covariance Matrix')
print(covariance_matrix)
print()

# Step 3 - Eigenanalysis - Find Eigenvector and Eigenvalue of Matrix C
w, v = np.linalg.eig(covariance_matrix)
print('Eigenvalues')
print(w)
print()
print('Eigenvectors')
print(v)
print()

# step 4 - calculate the share of variance of each principal component
sum_eigenvalues = w[0]+ w[1] # sum_variance = sum of eigenvalues

# share of (total)variance for each eigenvector
share_of_variance_1 = round(w[0] / sum_eigenvalues, 3)
share_of_variance_2 = round(w[1] / sum_eigenvalues, 3)


print('Share of variance PC1 (%):', share_of_variance_2*100)
print('Share of variance PC2 (%):', share_of_variance_1*100)