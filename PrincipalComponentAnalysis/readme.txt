## Principal Component Analysis (PCA)

In Machine Learning many tasks include datasets with a lot of attributes. Hence training or finding a right solution can take very a long time. This problem is also know as the "Curse of dimensionality". However, there are various methods that can help to reduce dimensionality during preprocessing. One of the most common way is the Principal Component Analysis (PCA). In PCA first a hyperplane is searched that comes closest to the data, second the data is projected onto this hyperplane.

Note: dimensionality reduction comes with loss of information. Since the original dimensions of the dataset are reduced also information will get lost.

When finding the right "hyperplane" the maximum variance of the data should be considered as this indicates the minimum loss of information when the data will projected.

In this code example I used a simple 2-D dataset just to illustrate the process. There are two Python scripts in this file:

1.) main.py - contains a PCA "from scratch" - without using the Scikit-Learn PCA() only numpy, pandas and matplotlib
2.) pca.py - makes use of Scikit-Learn PCA()


The main.py follows 4 simple steps:
#1: Transforming the dataset with a mean around (0,0)
#2: Creating the CoVariance Matrix C
#3: Calculating the Eigenvalues and Eigenvectors of Matrix C
# 4: Calculating the share of variance of each principal component

More information about PCA: https://en.wikipedia.org/wiki/Principal_component_analysis


KskiCode
