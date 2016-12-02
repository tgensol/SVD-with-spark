# easy way to install pyspark, no need for big configuration
import findspark
findspark.init()

import svd #import the code

import scipy.io #only for loading the matlab matrix here

# Test matrix from AMOS code
A = scipy.sparse.csr_matrix(scipy.io.loadmat('IEEERTS96Adjacency.mat')['A'])

# Once the RowMatrix is ready we can compute our Singular Value Decomposition
svd = svd.computeSVD(A,2,True)
print svd.s
print svd.U
print svd.V