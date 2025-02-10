#-----------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#-----------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
    dot(a, b)/(norm(a)*norm(b))

def rsme(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))


def pretty_print(U):
    U = pd.DataFrame(U)
    U.columns = ['']*U.shape[1]
    print(U.to_string(index=False))

rng = np.random.default_rng()

a = [
    [5, 4, 2, 3, 1],
    [4, 3, 1, 1, 5],
    [2, 3, 2, 4, 3],
]
#a = np.random.normal(size=[5000,50])


cols = len(a[0])
rows = len(a)

U, S, Vh = np.linalg.svd(a, full_matrices=True)
S[len(S) - 1] = 0; # drop lowest singular value ==> reduce noise / reduce dimensionality

Sx = np.zeros((rows, cols))
Sx[:cols, :rows] = np.diag(S) # map vector S back to a 3x3 matrix

# pretty_print(U)
# pretty_print(Sx)
# pretty_print(Vh)
US = np.dot(U, Sx) #U dot Sigma 
# pretty_print(US)

print(sum(US[0] * Vh.T[3]))
print(sum(US[1] * Vh.T[1]))
print(sum(US[2] * Vh.T[4]))
a_approx = np.dot(U, np.dot(Sx, Vh))

pretty_print(np.matrix.round(a_approx, 20)) # Print rounded values

# print(rsme(a_approx, a)); # print deviation from original data (0-1 scale, lower is better, large data demonstrates low error)