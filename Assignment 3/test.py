from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import itertools
import string


a = [[1,2,3],[4,5,6],[7,8,9]]
b = np.asarray(a)
c = [1,2,3,4]
d = [1,2,3,4]
print([c_i == d_i for c_i in c for d_i in d])