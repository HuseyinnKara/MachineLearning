import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4],
              [5, 6], [5, 6], [7, 8], [3, 4], [20, 6]])

mms = MinMaxScaler(feature_range = (-1, 1))

X_normalized_with_min_max = mms.fit_transform(X)

df = pd.DataFrame(X_normalized_with_min_max)
print("\n-----Min Max Normalized Data-----\n", df)

inverse = mms.inverse_transform(X_normalized_with_min_max)
df = pd.DataFrame(inverse)
print("\n-----Min Max Inverse Data-----\n", df)

stdS = StandardScaler()
X_normalized_with_standardScaler = stdS.fit_transform(X)
df = pd.DataFrame(X_normalized_with_standardScaler)
print("\n-----StandarScaler Normalized Data-----\n", df)

inverseStd = stdS.inverse_transform(X_normalized_with_standardScaler)
df = pd.DataFrame(inverseStd)
print("\n-----StandarScaler Inverse Data-----\n", df)
