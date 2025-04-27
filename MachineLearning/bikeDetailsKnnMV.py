import pandas as pd
from sklearn.impute import KNNImputer

knnData = pd.read_csv(r"\Data\bikedetails.csv")

print(knnData["ex_showroom_price"][:20])

fea_transform = KNNImputer(n_neighbors = 3)

values = fea_transform.fit_transform(knnData[["ex_showroom_price"]])

knnData["ex_showroom_price"] = pd.DataFrame(values)

print(knnData["ex_showroom_price"][:20])
