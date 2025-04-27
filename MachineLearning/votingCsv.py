import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

def get_voting():
    models = list()
    models.append(('knn1', KNeighborsClassifier(n_neighbors=2)))
    models.append(('knn3', KNeighborsClassifier(n_neighbors=3)))
    models.append(('knn5', KNeighborsClassifier(n_neighbors=5)))
    models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
    models.append(('knn9', KNeighborsClassifier(n_neighbors=9)))
    
    hardvoting = VotingClassifier(estimators=models, voting='hard')
    softvoting = VotingClassifier(estimators=models, voting='soft')
    return hardvoting, softvoting

def get_models():
    models = dict()
    models['knn1'] = KNeighborsClassifier(n_neighbors=2)
    models['knn3'] = KNeighborsClassifier(n_neighbors=3)
    models['knn5'] = KNeighborsClassifier(n_neighbors=5)
    models['knn7'] = KNeighborsClassifier(n_neighbors=7)
    models['knn9'] = KNeighborsClassifier(n_neighbors=9)
    
    models['hard_voting'], models['soft_voting'] = get_voting()
    return models

def evalute_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=13)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf)
    return scores

file_path = '/voting_classifier_dataset.csv'
data = pd.read_csv(file_path)

print(data.head())
print(data.isnull().sum())

#data.dropna(inplance=True)
data['Missing_Col_1'] = data['Missing_Col_1'].replace(np.nan, data['Missing_Col_1'].mean())
data['Missing_Col_2'] = data['Missing_Col_2'].replace(np.nan, data['Missing_Col_2'].median())
print(data.isnull().sum())

benzersiz_degerler = data.nunique()
tumu_ayni_sutunlar = benzersiz_degerler[benzersiz_degerler == 1].index.tolist()
tumu_farkli_sutunlar = benzersiz_degerler[benzersiz_degerler == len(data)].index.tolist()
print("Tum degerleri ayni olan sutunlar: ", tumu_ayni_sutunlar)
drop_sutunlar = tumu_ayni_sutunlar + tumu_farkli_sutunlar
data = data.drop(columns=drop_sutunlar, axis=1)
print(data.head())

X = data.drop('Target', axis=1)
y = data['Target']


models = get_models()

results, names = list(), list()
for name, model in models.items():
    scores = evalute_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    
plt.boxplot(results, labels=names, showmeans=True)
plt.show()