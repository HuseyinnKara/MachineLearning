from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X,y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)

base_models=[
    ('knn',KNeighborsRegressor()),
    ('svr',SVR()),
    ('random forest',RandomForestRegressor()),
    ('linear regresyon',LinearRegression())
    ]

stacked = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5)

for name, model in base_models:
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    msn = mean_squared_error(y_test,predict,squared=False)
    print('----------{}------',format(name))
    print('determinant(r^2) katsayısı{} ',format(r2))
    print('msn katsayısı {}',format(msn))
    print('----------------\n')

stacked.fit(X_train,y_train)
stacked_prediction = stacked.predict(X_test)
stacked_r2= stacked.score(X_test, y_test)
stacked_msn = mean_squared_error(y_test, stacked_prediction,squared=False)

print('---stacked ensemble--')
print('determinant katsayısı (r^2) {}',format(stacked_r2))
print('Kök Ortalama Kare Hatası (rmse) {}',format(stacked_msn))
print('------------')