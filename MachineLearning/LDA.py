from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  

iris = load_iris()
data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns=feature_names)
df["sınıf"] = y
x = data
print(df.head())

fig, axs = plt.subplots(1, 2)
row, col = 0, 0

# LDA ile 2 bileşene kadar indirgeme
for i in range(1, 3):  
    lda = LDA(n_components=i)  
    x_lda = lda.fit_transform(x, y)  
    explained_variance_ratio = np.var(x_lda, axis=0) / np.var(x, axis=0).sum()  
    print("\nn_components=", i)
    print("Variance ratio=", explained_variance_ratio)
    print("Sum ratio=", sum(explained_variance_ratio))

    axs[col].set(xlabel='Number of Component', ylabel='Cumulative Variance')
    axs[col].plot(np.cumsum(explained_variance_ratio))  
    col += 1
    
df_sns = pd.DataFrame({'variance': explained_variance_ratio, 'LD': ['LD1', 'LD2']})
sns.barplot(x='LD', y='variance', data=df_sns, color="c")
plt.show()
