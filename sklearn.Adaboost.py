from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

filepath="/Users/colin/Downloads/iris_binaryenc.csv"
df=pd.read_csv(filepath,header=None)

X = df.iloc[:, 0:4]

df['y']=[ 0 for i in range(150)]
for i in range(50):
    df.iloc[i,5]=1
for i in range(50,150):
    df.iloc[i,5]=0
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

clf = AdaBoostClassifier(n_estimators=30, random_state=0)

clf.fit(X_train, y_train)

# print(clf.predict([[5, 3, 1, 2]]))

score = clf.score(X_test, y_test)
print(f'score:{score}')
