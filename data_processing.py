from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def load_data(dataset_name, test_size=0.2, random_state=0):
    if dataset_name == 'abalone_2class':
        filepath="data/abalone.data"
        df=pd.read_csv(filepath,header=None)
        df.columns = [ 'Sex', 'Length', 'Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
        df = df.replace('M',0)
        df = df.replace('F',1)
        df = df.replace('I',-1)

        df['Categories'] = pd.cut(df['Rings'], bins = [-1,9,1000], labels=[-1,1])

        n = 2000
        posN= int(n*0.8)
        negN= int(n*0.2)

        random.seed(1)
        pos_index = random.sample(list(df.query('Categories == 1').index),posN)

        neg_index = random.sample(list(df.query('Categories == -1').index),negN)
        samples_index = pos_index + neg_index
        data = df.iloc[samples_index]
        X = np.asarray(data.iloc[:, 0:7])
        y = np.asarray(data['Categories'])

    elif dataset_name == 'abalone':
        filepath="data/abalone.data"
        df=pd.read_csv(filepath,header=None)
        df.columns = [ 'Sex', 'Length', 'Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
        df = df.replace('M',0)
        df = df.replace('F',1)
        df = df.replace('I',-1)

        X = np.asarray(df.iloc[:, 0:7])

        df['Categories'] = pd.cut(df['Rings'], bins = [-1,7,10, 15,1000], labels=[1,2,3,4])
        y = np.asarray(df['Categories'])

    elif dataset_name == 'car':
        df = pd.read_csv('data/car.data', sep=',', header=None)
        mapping = {'acc': 1, 'good': 2, 'unacc': 0, 'vgood': 3}
        df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
        enc = OneHotEncoder()
        enc.fit(np.asarray(df.iloc[:, :-1]))
        X = np.asarray(enc.transform(np.asarray(df.iloc[:, :-1])).toarray())
        y = np.asarray(df.iloc[:, -1])  

    elif dataset_name == 'car_2class':
        df = pd.read_csv('data/car.data', sep=',', header=None)
        mapping = {'acc': 0, 'good': 0, 'unacc': 0, 'vgood': 1}
        df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
        enc = OneHotEncoder()
        enc.fit(np.asarray(df.iloc[:, :-1]))
        X = np.asarray(enc.transform(np.asarray(df.iloc[:, :-1])).toarray())
        y = np.asarray(df.iloc[:, -1])

    elif dataset_name == 'tic-tac-toe':
        df = pd.read_csv('data/tic-tac-toe.data', sep=',', header=None)
        mapping = {'negative': 0, 'positive': 1}
        df.iloc[:, -1] = df.iloc[:, -1].map(mapping)
        enc = OneHotEncoder()
        enc.fit(np.asarray(df.iloc[:, :-1]))
        X = np.asarray(enc.transform(np.asarray(df.iloc[:, :-1])).toarray())
        y = np.asarray(df.iloc[:, -1])   

    elif dataset_name == 'red_wine':
        df = pd.read_csv('data/winequality-red.csv')
        df.quality = np.where(df.quality < 6.5, 0, 1)
        X = np.asarray(df.loc[:, df.columns!='quality'])
        y = np.asarray(df['quality'])     


    elif dataset_name == 'diabetes':
        df = pd.read_csv('data/diabetes.csv')
        X = np.asarray(df.loc[:, df.columns!='Outcome'])
        y = np.asarray(df['Outcome'])
 


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)
    print(Counter(y))
    return X_train, X_test, y_train, y_test
    


