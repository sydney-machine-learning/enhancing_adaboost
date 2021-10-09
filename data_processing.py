
import pandas as pd
from sklearn.model_selection import train_test_split
import random

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
        X = data.iloc[:, 0:7]
        y = data['Categories']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)
    return X_train, X_test, y_train, y_test
