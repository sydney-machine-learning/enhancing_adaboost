from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.over_sampling import SMOTE

# Define AdaBoost class
class AdaBoost:
    
    def __init__(self):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        self.learning_rate = 1.0
        self.K=2
        
    # Compute error rate, alpha and w
    def compute_error(self,y, y_pred, w_i):
        return sum(w_i * (np.not_equal(y, y_pred)).astype(int))

    def compute_alpha(self,error,learning_rate):
        return (np.log((1 - error) / error) + np.log(self.K - 1))*learning_rate

    def update_weights(self,w_i, alpha, y, y_pred):
        new_sample_weights = w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
        new_sample_weights /= new_sample_weights.sum()
        return new_sample_weights

    def fit(self, X, y, M,K,learning_rate):
        '''
        M: number of boosting rounds.integer
        '''
        
        # Clear before calling
        self.alphas = [] 
        self.training_errors = []
        self.M = M
        self.K=K
        self.learning_rate=learning_rate
        
        
        w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N 
        # Iterate over M weak classifiers
        for m in range(0, M):
            # sampler = SMOTE(random_state= m)
            # sampler = BorderlineSMOTE(random_state= m, kind="borderline-1")
            # sampler = KMeansSMOTE(random_state= m)
            # sampler = SVMSMOTE(random_state= m)
            # X, y = sampler.fit_resample(X, y)
            
            # (a) Fit weak classifier and predict labels
            G_m = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            
            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error
            error_m = self.compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            # (c) Compute alpha
            alpha_m = self.compute_alpha(error_m,self.learning_rate)
            self.alphas.append(alpha_m)
            
            # (d) Update w_i
            w_i = self.update_weights(w_i, alpha_m, y, y_pred)  # 更新样本权重

        assert len(self.G_M) == len(self.alphas)

    def predict(self, X):

        # Predict class label for each weak classifier, weighted by alpha_m
        F = np.zeros(shape=(len(X),self.K))
        for m in range(self.M):
            y_pred_m = (self.G_M[m].predict(X) == np.array(range(self.K)).reshape(-1,1)).T.astype(np.int8)
            alpha_m = self.alphas[m]
            F += y_pred_m*alpha_m
            

        # Calculate final predictions
        estimator_weights = np.array([self.alphas])
        F /= estimator_weights.sum()
        F /= (self.K -1)
        
        # softmax
        y_pred = np.e**F/((np.e**F).sum(axis = 1).reshape(-1,1))
        y_pred = np.argmax(y_pred, axis=1)

        # print(y_pred.value_counts())
        return y_pred





# Dataset
filepath="data/abalone.data"
df=pd.read_csv(filepath,header=None)
df.columns = [ 'Sex', 'Length', 'Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
df = df.replace('M',0)
df = df.replace('F',1)
df = df.replace('I',-1)

X = df.iloc[:, 0:7]

df['Categories'] = pd.cut(df['Rings'], bins = [-1,7,10, 15,1000], labels=[1,2,3,4])
y = df['Categories']
# print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# Fit model
ab = AdaBoost()
ab.fit(X_train, y_train, M = 30,K=4,learning_rate=1)

# Predict on test set
y_pred = ab.predict(X_test)
train_err = (y_pred != y_test.reset_index(drop=True)).mean()
print(f'Train error: {train_err:.1%}')