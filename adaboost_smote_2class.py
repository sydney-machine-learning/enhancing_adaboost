from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.over_sampling import SMOTE
from data_processing import load_data


# Compute error rate, alpha and w
def compute_error(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error):
    return np.log((1 - error) / (error+0.0000001))

def update_weights(w_i, alpha, y, y_pred):
    new_sample_weights = w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
    new_sample_weights /= new_sample_weights.sum()
    return new_sample_weights

#adboost

# Define AdaBoost class
class AdaBoost:
    
    def __init__(self):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []


    def fit(self, X, y, M):
        '''
        M: number of boosting rounds.integer
        '''

        # Clear before calling
        self.alphas = [] 
        self.training_errors = []
        self.M = M

        # Iterate over M weak classifiers
        for m in range(0, M):
            # sampler = SMOTE(random_state= m)
            # sampler = BorderlineSMOTE(random_state= m, kind="borderline-1")
            # sampler = KMeansSMOTE(random_state= m)
            # sampler = SVMSMOTE(random_state= m)
            # X, y = sampler.fit_resample(X, y)

            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                # (d) Update w_i
                w_i = update_weights(w_i, alpha_m, y, y_pred)
            # need to update X here I think make a copy of X and calle smote on the copy of             # X here 
            # (a) Fit weak classifier and predict labels
            G_m = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            
            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error
            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            # (c) Compute alpha
            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)

    def predict(self, X):

        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m

        # Calculate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred

X_train, X_test, y_train, y_test = load_data('abalone_2class',test_size=0.2, random_state=0)
        



# Fit model
ab = AdaBoost()
ab.fit(X_train, y_train, M = 30)

# Predict on test set
y_pred = ab.predict(X_test)

train_err = (y_pred != y_test.reset_index(drop=True)).mean()

print(f'Train error: {train_err:.1%}')
