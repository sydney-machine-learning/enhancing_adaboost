# Revived the following when I ran this:
# 441: UserWarning: X does not have valid feature names, but AdaBoostClassifier # was fitted with feature names
#
#

from sklearn.ensemble import AdaBoostClassifier
from data_processing import load_data

X_train, X_test, y_train, y_test = load_data('abalone_2class',test_size=0.2, random_state=0)

clf = AdaBoostClassifier(n_estimators=30, random_state=0, algorithm="SAMME")

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)
train_err = (y_pred != y_test.reset_index(drop=True)).mean()

print(f'Train error: {train_err:.1%}')
