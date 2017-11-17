import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics

iris = datasets.load_iris()

#print iris.DESCR
# print iris.keys()
# print iris.feature_names
# print iris.data.shape
# print iris.target.shape
# print iris.data[:5]
# print iris.target[:5]
# print iris.target

X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.15, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

#predicts class labels
Y_predict = model.predict(X_test)

Y_predict_probabilities = model.predict_proba(X_test)
# print Y_predict
# print Y_test

#the default is accuracy score
print model.score(X_test, Y_test)
print metrics.accuracy_score(Y_test, Y_predict)

print Y_test.shape
print Y_predict.shape

#another option is logistic loss, but need to predict probabilities
print metrics.log_loss(Y_test, Y_predict_probabilities)

#also are classification reports and confusion matrices
print metrics.classification_report(Y_test, Y_predict)
print metrics.confusion_matrix(Y_test, Y_predict)

