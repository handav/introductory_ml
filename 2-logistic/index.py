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
y = iris.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.15, random_state=2)

model = LogisticRegression()
model.fit(X_train, y_train)

#predicts class labels
y_predict = model.predict(X_test)

y_predict_probabilities = model.predict_proba(X_test)
# print y_predict
# print y_test

#the default is accuracy score
print model.score(X_test, y_test)
print metrics.accuracy_score(y_test, y_predict)

print y_test.shape
print y_predict.shape

#another option is logistic loss, but need to predict probabilities
print metrics.log_loss(y_test, y_predict_probabilities)

#also are classification reports and confusion matrices
print metrics.classification_report(y_test, y_predict)
print metrics.confusion_matrix(y_test, y_predict)

