import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import metrics

iris = datasets.load_iris()

X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.15, random_state=2)

#SVC for classification, #SVR for regression
model = SVC(kernel='rbf')
#print model
model.fit(X_train, Y_train)

Y_predict = model.predict(X_test)
print Y_predict
print Y_test

print model.score(X_test, Y_test)

# print metrics.classification_report(Y_test, Y_predict)
# print metrics.confusion_matrix(Y_test, Y_predict)
