import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

#SVC for classification, #SVR for regression
#can change kernels
model = SVC(kernel='rbf')
#print model
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print y_predict
print y_test

print model.score(X_test, y_test)

# print metrics.classification_report(y_test, y_predict)
# print metrics.confusion_matrix(y_test, y_predict)
