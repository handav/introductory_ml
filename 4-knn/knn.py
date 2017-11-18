import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

#can change the number of neighbors here
model = neighbors.KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train) 

#this is accuracy
print model.score(X_test, y_test)

y_predict = model.predict(X_test)
#classification reports and confusion matrices
print metrics.classification_report(y_test, y_predict)
print metrics.confusion_matrix(y_test, y_predict)
