import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn import metrics


boston = datasets.load_boston()
print boston.keys()
print boston.DESCR
print boston.feature_names
print boston.data.shape
print boston.target.shape

X = boston.data
Y = boston.target

#create training and test data
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.15, random_state=2)

model = LinearRegression()
model.fit(X_train, Y_train)

test_house = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
print model.predict(test_house)

Y_predict = model.predict(X_test)

#this is the r2 score
model_score = model.score(X_test, Y_test)
print model_score

#an alternate way to score is with mean squared error
mse = metrics.mean_squared_error(Y_test, Y_predict)
print(mse)

plt.scatter(Y_test, Y_predict)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted prices")
plt.show()




