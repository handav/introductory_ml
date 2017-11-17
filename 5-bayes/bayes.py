import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from pandas_ml import ConfusionMatrix


newsgroups_train = datasets.fetch_20newsgroups(subset='train')
newsgroups_test = datasets.fetch_20newsgroups(subset='test')

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)

y_train = newsgroups_train.target
y_test = newsgroups_test.target

print y_test

model = MultinomialNB()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print y_predict.shape

print model.score(X_test, y_test)

print metrics.classification_report(y_test, y_predict)
#print metrics.confusion_matrix(y_test, y_predict)

labels = list(newsgroups_train.target_names)
cm = ConfusionMatrix(y_test, y_predict, labels)
cm.plot()
plt.show()


