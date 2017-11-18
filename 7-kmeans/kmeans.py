import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

iris = datasets.load_iris()
print iris.feature_names

#just taking two features here
X = iris.data[:, 1:3]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

model = KMeans(n_clusters=8, random_state=0)
model.fit(X_train)

print model.labels_
print model.cluster_centers_
print model.inertia_

predictions = model.predict(X)

centroids = model.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='^', s=169, linewidths=3,
            color='m', zorder=10)

plt.scatter(X[:, 0], X[:, 1], c=predictions)
plt.xlabel("Sepal width")
plt.ylabel("Petal length")
plt.show()


