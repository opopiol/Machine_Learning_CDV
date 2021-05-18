from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
wine = load_wine()

features, target = wine.data, wine.target
X, y = features, target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_train, y_train)

dummy_clf.fit(X_test, y_test)
dummy_clf.score(X_test, y_test)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
neigh.score(X_train, y_train)

for n in list(range(1,11)):
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(X_train, y_train)
    ac = neigh.score(X_train, y_train)
    print(f'Score for {n} n_neighbors is {ac}')
    plt.plot(neigh.score(X_train, y_train), "-o")
    plt.plot(n, "-o")


for n in range(1,11):
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(X_train, y_train)
    ac = neigh.score(X_train, y_train)
    plt.plot(ac)

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()

standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
neigh.score(X_train, y_train)

for n in range(1,11):
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(X_train, y_train)
    neigh.score(X_train, y_train)
    print(f'Score for {n} n_neighbors is  {neigh.score(X_train, y_train)}')