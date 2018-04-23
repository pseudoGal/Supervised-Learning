#Naive Bayes - Classification algorithm of supervised learning 
# this program is just an attempt to understand Naive bayes using GaussianNB model

#load the iris data
from sklearn.datasets import load_iris
iris = load_iris()

#store the feature matrix(x) & response vector (y)
X = iris.data
Y = iris.target

#splitting X and Y into training and testing sets
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)

#traning the model on traning set 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

#making predictions on the testing set
y_pred = gnb.predict(X_test)

#comparing actual response values (y_test) with predicted response values(y_pred)
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy (in %)", metrics.accuracy_score(y_test, y_pred)*100)
