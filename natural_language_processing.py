import pandas as pd

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.linear_model import LogisticRegression # Lohistic Regression
from sklearn.neighbors import KNeighborsClassifier # k-NN
from sklearn.svm import SVC # SVC
from sklearn.tree import DecisionTreeClassifier # DT
from sklearn.ensemble import RandomForestClassifier # RF

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
naiveBayesClassifier = GaussianNB()
naiveBayesClassifier.fit(X_train, y_train)
naiveBayesy_pred = naiveBayesClassifier.predict(X_test)
naiveBayesCM = confusion_matrix(y_test, naiveBayesy_pred)
A_naiveBayesACC = accuracy_score(y_test, naiveBayesy_pred)
R_naiveBayesReport = classification_report(y_test, naiveBayesy_pred)

# Logistic Regression
logisticRegressionClassifier = LogisticRegression(random_state = 0)
logisticRegressionClassifier.fit(X_train, y_train)
logisticRegressiony_pred = logisticRegressionClassifier.predict(X_test)
logisticRegressionCM = confusion_matrix(y_test, logisticRegressiony_pred)
A_logisticRegressionACC = accuracy_score(y_test, logisticRegressiony_pred)
R_logisticRegressionReport = classification_report(y_test, logisticRegressiony_pred)

# k-NN
KNNeighborsClassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNNeighborsClassifier.fit(X_train, y_train)
KNNeighborsy_pred = KNNeighborsClassifier.predict(X_test)
KNeighborsCM = confusion_matrix(y_test, KNNeighborsy_pred)
A_KNeighborsACC = accuracy_score(y_test, KNNeighborsy_pred)
R_KNeighborsReport = classification_report(y_test, KNNeighborsy_pred)

# SVM
SVclassifier = SVC(kernel = 'linear', random_state = 0)
SVclassifier.fit(X_train, y_train)
SVy_pred = SVclassifier.predict(X_test)
SVCM = confusion_matrix(y_test, SVy_pred)
A_SVACC = accuracy_score(y_test, SVy_pred)
R_SVReport = classification_report(y_test, SVy_pred)

# Kernel SVM
KSVclassifier = SVC(kernel = 'rbf', random_state = 0)
KSVclassifier.fit(X_train, y_train)
KSVy_pred = KSVclassifier.predict(X_test)
KSVCM = confusion_matrix(y_test, KSVy_pred)
A_KSVACC = accuracy_score(y_test, KSVy_pred)
R_KSVReport = classification_report(y_test, KSVy_pred)

# Decision Tree
DTclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTclassifier.fit(X_train, y_train)
DTy_pred = DTclassifier.predict(X_test)
DTCM = confusion_matrix(y_test, DTy_pred)
A_DTACC = accuracy_score(y_test, DTy_pred)
R_DTReport = classification_report(y_test, DTy_pred)

# Random Forest
RFclassifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
RFclassifier.fit(X_train, y_train)
RFy_pred = RFclassifier.predict(X_test)
RFCM = confusion_matrix(y_test, RFy_pred)
A_RFACC = accuracy_score(y_test, RFy_pred)
R_RFReport = classification_report(y_test, RFy_pred)