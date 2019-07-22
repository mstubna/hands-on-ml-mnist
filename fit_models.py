import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

n_jobs = -1

# load the data set
try:
  mnist
except NameError:
  mnist = fetch_openml('mnist_784', version=1)
  X, y = mnist['data'], mnist['target']
  y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

def plot_roc_curve(actuals, scores, label=None):
  fpr, tpr, _thresholds = roc_curve(actuals, scores)
  plt.plot(fpr, tpr, linewidth=2, label=label)
  plt.plot([0, 1], [0, 1], '--')
  plt.axis([0, 1, 0, 1])
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')

# Stochastic Gradian Descent classifier on a single digit
sgd_clf = SGDClassifier(random_state=42)
sgd_y_train_pred = cross_val_predict(
  estimator=sgd_clf,
  X=X_train,
  y=y_train_5,
  method='decision_function',
  cv=5,
  n_jobs=n_jobs,
  verbose=3
)
plot_roc_curve(y_train_5, sgd_y_train_pred, label='SGD')
print('SGD classifier AUC score:')
print(roc_auc_score(y_train_5, sgd_y_train_pred))
print('SGD classifier precision score:')
print(precision_score(y_train_5, (sgd_y_train_pred > 0)))
print('SGD classifier recall score:')
print(recall_score(y_train_5, (sgd_y_train_pred > 0)))


# Random forest classifier on a single digit
rf_clf = RandomForestClassifier(random_state=42)
rf_y_train_pred = cross_val_predict(
  estimator=rf_clf,
  X=X_train,
  y=y_train_5,
  method='predict_proba',
  cv=5,
  n_jobs=n_jobs,
  verbose=3
)
rf_y_train_scores = rf_y_train_pred[:, 1] # score is proba of positive class
plot_roc_curve(y_train_5, rf_y_train_scores, label='Random forest')
plt.legend(loc='lower right')
plt.show()

print('RF classifier AUC score:')
print(roc_auc_score(y_train_5, rf_y_train_scores))
print('RF classifier precision score:')
print(precision_score(y_train_5, (rf_y_train_scores > 0.5)))
print('RF classifier recall score:')
print(recall_score(y_train_5, (rf_y_train_scores > 0.5)))

# SGD multi-label classifier on all digits
sgd_clf = SGDClassifier(random_state=42)
sgd_y_train_scores = cross_val_score(
  estimator=sgd_clf,
  X=X_train,
  y=y_train,
  scoring='accuracy',
  cv=5,
  n_jobs=n_jobs,
  verbose=3
)
print('SGD multi-label classifier AUC scores:')
print(sgd_y_train_scores)

# KNN multi-label classifier on all digits
knn_clf = KNeighborsClassifier(
  n_neighbors=4,
  p=2,
  weights='distance',
  n_jobs=n_jobs
)
scores = cross_val_score(
  estimator=knn_clf,
  X=X_train,
  y=y_train,
  scoring='accuracy',
  cv=5,
  n_jobs=n_jobs,
  verbose=3
)
print('KNN multi-label classifier cross-validation accuracy scores:')
print(scores)
knn_clf.fit(X_train, y_train)
knn_y_pred = knn_clf.predict(X_test)
print('KNN multi-label classifier accuracy score on test data:')
print(accuracy_score(y_test, knn_y_pred))

#
# Warning - takes a long time to run
#

# KNN multi-label classifier on all digits
# param_grid = [
#   { 'n_neighbors': [3, 4, 5], 'weights': ['uniform', 'distance'] }
# ]
# knn_clf = KNeighborsClassifier()
# grid_search = GridSearchCV(
#   estimator=knn_clf,
#   param_grid=param_grid,
#   scoring='accuracy',
#   return_train_score=True,
#   cv=5,
#   n_jobs=n_jobs,
#   verbose=3
# )
# grid_search.fit(X_train, y_train)
# knn_clf = grid_search.best_estimator_
