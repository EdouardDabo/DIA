import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


iris_data = pd.read_csv("Iris_Dataset.csv")

print(iris_data.head())

print(iris_data.info())

print("Mising values :\n", iris_data.isnull().sum())

X = iris_data.drop('species', axis=1)
y = iris_data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))
print("\nclassification report :\n", classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
grid_model = GridSearchCV(model, param_grid, refit=True, verbose=3)
grid_model.fit(X_train, y_train)

print("Best hyperparameters :", grid_model.best_params_)

y_pred_test = grid_model.predict(X_test)

print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification report :\n", classification_report(y_test, y_pred_test))

#plot the confusion matrix
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#plot the feature importance
feature_importance = pd.Series(grid_model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.show()


#save the model
import pickle
pickle.dump(grid_model, open('model.pkl', 'wb'))


