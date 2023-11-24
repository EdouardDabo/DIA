# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# Load the Telco Customer Churn dataset from an Excel file (replace 'your_dataset.xlsx' with the actual file name)
data = pd.read_excel('Telco_customer_churn.xlsx')

# Explore the dataset
print(data.head())

# Identify features and target variable
X = data.drop(['Churn Label', 'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason'], axis=1)  # Features
y = data['Churn Label']  # Use the appropriate column as the target variable

"""
X = data.drop('Churn', axis=1)  # Features
y = data['Churn']  # Target variable
"""
# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

 #show graphs of the final model training and evaluation.
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the confusion matrix
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot the feature importance
feature_importance = pd.Series(rf_classifier.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.show()

# Plot the ROC curve
from sklearn.metrics import roc_curve, roc_auc_score

# Get the probability scores for each point in the dataset
y_scores = rf_classifier.predict_proba(X_test)[:, 1]

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Convert 'No' and 'Yes' to 0 and 1 in y_test
y_test_binary = label_encoder.fit_transform(y_test)

# Get the probability scores for each point in the dataset
y_scores = rf_classifier.predict_proba(X_test)[:, 1]

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_scores)

# Compute the AUC score
roc_auc = roc_auc_score(y_test_binary, y_scores)
print(f'AUC Score: {roc_auc}')

# Plot the ROC curve with AUC score
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Save the model
import pickle

pickle.dump(rf_classifier, open('churn_model.pkl', 'wb'))




