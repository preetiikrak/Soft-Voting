# Import necessary libraries
import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Data\Breast_Cancer.csv")

# Encode categorical variables and target
# Encode 'Status' column as target (0 for 'Dead', 1 for 'Alive')
data['Status'] = data['Status'].apply(lambda x: 1 if x == 'Alive' else 0)

# Updated categorical columns
categorical_columns = ['Race', 'Marital_Status', 'T_Stage', 'N_Stage', '6th_Stage', 'differentiate', 
                       'Grade', 'A_Stage', 'Estrogen_Status', 'Progesterone_Status']

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical columns
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Define features (X) and target (y)
X = data.drop('Status', axis=1)  # Features
y = data['Status']  # Target

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features (standardize them)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use Lazy Predict to quickly compare models
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Display Lazy Predict results
print("Lazy Predict Model Results:")
print(models)

# Define specific models for soft voting
log_clf = LogisticRegression(solver='liblinear', random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
dt_clf = DecisionTreeClassifier(random_state=42)

# Create a Soft Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rf_clf), ('dt', dt_clf)], voting='soft')

# Train the Voting Classifier
voting_clf.fit(X_train, y_train)

# Plot ROC Curve for individual models and the voting classifier
for clf in (log_clf, rf_clf, dt_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{clf.__class__.__name__} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Calibration Plot for each classifier
for clf in (log_clf, rf_clf, dt_clf, voting_clf):
    clf.fit(X_train, y_train)
    prob_pos = clf.predict_proba(X_test)[:, 1]
    
    # Use CalibrationDisplay.from_predictions to plot the calibration curve
    disp = CalibrationDisplay.from_predictions(y_test, prob_pos, n_bins=10)
    plt.title(f'Calibration Curve ({clf.__class__.__name__})')

plt.show()

# Bagging Classifier using the Voting Classifier
bagging_clf = BaggingClassifier(estimator=voting_clf, n_estimators=10, random_state=42)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)


# Evaluate Bagging Classifier with ROC Curve
for clf in (log_clf, rf_clf, dt_clf, voting_clf):
    clf.fit(X_train, y_train)
    RocCurveDisplay.from_estimator(clf, X_test, y_test, name=clf.__class__.__name__)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Bagging Classifier')
plt.legend(loc="lower right")
plt.show()