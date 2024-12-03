import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Load the dataset
raw_data = pd.read_csv(r"ai4i2020 (1).csv")

# Prepare the data
X = raw_data.drop(columns=['UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']).values
y = raw_data['Machine failure'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning for Logistic Regression
log_reg_params = {
    'penalty': ['l1', 'l2'],  # Regularization type
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'solver': ['liblinear', 'saga']  # Optimization algorithms
}

# Create GridSearchCV for Logistic Regression
log_reg_grid = GridSearchCV(
    LogisticRegression(random_state=420, max_iter=1000), 
    log_reg_params, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1  # Use all available cores
)

# Fit GridSearchCV
log_reg_grid.fit(X_train, y_train)

# Print best parameters and score for Logistic Regression
print("Best Logistic Regression Parameters:")
print(log_reg_grid.best_params_)
print(f"Best Cross-Validation Score: {log_reg_grid.best_score_:.4f}")

# Get the best Logistic Regression model
best_log_reg = log_reg_grid.best_estimator_

# Predict and evaluate Logistic Regression
y_pred_log_reg = best_log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print("\nLogistic Regression Test Accuracy:", log_reg_accuracy)
print("\nLogistic Regression Confusion Matrix:")
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
print(cm_log_reg)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# Visualize Logistic Regression Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Hyperparameter Tuning for Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required to be at a leaf node
}

# Create GridSearchCV for Random Forest
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    rf_params, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1  # Use all available cores
)

# Fit GridSearchCV
rf_grid.fit(X_train, y_train)

# Print best parameters and score for Random Forest
print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)
print(f"Best Cross-Validation Score: {rf_grid.best_score_:.4f}")

# Get the best Random Forest model
best_rf = rf_grid.best_estimator_

# Predict and evaluate Random Forest
y_pred_rf = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Test Accuracy:", rf_accuracy)
print("\nRandom Forest Confusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Visualize Random Forest Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance for Random Forest (optional)
feature_names = raw_data.drop(columns=['UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']).columns
feature_importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importances in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

print("\nTop 5 Most Important Features:")
print(feature_importance_df.head())

# Extract the coefficients from the Logistic Regression model
log_reg_coefficients = best_log_reg.coef_.flatten()  # Flatten to 1D array
log_reg_feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': log_reg_coefficients  # Use absolute value of coefficients
}).sort_values('importance', ascending=False)




# Visualize the Logistic Regression Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=log_reg_feature_importance)
plt.title('Feature Importance in Logistic Regression Model')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Print the top 5 most important features for Logistic Regression
print("\nTop 5 Most Important Features in Logistic Regression:")
print(log_reg_feature_importance.head())


# ROC Curve for Logistic Regression
#y_pred_prob_log_reg = best_log_reg.predict_proba(X_test)[:, 1]
#fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_pred_prob_log_reg)
#roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)
#
#plt.figure(figsize=(8, 6))
#plt.plot(fpr_log_reg, tpr_log_reg, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_log_reg:.2f})')
#plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend(loc='lower right')
#plt.show()
#
## Precision-Recall Curve for Logistic Regression
##precision_log_reg, recall_log_reg, _ = precision_recall_curve(y_test, y_pred_prob_log_reg)
#
#plt.figure(figsize=(8, 6))
#plt.plot(recall_log_reg, precision_log_reg, color='blue', lw=2, label='Logistic Regression')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall (PR) Curve')
#plt.legend(loc='lower left')
#plt.show()
#
## ROC Curve for Random Forest
#y_pred_prob_rf = best_rf.predict_proba(X_test)[:, 1]
#fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
#roc_auc_rf = auc(fpr_rf, tpr_rf)
#
#plt.figure(figsize=(8, 6))
#plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
#plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend(loc='lower right')
#plt.show()
#
## Precision-Recall Curve for Random Forest
#precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_prob_rf)
#
#plt.figure(figsize=(12, 6))
#
## Plot ROC Curve
#plt.subplot(1, 2, 1)
#plt.plot(fpr_log_reg, tpr_log_reg, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_log_reg:.2f})')
#plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
#plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
#plt.legend(loc='lower right')
#
## Plot Precision-Recall Curve
#plt.subplot(1, 2, 2)
#plt.plot(recall_log_reg, precision_log_reg, color='blue', lw=2, label='Logistic Regression')
#plt.plot(recall_rf, precision_rf, color='green', lw=2, label='Random Forest')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall (PR) Curve')
#plt.legend(loc='lower left')
#
#plt.tight_layout()
#plt.show()