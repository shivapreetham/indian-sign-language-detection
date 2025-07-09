import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the feature data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Loaded {len(data)} samples with {len(data[0])} features each")
print(f"Classes: {set(labels)}")
print(f"Class distribution: {np.unique(labels, return_counts=True)}")

# Split the data into training and test sets (using stratification)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Scale features for better performance
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Define hyperparameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20, None],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
print("Starting grid search...")
grid_search.fit(x_train_scaled, y_train)

# Retrieve the best estimator
best_rf = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Evaluate on validation set with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []

print("\nCross-validation scores:")
for fold, (train_idx, val_idx) in enumerate(cv.split(x_train_scaled, y_train)):
    X_train_fold, X_val_fold = x_train_scaled[train_idx], x_train_scaled[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    best_rf.fit(X_train_fold, y_train_fold)
    val_pred = best_rf.predict(X_val_fold)
    val_acc = accuracy_score(y_val_fold, val_pred)
    val_scores.append(val_acc)
    print(f"Fold {fold+1}: {val_acc*100:.2f}%")

print(f"Average validation accuracy: {np.mean(val_scores)*100:.2f}%")

# Plot validation accuracy per fold
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(val_scores)+1), val_scores, marker='o')
plt.title('Validation Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(range(1, len(val_scores)+1))
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Final training on all training data
best_rf.fit(x_train_scaled, y_train)

# Generate confusion matrix
y_pred = best_rf.predict(x_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', aspect='auto')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(len(np.unique(labels))), np.unique(labels), rotation=45)
plt.yticks(np.arange(len(np.unique(labels))), np.unique(labels))
plt.tight_layout()
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save confusion matrix plot
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', aspect='auto')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(len(np.unique(labels))), np.unique(labels), rotation=45)
plt.yticks(np.arange(len(np.unique(labels))), np.unique(labels))
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Save feature importances
feature_importances = best_rf.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
top_20_idx = sorted_idx[:20]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(20), feature_importances[top_20_idx])
plt.xticks(range(20), top_20_idx, rotation=45)
plt.title('Top 20 Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Save feature importances plot
plt.figure(figsize=(10, 6))
plt.bar(range(20), feature_importances[top_20_idx])
plt.xticks(range(20), top_20_idx, rotation=45)
plt.title('Top 20 Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# Evaluate the best model on the test set
score = accuracy_score(y_test, y_pred)
print(f'{score * 100:.2f}% of samples classified correctly on test data!')

# Save the best model and scaler for inference
with open('model.p', 'wb') as f:
    pickle.dump({
        'model': best_rf,
        'scaler': scaler,
        'feature_importances': feature_importances
    }, f)

print("Training complete. Model saved to 'model.p'")
