import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# Uncomment the following line if you wish to use PCA for dimensionality reduction.
# from sklearn.decomposition import PCA

# Load the feature data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and test sets (using stratification)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Scale features for better performance
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Optional: Dimensionality reduction via PCA (uncomment to use)
# pca = PCA(n_components=0.95, svd_solver='full')
# x_train_scaled = pca.fit_transform(x_train_scaled)
# x_test_scaled = pca.transform(x_test_scaled)

# Define hyperparameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(x_train_scaled, y_train)

# Retrieve the best estimator
best_rf = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the best model on the test set
y_predict = best_rf.predict(x_test_scaled)
score = accuracy_score(y_predict, y_test)
print('{}% of samples classified correctly on test data!'.format(score * 100))

# Save the best model, scaler, and PCA (if used) for inference
with open('model.p', 'wb') as f:
    pickle.dump({
        'model': best_rf,
        'scaler': scaler,
        # 'pca': pca  # Uncomment this line if PCA is enabled
    }, f)

print("Training complete. Model saved to 'model.p'")
