from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd

# load iris dataset
iris = load_iris()

# convert the dataset into dataframe
iris_dataf = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# add target variable(species) to dataframe
iris_dataf['species'] = iris.target

# map target variable to corresponding species
iris_dataf['species'] = iris_dataf['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# display first few rows of the dataframe
print("Original Dataset:")
print(iris_dataf.head())

# Split the dataset into features(X) and target variable(Y)
X = iris_dataf.drop(columns=['species'])  
Y = iris_dataf['species'] 

# Split data into training and testing sets (70% training, 30% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Initialise decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train decision tree classifier on the training data(70%)
clf.fit(X_train, Y_train)


Y_pred = clf.predict(X_test)

# performance of the model(metrics)
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')

# display metrics before optimisation, regularisation, and cross-validation
print("\nEvaluation Metrics Before Optimisation, Regularisation, and Cross-Validation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# classification report
print("\nClassification Report Before Optimization, Regularization, and Cross-Validation:")
print(classification_report(Y_test, Y_pred))

# Perform 5-fold cross-validation
scores = cross_val_score(clf, X, Y, cv=5)

# Print the cross-validation scores before any optimization, regularization, or cross-validation
print("\nCross-Validation Scores Before Optimization, Regularization, and Cross-Validation:")
print(scores)
print("Average Cross-Validation Score Before Optimization, Regularization, and Cross-Validation:", scores.mean())

# Define the hyperparameters and their possible values
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, Y)

# Print the best hyperparameters and corresponding score
print("\nBest Hyperparameters After Grid Search Optimization:")
print(grid_search.best_params_)
print("Best Score After Grid Search Optimization:", grid_search.best_score_)

# Get the cross-validation scores after optimization
cv_scores_after_optimization = cross_val_score(grid_search.best_estimator_, X, Y, cv=5)

# Print the cross-validation scores after optimization
print("\nCross-Validation Scores After Grid Search Optimization:")
print(cv_scores_after_optimization)
print("Average Cross-Validation Score After Grid Search Optimization:", cv_scores_after_optimization.mean())

# Initialize the decision tree classifier with regularization parameters
clf_regularized = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)

# Train the regularized decision tree classifier
clf_regularized.fit(X_train, Y_train)

# Predict the labels of the test set using regularized model
Y_pred_regularized = clf_regularized.predict(X_test)

# Evaluate the regularized model on the test set
accuracy_regularized = accuracy_score(Y_test, Y_pred_regularized)

# Display the evaluation metrics after regularization
print("\nEvaluation Metrics After Regularization:")
print("Regularized Model Accuracy:", accuracy_regularized)