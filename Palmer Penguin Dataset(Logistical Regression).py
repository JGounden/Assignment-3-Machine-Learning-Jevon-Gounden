import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings



# suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

# load the Palmer Penguin Dataset from seaborn
penguins = sns.load_dataset('penguins')

# Drop rows with missing values
penguins.dropna(inplace=True)

# Convert categorical variables to numerical
penguins = pd.get_dummies(penguins, drop_first=True)

# Display original dataset
print("Original Dataset:")
print(penguins.head())

# Split the dataset into features (X) and target variable (y)
X = penguins.drop('species_Chinstrap', axis=1)  # Features
y = penguins['species_Chinstrap']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training set and get its performance before hyperparameter optimization
model.fit(X_train, y_train)
y_pred_before = model.predict(X_test)

# Evaluate the model's performance before hyperparameter optimization
accuracy_before = accuracy_score(y_test, y_pred_before)
precision_before = precision_score(y_test, y_pred_before)
recall_before = recall_score(y_test, y_pred_before)
f1_before = f1_score(y_test, y_pred_before)

print("\nModel Performance Before Hyperparameter Optimization:")
print("Accuracy:", accuracy_before)
print("Precision:", precision_before)
print("Recall:", recall_before)
print("F1-score:", f1_before)

# Define the parameter grid for Grid Search
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
}

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Initialize Grid Search with cross-validation (5 folds)
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy'
)

# Fit Grid Search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by Grid Search
best_params = grid_search.best_params_
print("\nBest Hyperparameters:", best_params)

# Evaluate the model's performance with the best hyperparameters
best_model = grid_search.best_estimator_
y_pred_after = best_model.predict(X_test)

# Calculate performance metrics after hyperparameter optimization
accuracy_after = accuracy_score(y_test, y_pred_after)
precision_after = precision_score(y_test, y_pred_after)
recall_after = recall_score(y_test, y_pred_after)
f1_after = f1_score(y_test, y_pred_after)

# Print the model's performance after hyperparameter optimization
print("\nModel Performance After Hyperparameter Optimization:")
print("Accuracy:", accuracy_after)
print("Precision:", precision_after)
print("Recall:", recall_after)
print("F1-score:", f1_after)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_after))
