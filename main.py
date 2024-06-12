import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Read JSON file into pandas DataFrame
# data = pd.read_json('data.json')
# print(pd.DataFrame.from_dict(data))
# print(data.head())

# Step 2: Data Preprocessing (if needed)
# Perform any preprocessing steps here, such as handling missing values, encoding categorical variables, etc.

# Step 3: Split data into features and target variable
# X = data.drop('target_variable', axis=1)  # Assuming 'target_variable' is the column you want to predict
# y = data['target_variable']

# Step 4: Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and train machine learning model
# model = LinearRegression()
# model.fit(X_train, y_train)

# Step 6: Make predictions (if applicable)
# Use the trained model to make predictions on new data, if needed

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Sample data (replace with actual data loading)
data = {
    "user_id": "U12345",
    "solved_problems": [
        {"problem_id": "P1", "name": "Two Sum", "difficulty": "Easy", "tags": ["Array", "Hash Table"], "time_taken": "10 minutes", "attempts": 1, "hints_used": 0, "submission_status": "Accepted", "submission_count": 1},
        # Add more problems here...
        {"problem_id": "P25", "name": "Valid Sudoku", "difficulty": "Medium", "tags": ["Array", "Hash Table"], "time_taken": "35 minutes", "attempts": 4, "hints_used": 2, "submission_status": "Accepted", "submission_count": 5}
    ],
    "average_difficulty_level": "Medium",
    "preferred_tags": ["Array", "String", "Linked List"],
    "learning_path": ["Easy", "Medium", "Medium", "Hard", "Medium"],
    "performance_trends": "Improving",
    "learning_style": "Balanced"
}

# Convert JSON to DataFrame
problems = pd.json_normalize(data['solved_problems'])

# Feature Engineering
problems['difficulty'] = problems['difficulty'].map({'Easy': 0, 'Medium': 1, 'Hard': 2})
problems['time_taken'] = problems['time_taken'].apply(lambda x: int(x.split()[0]))
problems['tags'] = problems['tags'].apply(lambda x: len(x))

# Define features and labels
X = problems[['difficulty', 'time_taken', 'attempts', 'hints_used', 'submission_count', 'tags']]
y = problems['difficulty']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline
pipeline = Pipeline([
    ('clf', RandomForestClassifier(random_state=42))
])

# Hyperparameters
param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

# Train the model
grid_search.fit(X_train, y_train)

# Predict
y_pred = grid_search.predict(X_test)

# Evaluate
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

