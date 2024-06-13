# /mnt/data/ml_model.py

import pymongo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify
from connector import MongoDBConnector as con

# Connect to MongoDB
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["problem_solving_db"]
# users_collection = db["users"]
# problems_collection = db["problems"]

connector = con()

# Fetch user data
users_data = connector.load_data("userData")
problems_data = connector.load_data("hackrank1")

print(f"Users data: {users_data}")
print(f"Problems data: {problems_data}")

# Data Preprocessing
users_df = pd.DataFrame(users_data)
problems_df = pd.DataFrame(problems_data)

# Feature Engineering


def extract_features(user):

    # print(f"User: {user}")
    solved_problems = user['solved_problems']
    average_difficulty = user['average_difficulty_level']
    preferred_tags = user['preferred_tags']
    learning_path = user['learning_path']
    performance_trends = user['performance_trends']
    learning_style = user['learning_style']

    # Convert categorical to numerical
    label_encoder = LabelEncoder()
    user['average_difficulty_level'] = label_encoder.fit_transform(
        [average_difficulty])[0]
    user['performance_trends'] = label_encoder.fit_transform([performance_trends])[
        0]
    user['learning_style'] = label_encoder.fit_transform([learning_style])[0]

    # Encode preferred tags
    ohe = OneHotEncoder()
    tags_encoded = ohe.fit_transform([preferred_tags]).toarray()[0]

    # Aggregate other features if needed

    features = [
        user['average_difficulty_level'],
        user['performance_trends'],
        user['learning_style']
    ] + list(tags_encoded)

    return features


# Extract features and labels
# X = np.array([extract_features(user) for user in users_data])
X = np.array(extract_features(users_data))
y_avg_diff = np.array([user['average_difficulty_level']
                      for user in users_data])
y_perf_trends = np.array([user['performance_trends'] for user in users_data])
y_learning_style = np.array([user['learning_style'] for user in users_data])

# Split the data
X_train, X_test, y_train_avg_diff, y_test_avg_diff = train_test_split(
    X, y_avg_diff, test_size=0.2, random_state=42)
_, _, y_train_perf_trends, y_test_perf_trends = train_test_split(
    X, y_perf_trends, test_size=0.2, random_state=42)
_, _, y_train_learning_style, y_test_learning_style = train_test_split(
    X, y_learning_style, test_size=0.2, random_state=42)

# Train models


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


model_avg_diff = train_model(X_train, y_train_avg_diff)
model_perf_trends = train_model(X_train, y_train_perf_trends)
model_learning_style = train_model(X_train, y_train_learning_style)

# Evaluate models


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))


evaluate_model(model_avg_diff, X_test, y_test_avg_diff)
evaluate_model(model_perf_trends, X_test, y_test_perf_trends)
evaluate_model(model_learning_style, X_test, y_test_learning_style)

# Save models
joblib.dump(model_avg_diff, 'model_avg_diff.pkl')
joblib.dump(model_perf_trends, 'model_perf_trends.pkl')
joblib.dump(model_learning_style, 'model_learning_style.pkl')

# Suggest a problem based on model predictions


def suggest_problem(user_features, problems_df):
    avg_diff_pred = model_avg_diff.predict([user_features])[0]
    perf_trends_pred = model_perf_trends.predict([user_features])[0]
    learning_style_pred = model_learning_style.predict([user_features])[0]

    # Decode the predictions
    label_encoder = LabelEncoder()
    avg_diff_pred_label = label_encoder.inverse_transform([avg_diff_pred])[0]
    perf_trends_pred_label = label_encoder.inverse_transform([perf_trends_pred])[
        0]
    learning_style_pred_label = label_encoder.inverse_transform(
        [learning_style_pred])[0]

    # Filter problems based on predicted average difficulty level and preferred tags
    filtered_problems = problems_df[problems_df['difficulty_name']
                                    == avg_diff_pred_label]

    # Further filter or sort the problems based on user performance trends and learning style
    if perf_trends_pred_label == "Improving":
        suggested_problem = filtered_problems.sample(n=1).iloc[0]
    else:
        # If performance is declining, suggest an easier problem
        easier_problems = problems_df[problems_df['difficulty_name']
                                      < avg_diff_pred_label]
        suggested_problem = easier_problems.sample(
            n=1).iloc[0] if not easier_problems.empty else filtered_problems.sample(n=1).iloc[0]

    return suggested_problem


# Create Flask API
app = Flask(__name__)


@app.route('/suggest_problem', methods=['POST'])
def suggest_problem_endpoint():
    user_data = request.json
    user_features = extract_features(user_data)
    suggested_problem = suggest_problem(user_features, problems_df)

    response = {
        "average_difficulty_level": suggested_problem['difficulty_name'],
        "performance_trends": suggested_problem['difficulty_name'],
        "learning_style": suggested_problem['difficulty_name'],
        "suggested_problem": {
            "name": suggested_problem['name'],
            "link": suggested_problem['link'],
            "tags": suggested_problem['tag_names']
        }
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
