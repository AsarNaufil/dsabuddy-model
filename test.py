import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from connector import MongoDBConnector as con

# Initialize Flask app
app = Flask(__name__)

# Fetch user and problem data
connector = con()
users_data = connector.get_training_user_data()
problems_data = connector.get_problem_data()

# Convert data to DataFrames
users_df = pd.DataFrame(users_data)
problems_df = pd.DataFrame(problems_data)

# display all columns and values
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(users_df)


def convert_to_dataframe(user_data):
    solved_problems = user_data["solved_problems"][0]
    for problem in solved_problems:
        problem["time_taken"] = int(problem["time_taken"])
        problem["solved_at"] = pd.to_datetime(problem["solved_at"])
    return pd.DataFrame(solved_problems)


def calculate_mode_difficulty(df):
    difficulty_values = {"easy": 1, "medium": 2, "hard": 3}
    df["difficulty_score"] = df["difficulty"].map(difficulty_values)
    return df["difficulty"].mode().values[0]


def predict_performance_trend(df):
    if df.empty:
        return "Insufficient data"
    average_time = df["time_taken"].mean()
    if average_time < 15:
        return "Improving very quickly"
    elif average_time < 30:
        return "Improving steadily"
    else:
        return "Improving slowly"


def assess_learning_style(df_problems):
    if df_problems.empty or len(df_problems) == 0:
        return "Insufficient data"
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    last_week = datetime.today() - timedelta(days=7)
    recent_problems = df_problems[df_problems["solved_at"] >= last_week]
    if recent_problems.empty:
        return "Insufficient data"
    for difficulty_level in difficulty_counts:
        difficulty_counts[difficulty_level] = (
            recent_problems["difficulty"] == difficulty_level).sum()
    max_difficulty = max(difficulty_counts.values())
    learning_style = [
        key for key, value in difficulty_counts.items() if value == max_difficulty]
    return f"Prefers {learning_style[0]} problems" if len(learning_style) == 1 else "Balanced approach"


def extract_features(user):
    df_problems = convert_to_dataframe(user)
    average_difficulty = calculate_mode_difficulty(df_problems)
    performance_trends = predict_performance_trend(df_problems)
    learning_style = assess_learning_style(df_problems)
    preferred_tags = df_problems["tags"].mode().values[0]

    label_encoder = LabelEncoder()
    user['average_difficulty_level'] = label_encoder.fit_transform(
        [average_difficulty])[0]
    user['performance_trends'] = label_encoder.fit_transform(
        [performance_trends])[0]
    user['learning_style'] = label_encoder.fit_transform([learning_style])[0]

    ohe = OneHotEncoder()
    tags_encoded = ohe.fit_transform([preferred_tags]).toarray()[0]

    features = [
        user['average_difficulty_level'],
        user['performance_trends'],
        user['learning_style']
    ] + list(tags_encoded)
    return features


def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return model


X = np.array([extract_features(user) for user in users_data])
y_avg_diff = np.array([user['average_difficulty_level']
                      for user in users_data])
y_perf_trends = np.array([user['performance_trends'] for user in users_data])
y_learning_style = np.array([user['learning_style'] for user in users_data])

model_avg_diff = train_and_evaluate_model(X, y_avg_diff)
model_perf_trends = train_and_evaluate_model(X, y_perf_trends)
model_learning_style = train_and_evaluate_model(X, y_learning_style)

joblib.dump(model_avg_diff, 'model_avg_diff.pkl')
joblib.dump(model_perf_trends, 'model_perf_trends.pkl')
joblib.dump(model_learning_style, 'model_learning_style.pkl')


def suggest_problem(user_features, problems_df):
    avg_diff_pred = model_avg_diff.predict([user_features])[0]
    perf_trends_pred = model_perf_trends.predict([user_features])[0]
    learning_style_pred = model_learning_style.predict([user_features])[0]

    label_encoder = LabelEncoder()
    avg_diff_pred_label = label_encoder.inverse_transform([avg_diff_pred])[0]
    perf_trends_pred_label = label_encoder.inverse_transform(
        [perf_trends_pred])[0]
    learning_style_pred_label = label_encoder.inverse_transform(
        [learning_style_pred])[0]

    filtered_problems = problems_df[problems_df['difficulty_name']
                                    == avg_diff_pred_label]
    if perf_trends_pred_label == "Improving":
        suggested_problem = filtered_problems.sample(n=1).iloc[0]
    else:
        easier_problems = problems_df[problems_df['difficulty_name']
                                      < avg_diff_pred_label]
        suggested_problem = easier_problems.sample(
            n=1).iloc[0] if not easier_problems.empty else filtered_problems.sample(n=1).iloc[0]
    return suggested_problem


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
