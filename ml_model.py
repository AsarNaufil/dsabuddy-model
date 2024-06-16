import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from connector import MongoDBConnector as con

# Fetch user data

connector = con()

# TODO: fetch me daddy
user_id = "u123456"
users_data = connector.get_training_user_data(user_id)
problems_data = connector.get_problem_data()

# print(f"Users data: {users_data["solved_problems"][0][0].keys()}")
# print(f"Problems data: {problems_data}")

# Data Preprocessing
users_df = pd.DataFrame(users_data)
problems_df = pd.DataFrame(problems_data)

# Feature Engineering

TOPICS = ["array", "hash table", "string", "math", "stack", "linked list",
          "union find", "bit manipulation", "combinatorics", "divide and conquer",
          "merge sort", "segment tree", "binary indexed tree", "ordered set",
          "matrix", "binary search", "recursion", "sorting", "two pointers",
          "sliding window", "dynamic programming"]


def calculate_mode_difficulty(df):
    # return the difficulty score with the highest frequency
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
    # Define difficulty levels and initialize counts
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    # Calculate the start date for the time period (e.g., last week)
    today = datetime.today()
    last_week = today - timedelta(days=7)
    # Filter solved problems within the last week
    recent_problems = df_problems[df_problems["solved_at"] >= last_week]
    if recent_problems.empty or len(recent_problems) == 0:
        return "Insufficient data"
    # Count occurrences of each difficulty level in the recent problems
    for difficulty_level in difficulty_counts:
        difficulty_counts[difficulty_level] = (
            recent_problems["difficulty"] == difficulty_level).sum()
    # Determine learning style based on counts
    max_difficulty = max(difficulty_counts.values())
    learning_style = [
        key for key, value in difficulty_counts.items() if value == max_difficulty]
    if len(learning_style) == 1:
        return f"Prefers {learning_style[0]} problems"
    else:
        return "Balanced approach"


def convert_to_dataframe(user_data):

    # print(user_data["solved_problems"][0][0])
    # print(type(solved_problems[0][0]))  # Here's your dict
    solved_problems = user_data["solved_problems"][0]

    if not solved_problems:
        return pd.DataFrame()

    for problem in solved_problems:
        # sanitize time_taken
        time_taken = problem["time_taken"]
        problem["time_taken"] = int(time_taken.split()[0])

        # sanitize solved_at
        solved_at = problem["solved_at"]
        problem["solved_at"] = pd.to_datetime(solved_at)

        # reverse the to_datetime to ISOdate format
        # problem["solved_at"] = pd.to_datetime(solved_at).to_pydatetime()

    return pd.DataFrame(solved_problems)


def suggest_next_difficulty(df):

    if df.empty or len(df) < 10:
        return "easy"

    difficulty_values = {"easy": 1, "medium": 2, "hard": 3}
    reverse_difficulty_values = {1: "easy", 2: "medium", 3: "hard"}

    df["difficulty_score"] = df["difficulty"].map(difficulty_values)
    mode_difficulty_score = df["difficulty_score"].mode().iloc[0]

    # Calculate average performance metrics
    avg_attempts = df["attempts"].mean()
    avg_hints_used = df["hints_used"].mean()
    avg_time_taken = df["time_taken"].mean()

    # Heuristic to suggest next difficulty based on current performance
    if avg_attempts <= 2 and avg_hints_used <= 1 and avg_time_taken <= 30:
        next_difficulty_score = min(mode_difficulty_score + 1, 3)
    else:
        next_difficulty_score = mode_difficulty_score

    performance_trend = predict_performance_trend(df)

    # Adjust next_difficulty_score based on performance trend
    if performance_trend == "Improving very quickly":
        next_difficulty_score = min(next_difficulty_score + 2, 3)
    elif performance_trend == "Improving steadily":
        next_difficulty_score = min(next_difficulty_score + 1, 3)
    elif performance_trend == "Improving slowly":
        pass  # Maintain the calculated next_difficulty_score
    else:
        # Default to maintaining the current difficulty if performance trend is not clear
        pass

    # Assess learning style to further refine the suggestion
    learning_style_result = assess_learning_style(df)
    if "Prefers medium problems" in learning_style_result:
        next_difficulty_score = 2
    elif "Prefers hard problems" in learning_style_result:
        next_difficulty_score = 3
    elif "Balanced approach" in learning_style_result:
        pass  # Maintain the calculated next_difficulty_score

    next_difficulty = reverse_difficulty_values[next_difficulty_score]

    return next_difficulty


def suggest_next_tags(df):

    if df.empty:
        return ["array", "hash table", "string"]

    tags_data = df.explode('tags')

    # Calculate performance metrics per tag
    tag_performance = tags_data.groupby('tags').agg({
        'attempts': 'mean',
        'hints_used': 'mean',
        'time_taken': 'mean',
        'submission_status': lambda x: (x == 'accepted').mean()  # Success rate
    }).reset_index()

    # Define thresholds for good performance
    thresholds = {
        'attempts': 2,
        'hints_used': 1,
        'time_taken': 30,
        'success_rate': 0.75  # At least 75% success rate
    }

    # Filter tags with good performance
    preferred_tags = tag_performance[
        (tag_performance['attempts'] <= thresholds['attempts']) &
        (tag_performance['hints_used'] <= thresholds['hints_used']) &
        (tag_performance['time_taken'] <= thresholds['time_taken']) &
        (tag_performance['submission_status'] >= thresholds['success_rate'])
    ]['tags'].tolist()

    # If no tags meet the criteria, fall back to most frequent tags
    if not preferred_tags:
        preferred_tags = tags_data['tags'].value_counts().head(
            3).index.tolist()

    return preferred_tags


def extract_features(user):

    # print(f"User: {user['solved_problems'][0][0]}")
    df_problems = convert_to_dataframe(user)
    average_difficulty = calculate_mode_difficulty(df_problems)
    performance_trends = predict_performance_trend(df_problems)
    learning_style = assess_learning_style(df_problems)
    preferred_tags = df_problems["tags"].mode().values[0]

    # print(f"Average difficulty: {average_difficulty}")
    # print(f"Performance trends: {performance_trends}")
    # print(f"Learning style: {learning_style}")
    # print(f"Preferred tags: {preferred_tags}")

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


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))


def train_and_evaluate_models(X, y_avg_diff, y_perf_trends, y_learning_style):

    X = np.array(extract_features(users_data))
    y_avg_diff = np.array([user['average_difficulty_level']
                          for user in users_data])
    y_perf_trends = np.array([user['performance_trends']
                             for user in users_data])
    y_learning_style = np.array([user['learning_style']
                                for user in users_data])

    # Split the data
    X_train, X_test, y_train_avg_diff, y_test_avg_diff = train_test_split(
        X, y_avg_diff, test_size=0.2, random_state=42)
    _, _, y_train_perf_trends, y_test_perf_trends = train_test_split(
        X, y_perf_trends, test_size=0.2, random_state=42)
    _, _, y_train_learning_style, y_test_learning_style = train_test_split(
        X, y_learning_style, test_size=0.2, random_state=42)

    model_avg_diff = train_model(X_train, y_train_avg_diff)
    model_perf_trends = train_model(X_train, y_train_perf_trends)
    model_learning_style = train_model(X_train, y_train_learning_style)

    evaluate_model(model_avg_diff, X_test, y_test_avg_diff)
    evaluate_model(model_perf_trends, X_test, y_test_perf_trends)
    evaluate_model(model_learning_style, X_test, y_test_learning_style)

    # Save models
    joblib.dump(model_avg_diff, 'model_avg_diff.pkl')
    joblib.dump(model_perf_trends, 'model_perf_trends.pkl')
    joblib.dump(model_learning_style, 'model_learning_style.pkl')


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


def problem(user):

    df_problems = convert_to_dataframe(user)
    # average_difficulty = calculate_mode_difficulty(df_problems)
    # performance_trends = predict_performance_trend(df_problems)
    # learning_style = assess_learning_style(df_problems)
    # preferred_tags = df_problems["tags"].mode().values[0]

    next_difficulty = suggest_next_difficulty(df_problems)
    next_tags = suggest_next_tags(df_problems)

    print(f"Next difficulty: {next_difficulty}")
    print(f"Next tags: {next_tags}")

    query = {
        "difficulty": next_difficulty,
        "tags": {"$in": next_tags}
    }

    result = list(connector.search_problems(query))[0]

    # "$meta": "textScore",
    # result = result.sort([
    #     ("tag_names", {
    #         "$in": next_tags
    #     }),
    #     ("tag_names", {"$size": -1})
    # ]).limit(1)
    #
    # result = result[0] if result else None

    print(f"Suggested problem: ")
    for key, value in result.items():
        print(f"{key}: {value}")
    return result


problem(users_data)


# Create Flask API
app = Flask(__name__)


@ app.route('/suggest_problem', methods=['POST'])
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
