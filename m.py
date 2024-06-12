
import pandas as pd
import json

# Load JSON data
with open('data.json', 'r') as f:
    data = json.load(f)

# Flatten the 'solved_problems' array
flattened_data = []
for problem in data['solved_problems']:
    flat_problem = {key: value for key, value in problem.items() if key != 'tags'}
    for tag in problem['tags']:
        flat_problem[f'tag_{tag}'] = 1
    flattened_data.append(flat_problem)

# Create DataFrame
df = pd.DataFrame(flattened_data)

# Display DataFrame
print(df)
