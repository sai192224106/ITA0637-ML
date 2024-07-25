import pandas as pd

# Dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Air Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'Enjoy Sport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Display the dataset
print("Dataset:")
print(df)

# Find-S algorithm implementation
def find_s(df):
    specific_hypothesis = []
    for i, row in df.iterrows():
        if row['Enjoy Sport'] == 'Yes':
            if not specific_hypothesis:
                specific_hypothesis = row[:-1].tolist()
            else:
                for j in range(len(specific_hypothesis)):
                    if specific_hypothesis[j] != row[j]:
                        specific_hypothesis[j] = '?'
    return specific_hypothesis

# Apply the Find-S algorithm
specific_hypothesis = find_s(df)

# Display the most specific hypothesis
print("\nThe most specific hypothesis is:")
print(specific_hypothesis)