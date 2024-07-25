import pandas as pd

# Define the dataset
data = {
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': ['1980', '1970', '1990', '1980', '1980'],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Function to implement Find-S algorithm
def find_s_algorithm(df):
    # Initialize the most specific hypothesis
    hypothesis = ['ϕ'] * (df.shape[1] - 1)

    # Iterate through the dataset
    for i in range(df.shape[0]):
        if df.iloc[i]['Example Type'] == 'Positive':
            if hypothesis == ['ϕ'] * (df.shape[1] - 1):
                hypothesis = list(df.iloc[i][:-1])
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != df.iloc[i, j]:
                        hypothesis[j] = '?'
    return hypothesis

# Apply the Find-S algorithm to the dataset
hypothesis = find_s_algorithm(df)

# Output the most specific hypothesis
print("The most specific hypothesis is:", hypothesis)

