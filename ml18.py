import pandas as pd

# Load the CSV file
data = {
    'Shape':['Circular','Circular','Oval','Oval'],
    'Size' :['Large','Large','Large','Large'],
    'Color':['Light','Light','Dark','Light'],
    'Surface' :['Smooth','Irregular','Smooth','Irregular'],
    'Thickness' :['Thick','Thick','Thin','Thick'],
    'Target Concept' :['Maligant(+)','Malignant(+)','Benign(-)','Malignant(+)']
}
df = pd.DataFrame(data)
# Initialize S and G
# S is initialized to the most specific hypothesis
# G is initialized to the most general hypothesis
S = ['0'] * (len(df.columns) - 2)
G = [['?' for _ in range(len(df.columns) - 2)]]

# Function to update S
def update_S(s, example):
    for i in range(len(s)):
        if s[i] == '0':
            s[i] = example[i]
        elif s[i] != example[i]:
            s[i] = '?'
    return s

# Function to update G
def update_G(g, s, example):
    new_g = []
    import pandas as pd

# Load the CSV file
data = {
    'Shape':['Circular','Circular','Oval','Oval'],
    'Size' :['Large','Large','Large','Large'],
    'Color':['Light','Light','Dark','Light'],
    'Surface' :['Smooth','Irregular','Smooth','Irregular'],
    'Thickness' :['Thick','Thick','Thin','Thick'],
    'Target Concept' :['Maligant(+)','Malignant(+)','Benign(-)','Malignant(+)']
}
df = pd.DataFrame(data)
# Initialize S and G
# S is initialized to the most specific hypothesis
# G is initialized to the most general hypothesis
S = ['0'] * (len(df.columns) - 2)
G = [['?' for _ in range(len(df.columns) - 2)]]

# Function to update S
def update_S(s, example):
    for i in range(len(s)):
        if s[i] == '0':
            s[i] = example[i]
        elif s[i] != example[i]:
            s[i] = '?'
    return s

# Function to update G
def update_G(g, s, example):
    new_g = []
    # Output the final S and G
print("Final specific hypothesis S:", S)
print("Final general hypotheses G:", G)
