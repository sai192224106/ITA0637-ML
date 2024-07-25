import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Sample mobile price dataset
data = {
    'battery_power': [842, 1021, 563, 615, 1821, 1859, 1821, 1044, 1907, 1558],
    'blue': [0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    'clock_speed': [2.2, 0.5, 0.5, 2.5, 1.2, 0.5, 1.9, 0.6, 2.2, 0.5],
    'dual_sim': [0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    'fc': [1, 0, 0, 2, 13, 3, 4, 0, 13, 3],
    'four_g': [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    'int_memory': [7, 53, 41, 10, 44, 22, 10, 24, 54, 16],
    'm_dep': [0.6, 0.7, 0.9, 0.2, 0.2, 0.2, 0.8, 0.6, 0.8, 0.2],
    'mobile_wt': [188, 136, 145, 131, 141, 164, 139, 187, 174, 93],
    'n_cores': [2, 3, 5, 6, 2, 1, 8, 2, 2, 1],
    'pc': [2, 6, 6, 9, 14, 7, 10, 0, 16, 7],
    'px_height': [20, 905, 1263, 1216, 1208, 1004, 381, 512, 1988, 214],
    'px_width': [756, 1988, 1716, 1786, 1216, 1654, 1218, 1149, 1440, 1111],
    'ram': [2549, 2631, 2603, 2769, 1411, 1067, 3220, 700, 3046, 2969],
    'sc_h': [9, 17, 11, 16, 8, 13, 17, 19, 5, 11],
    'sc_w': [7, 3, 2, 8, 2, 8, 1, 10, 0, 0],
    'talk_time': [19, 7, 9, 11, 15, 10, 18, 14, 5, 6],
    'three_g': [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    'touch_screen': [0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    'wifi': [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    'price_range': [1, 2, 2, 2, 1, 3, 0, 3, 0, 1]
}

df = pd.DataFrame(data)

# b) Print the 1st five rows
print("First five rows of the dataset:")
print(df.head())

# c) Basic statistical computations on the data set or distribution of data
print("\nBasic statistical computations:")
print(df.describe())

# d) The columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# e) Detects null values in the dataset. If there is any null values replaced it with mode value
print("\nNull values in the dataset:")
print(df.isnull().sum())
if df.isnull().sum().any():
    for column in df.columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    print("\nNull values after filling with mode:")
    print(df.isnull().sum())

# f) Explore the data set using heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# g) Split the data into test and train
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# h) Fit into the model Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# i) Predict the model
y_pred = model.predict(X_test)

# j) Find the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the Naive Bayes Classifier model:")
print(accuracy)