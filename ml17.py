import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Step 2: Plot the data using a scatter plot "sepal_width" versus "sepal_length" and color species
plt.figure(figsize=(10, 6))
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
plt.scatter(df['sepal width (cm)'], df['sepal length (cm)'], c=df['species'].apply(lambda x: colors[x]), label=colors)
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.title('Sepal Width vs Sepal Length')
plt.legend(colors)
plt.show()

# Step 3: Split the data
X = df.drop(columns='species')
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Fit the data to the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(f'Training Accuracy: {accuracy_score(y_train, y_pred_train):.2f}')
print(f'Testing Accuracy: {accuracy_score(y_test, y_pred_test):.2f}')

# Step 5: Predict the model with new test data [5, 3, 1, 0.3]
new_sample = np.array([[5, 3, 1, 0.3]])
prediction = model.predict(new_sample)
predicted_species = prediction[0]
print(f'The predicted species for the new sample [5, 3, 1, 0.3] is: {predicted_species}')