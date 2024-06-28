import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

sns.scatterplot(x='Hours_Studied', y='Pass', data=df)
plt.title('Pass vs Hours Studied')
plt.xlabel('Hours Studied')
plt.ylabel('Pass (1: Yes, 0: No)')
plt.show()

X = df[['Hours_Studied']]
y = df['Pass']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Visualize the decision boundary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hours_Studied', y='Pass', data=df)
plt.title('Pass vs Hours Studied with Decision Boundary')
plt.xlabel('Hours Studied')
plt.ylabel('Pass (1: Yes, 0: No)')

# Plot decision boundary
x_values = [i for i in range(1, 11)]
y_values = [(model.coef_ * x + model.intercept_)[0] for x in x_values]
plt.plot(x_values, y_values, color='red')

plt.show()