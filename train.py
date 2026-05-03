import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/Iris.csv")

# Features and target
X = df.iloc[:, 1:-1]   # skipping Id column
y = df.iloc[:, -1]     # Species column

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create outputs folder if not exists
os.makedirs("../outputs", exist_ok=True)

# Plot graph
plt.figure()
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Iris Classification Confusion Matrix")

# Save graph
plt.savefig("../outputs/confusion_matrix.png")

plt.show()