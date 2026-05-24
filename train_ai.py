import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("1. Loading the dataset...")
# Load your real Kaggle dataset!
df = pd.read_csv('cropdata_updated.csv')

# Drop any blank rows just to be safe
df = df.dropna()

print("2. Preparing the data...")
# X contains the environment data (Inputs)
# We map them exactly to the column names in your CSV: 'temp', 'humidity', 'MOI'
X = df[['temp', 'humidity', 'MOI']]

# y contains the target answer (Output: 'result' column which is 1 or 0)
y = df['result']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Training the AI Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("4. Testing the AI Model...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"--> AI Accuracy: {accuracy * 100:.2f}%")

print("5. Saving the AI Brain...")
joblib.dump(model, 'smart_brain.pkl')
print("Success! 'smart_brain.pkl' has been saved to your folder.")