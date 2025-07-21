# da.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("customers-100.csv")
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

# Preprocessing
data['subscription_date'] = pd.to_datetime(data['subscription_date'])
data['days_since_subscribed'] = (datetime.now() - data['subscription_date']).dt.days
data['email_domain'] = data['email'].apply(lambda x: x.split('@')[-1])

# Create binary target: 1 if subscribed after Jan 1, 2021
data['recent_subscriber'] = (data['subscription_date'] > '2021-01-01').astype(int)

# Encode categorical columns
for col in ['city', 'country', 'email_domain']:
    data[col] = LabelEncoder().fit_transform(data[col])

# Features and Target
features = ['city', 'country', 'days_since_subscribed', 'email_domain']
X = StandardScaler().fit_transform(data[features])
y = data['recent_subscriber']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("🔷 Confusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Prosentropy Error
def prosentropy_error(y_true, y_pred):
    errors = y_true != y_pred
    percent_error = np.mean(errors) * 100
    entropy = -np.sum(errors * np.log2(np.random.rand(len(errors)) + 1e-9)) / len(errors)
    return round(percent_error, 2), round(entropy, 4)

err, entropy = prosentropy_error(y_test, y_pred)
print(f"\n⚠️ Prosentropy Error:\n - % Error: {err}%\n - Entropy: {entropy}")

# 10-Fold Cross Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kfold)
print("\n🔁 10-Fold Accuracy Scores:", scores)
print("📈 Average Accuracy:", round(scores.mean(), 4))

# Accuracy with various train/test ratios
print("\n📚 Accuracy with different splits:")
for r in [0.2, 0.3, 0.4]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{int((1-r)*100)}:{int(r*100)} split → Accuracy: {round(acc, 4)}")
