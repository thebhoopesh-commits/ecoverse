import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n_samples = 2000


data = {
    'gait_sensor': np.random.normal(0, 1, n_samples),
    'fall': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
}


data['fall'] = (abs(data['gait_sensor']) > 2.5).astype(int)

df = pd.DataFrame(data)
print("✅ Created test data:", df.shape)
print(df.head())
print("\nFall distribution:\n", df['fall'].value_counts())


X = df[['gait_sensor']]
y = df['fall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
