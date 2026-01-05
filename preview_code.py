import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(r"E:\datasets for the hackatho\gait_fall_dataset.csv")

print(" Dataset loaded!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))


label_col = 'label'
print(f"\n Label distribution:")
print(df[label_col].value_counts())

X = df.drop(columns=['label'])
y_str = df['label']

le = LabelEncoder()
y = le.fit_transform(y_str)

print("\nFeatures shape:", X.shape)
print("Classes:", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
y_pred_str = le.inverse_transform(y_pred)
y_test_str = le.inverse_transform(y_test)

print(f"\n Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\n Classification Report:")
print(classification_report(y_test_str, y_pred_str))


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\n Top 10 Important Features:")
print(feature_importance)
