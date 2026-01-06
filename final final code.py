import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# DATASET 1: Gait Fall Detection 
print("DATASET 1: GAIT FALL DETECTION")
print("="*60)

df1 = pd.read_csv(r"E:\datasets for the hackatho\gait_fall_dataset.csv")
label_col1 = 'label'

X1 = df1.drop(columns=[label_col1])
y1_str = df1[label_col1]
le1 = LabelEncoder()
y1 = le1.fit_transform(y1_str)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)
rf1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf1.fit(X1_train, y1_train)

y1_pred = rf1.predict(X1_test)
acc1 = accuracy_score(y1_test, y1_pred)
print(f" Accuracy: {acc1:.3f}")
print(f"Classes: {le1.classes_}")

# DATASET 2: UCI HAR Smartphone 
print("\n"*2 + "="*60)
print("DATASET 2: UCI HAR SMARTPHONE ACTIVITIES")
print("="*60)

df2 = pd.read_csv(r"E:\datasets for the hackatho\test.csv")  
label_col2 = 'Activity'  

X2 = df2.drop(columns=[label_col2])
y2_str = df2[label_col2]
le2 = LabelEncoder()
y2 = le2.fit_transform(y2_str)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2.fit(X2_train, y2_train)

y2_pred = rf2.predict(X2_test)
acc2 = accuracy_score(y2_test, y2_pred)
print(f" Accuracy: {acc2:.3f}")
print(f"Classes: {le2.classes_}")

# DATASET 3: MotionSense Subject Info 
print("\n"*2 + "="*60)
print("DATASET 3: MOTIONSENSE SUBJECT INFO")
print("="*60)

df3 = pd.read_csv(r"E:\datasets for the hackatho\data_subjects_info.csv")
print("Dataset 3 columns:", df3.columns.tolist())
print("Shape:", df3.shape)


label_col3 = None
for col in ['label', 'activity', 'class', 'gender']:
    if col in df3.columns:
        label_col3 = col
        break

if label_col3 is None:
    print("❌ No suitable label found. Using 'gender' as classification target")
    label_col3 = 'gender'  

print(f"Using label: '{label_col3}'")

X3 = df3.drop(columns=[label_col3])
y3_str = df3[label_col3]
le3 = LabelEncoder()
y3 = le3.fit_transform(y3_str)

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)
rf3 = RandomForestClassifier(n_estimators=100, random_state=42)
rf3.fit(X3_train, y3_train)

y3_pred = rf3.predict(X3_test)
acc3 = accuracy_score(y3_test, y3_pred)
print(f" Accuracy: {acc3:.3f}")
print(f"Classes: {le3.classes_}")

# FINAL COMPARISON
print("\n"*2 + "="*60)
print(" FINAL COMPARISON")
print("="*60)
comparison = pd.DataFrame({
    'Dataset': ['Gait Fall', 'UCI HAR', 'MotionSense Subjects'],
    'Accuracy': [acc1, acc2, acc3],
    'Num Classes': [len(le1.classes_), len(le2.classes_), len(le3.classes_)],
    'Best Model': ['RF1', 'RF2', 'RF3']
})
print(comparison.sort_values('Accuracy', ascending=False))


print("\n Dataset 1 Top Features:")
feature_df1 = pd.DataFrame({
    'feature': X1.columns,
    'importance': rf1.feature_importances_
}).sort_values('importance', ascending=False).head(5)
print(feature_df1)

