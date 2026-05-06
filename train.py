import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Setup Directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 2. Load Data
# We are pulling a clean version of the standard Titanic dataset directly
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
print("Loading data...")
df = pd.read_csv(url)
df.to_csv('data/titanic.csv', index=False) # Save a local copy

# 3. Data Preprocessing & Feature Engineering
print("Cleaning data and engineering features...")

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop('Cabin', axis=1, inplace=True) # Dropped because ~77% of data is missing

# Feature Engineering: Family Size
df['FamilySize'] = df['SibSp'] + df['Parch']

# Feature Engineering: Extract Titles from Names
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Encode Categorical Variables
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
le_title = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
df['Title'] = le_title.fit_transform(df['Title'])

# Select final features for the model
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Title', 'Embarked']
X = df[features]
y = df['Survived']

# 4. Train/Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler so we can scale user input the exact same way in our Streamlit app
joblib.dump(scaler, 'models/scaler.joblib')

# 5. Model Training and Evaluation
print("\n--- Model Training & Comparison ---")

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report (Precision & Recall):")
    print(classification_report(y_test, y_pred))
    
    # Keep track of the best model to save for production
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# 6. Save the Best Model
print(f"\nSaving the best model (Accuracy: {best_accuracy:.4f}) for the web app...")
joblib.dump(best_model, 'models/best_titanic_model.joblib')
print("Training complete. Scaler and Model saved in 'models/' directory.")