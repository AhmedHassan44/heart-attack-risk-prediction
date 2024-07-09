import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix

# Load the dataset into a DataFrame
df = pd.read_csv('heart.csv')

# Display the first few rows of the dataset and column names
print(df.head())
print("Dataset dimensions:", df.shape)
print("Column names:", df.columns)

# -------------------------------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------------------------------

# Distributions of Numerical Variables

# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=20, kde=True, color='blue', alpha=0.7)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Distribution of Resting Blood Pressure (trtbps)
plt.figure(figsize=(10, 6))
sns.histplot(df['trtbps'], bins=20, kde=True, color='green', alpha=0.7)
plt.title('Distribution of Resting Blood Pressure')
plt.xlabel('Resting Blood Pressure (mm Hg)')
plt.ylabel('Frequency')
plt.show()

# Distribution of Cholesterol Levels (chol)
plt.figure(figsize=(10, 6))
sns.histplot(df['chol'], bins=20, kde=True, color='orange', alpha=0.7)
plt.title('Distribution of Cholesterol Levels')
plt.xlabel('Cholesterol Level (mg/dl)')
plt.ylabel('Frequency')
plt.show()

# Distribution of Maximum Heart Rate Achieved (thalachh)
plt.figure(figsize=(10, 6))
sns.histplot(df['thalachh'], bins=20, kde=True, color='red', alpha=0.7)
plt.title('Distribution of Maximum Heart Rate Achieved')
plt.xlabel('Maximum Heart Rate Achieved')
plt.ylabel('Frequency')
plt.show()

# -------------------------------------------------------
# Relationships Between Variables
# -------------------------------------------------------

# Pairplot to visualize relationships between numerical variables
sns.pairplot(df, vars=['age', 'trtbps', 'chol', 'thalachh'], hue='output', palette='viridis')
plt.suptitle('Pairplot of Numerical Variables by Heart Attack Risk', y=1.02)
plt.show()

# Box plots to compare distributions of numerical variables by heart attack risk
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.boxplot(x='output', y='age', data=df, palette='viridis', hue='output', legend=False)
plt.title('Age vs. Heart Attack Risk')

plt.subplot(2, 2, 2)
sns.boxplot(x='output', y='trtbps', data=df, palette='viridis', hue='output', legend=False)
plt.title('Resting Blood Pressure vs. Heart Attack Risk')

plt.subplot(2, 2, 3)
sns.boxplot(x='output', y='chol', data=df, palette='viridis', hue='output', legend=False)
plt.title('Cholesterol Levels vs. Heart Attack Risk')

plt.subplot(2, 2, 4)
sns.boxplot(x='output', y='thalachh', data=df, palette='viridis', hue='output', legend=False)
plt.title('Maximum Heart Rate vs. Heart Attack Risk')

plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------

# Check for missing data
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Handle missing data (if any)
# Example: Replace missing values with mean or median
df.fillna(df.mean(), inplace=True)

# Detect and handle outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[~outliers]

# Print dimensions after handling missing data and outliers
print("Dataset dimensions after preprocessing:", df.shape)

# -------------------------------------------------------
# Encode Categorical Variables
# -------------------------------------------------------

# Example: Encode categorical variable 'sex' (if it's not already encoded)
label_encoder = LabelEncoder()
df['sex_encoded'] = label_encoder.fit_transform(df['sex'])

# Drop original 'sex' column if not needed
df.drop(columns=['sex'], inplace=True)

# -------------------------------------------------------
# Scale Numerical Features
# -------------------------------------------------------

# Select numerical columns to scale
numerical_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# -------------------------------------------------------
# Predictive Modeling
# -------------------------------------------------------

# Split data into training and testing sets
X = df.drop(columns=['output'])
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Train and evaluate models
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Results for {model_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    print("---------------------------------------")

# -------------------------------------------------------
# Integrating MySQL and Excel
# -------------------------------------------------------

# Connect to MySQL database
engine = create_engine('mysql+mysqlconnector://root:17#Mysql17@localhost/heart_attack_analysis')

# Export cleaned and preprocessed DataFrame to MySQL
df.to_sql(name='heart_attack_data', con=engine, if_exists='replace', index=False)

print("Data loaded successfully into MySQL.")


