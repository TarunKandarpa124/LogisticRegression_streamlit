import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Load data
@st.cache
def load_data():
    train_df = pd.read_csv('Titanic_train.csv')
    test_df = pd.read_csv('Titanic_test.csv')
    return train_df, test_df

train_df, test_df = load_data()

# Display raw data
if st.checkbox('Show raw data'):
    st.write(train_df.head())

# Data Preprocessing
def preprocess_data(df, categoric, numeric):
    df.dropna(inplace=True)
    df = label_encode(df, categoric)
    df = standard_scaler(df, numeric)
    return df

def label_encode(df, categoric):
    le = LabelEncoder()
    for column in categoric:
        df[column] = le.fit_transform(df[column])
    return df

def standard_scaler(df, numeric):
    ss = StandardScaler()
    df[numeric] = ss.fit_transform(df[numeric])
    return df

categoric = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

train_df = preprocess_data(train_df, categoric, numeric)
test_df = preprocess_data(test_df, categoric, numeric)

# Split data
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lor = LogisticRegression()
lor.fit(X_train, y_train)

# Predict and evaluate
y_pred_train = lor.predict(X_train)
y_pred_test = lor.predict(X_test)
y_prob_test = lor.predict_proba(X_test)[:, 1]

# Display metrics
st.write("Train Accuracy:", accuracy_score(y_train, y_pred_train))
st.write("Test Accuracy:", accuracy_score(y_test, y_pred_test))
st.write("Train Precision:", precision_score(y_train, y_pred_train))
st.write("Test Precision:", precision_score(y_test, y_pred_test))
st.write("Train Recall:", recall_score(y_train, y_pred_train))
st.write("Test Recall:", recall_score(y_test, y_pred_test))
st.write("Train F1 Score:", f1_score(y_train, y_pred_train))
st.write("Test F1 Score:", f1_score(y_test, y_pred_test))
st.write("Train ROC AUC Score:", roc_auc_score(y_train, y_pred_train))
st.write("Test ROC AUC Score:", roc_auc_score(y_test, y_pred_test))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob_test):.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Predict on test data
predicted_survive = lor.predict(test_df)
test_df['Survived'] = predicted_survive

# Display prediction
st.write(test_df.head())

if st.button('Predict on test data'):
    st.write(predicted_survive)
