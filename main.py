import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('winequality_red_cleaned.csv')

st.title("Wine Quality Prediction App")
st.write("This app predicts the quality of red wine based on its chemical properties.")

if st.checkbox("Show raw data"):
    st.write(data.head(7))

X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.header("Choose a model")
model_type = st.sidebar.selectbox("Model", ("Random Forest", "Decision Tree", "SVM"))

params = dict()

if model_type == "Decision Tree":
    params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    params['min_samples_split'] = st.sidebar.number_input("Min Samples Split", 2, 10, 2)
    params['criterion'] = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
    params['splitter'] = st.sidebar.radio("Splitter", ("best", "random"))

    model = DecisionTreeClassifier(**params, random_state=42)

elif model_type == "Random Forest":
    st.sidebar.subheader("Random Forest Hyperparameters")
    params['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 200, 100)
    params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    params['min_samples_split'] = st.sidebar.number_input("Min Samples Split", 2, 10, 2)
    params['criterion'] = st.sidebar.selectbox("Criterion", ("gini", "entropy"))

    model = RandomForestClassifier(**params, random_state=42)


elif model_type == "SVM":
    st.sidebar.subheader("SVM Hyperparameters")
    params['C'] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
    params['kernel'] = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly"))
    params['gamma'] = st.sidebar.selectbox("Gamma", ("scale", "auto"))

    model = SVC(**params, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Model Performance")
st.write("Accuracy: ", accuracy_score(y_test, y_pred))
st.write("Classification Report:\n", classification_report(y_test, y_pred))
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

