import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, LassoLars
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pickle
import streamlit as st


def preprocess_data(df):
    # Preprocess the DataFrame
    df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce')
    df.rename(columns={'quantity tons': 'quantity_tons'}, inplace=True)
    df['quantity_tons'] = df['quantity_tons'].replace('e', np.NaN).astype('float64').abs()
    df['quantity_tons'] = df['quantity_tons'].map('{:.2f}'.format)
    df['quantity_tons'] = pd.to_numeric(df['quantity_tons'], errors='coerce')
    df.loc[df['material_ref'].astype(str).str.startswith('00000'), 'material_ref'] = None
    df['product_ref'] = df['product_ref'].astype(str)
    df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce')

    mode = ['item_date', 'customer', 'status', 'item type', 'application', 'material_ref', 'country', 'delivery date', 'product_ref']
    mean = ['quantity_tons', 'thickness', 'width', 'selling_price']
    columns = list(df.columns)

    for i in columns:
        if i in mode:
            df[i].fillna(df[i].mode()[0], inplace=True)
        elif i in mean:
            df[i].fillna(df[i].mean(), inplace=True)

def remove_outliers(df):
    # Remove outliers using IQR method
    continuous_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
    q1 = df[continuous_columns].quantile(0.25)
    q3 = df[continuous_columns].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[((df[continuous_columns] >= lower_bound) & (df[continuous_columns] <= upper_bound)).all(axis=1)]
    return df

def transform_skewed_columns(df):
    # Transform skewed columns using log transformation
    continuous_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
    skewness = df[continuous_columns].apply(lambda x: skew(x))
    skewed_columns = skewness[skewness > 0.5].index
    df[skewed_columns] = df[skewed_columns].apply(lambda x: np.log1p(x))

def train_regression_models(X_train, X_test, Y_train, Y_test):
    # Train regression models and evaluate them
    models = [
        LinearRegression(),
        ElasticNet(),
        LassoLars(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        ExtraTreesRegressor()
    ]

    model_names = ['Linear Regression', 'ElasticNet', 'LassoLars', 'KNeighbors', 'Decision Tree', 'Extra Trees']
    mse_scores = []
    r2_scores = []
    mae_scores = []
    for model, model_name in zip(models, model_names):
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        mae = mean_absolute_error(Y_test, y_pred)
        mse_scores.append(mse)
        r2_scores.append(r2)
        mae_scores.append(mae)
        print(f"{model_name} - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}, Mean Absolute Error: {mae:.4f}")
    return mse_scores, r2_scores, mae_scores

def train_classification_models(X_train, X_test, y_train, y_test):
    # Train classification models and evaluate them
    models = [
        LogisticRegression(),
        KNeighborsClassifier(),
        AdaBoostClassifier(),
        DecisionTreeClassifier(),
        GaussianNB(),
        RandomForestClassifier(),
        ExtraTreesClassifier(),
        XGBClassifier()
    ]

    model_names = ['Logistic Regression', 'KNeighbors', 'AdaBoost', 'Decision Tree', 'GaussianNB', 'Random Forest',
               'Extra Trees', 'XGBoost']
    accuracy_scores = []
    f1_scores = []
    roc_auc_scores = []
    conf_matrices = []
    for model, model_name in zip(models, model_names):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        roc_auc_scores.append(roc_auc)
        conf_matrices.append(conf_matrix)
        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    return accuracy_scores, f1_scores, roc_auc_scores, conf_matrices

def save_model_artifacts(scaler, model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(scaler, f)
        pickle.dump(model, f)

def load_model_artifacts(filename):
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)
        model = pickle.load(f)
    return scaler, model

def load_model_artifacts(filename):
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)
        model = pickle.load(f)
    return scaler, model

def run_regression_app(df):
    st.subheader("Regression App")
    
    # Load the regression model artifacts
    scaler_reg, model_reg = load_model_artifacts('regression_model_artifacts.pkl')

    # Input features for regression
    st.sidebar.subheader("Input Features for Regression")
    quantity_tons = st.sidebar.number_input("Quantity in tons", min_value=0.01, max_value=10000.0, value=1.0)
    country = st.sidebar.selectbox("Country", sorted(df['country'].unique()))
    application = st.sidebar.selectbox("Application", sorted(df['application'].unique()))
    thickness = st.sidebar.number_input("Thickness", min_value=0.1, max_value=100.0, value=1.0)
    width = st.sidebar.number_input("Width", min_value=0.1, max_value=100.0, value=1.0)
    delivery_period = st.sidebar.number_input("Delivery Period (days)", min_value=0, max_value=365, value=30)

    # Preprocess the input data
    input_data = np.array([[quantity_tons, country, application, thickness, width, delivery_period]])
    input_data_scaled = scaler_reg.transform(input_data)

    if st.sidebar.button("Predict Selling Price"):
        # Make prediction using the regression model
        selling_price_pred = model_reg.predict(input_data_scaled)[0]

        st.write("Predicted Selling Price: {:.2f}".format(selling_price_pred))

def run_classification_app(df):
    st.subheader("Classification App")
    
    # Load the classification model artifacts
    scaler_clf, model_clf = load_model_artifacts('classification_model_artifacts.pkl')

    # Input features for classification
    st.sidebar.subheader("Input Features for Classification")
    quantity_tons_clf = st.sidebar.number_input("Quantity in tons", min_value=0.01, max_value=10000.0, value=1.0)
    country_clf = st.sidebar.selectbox("Country", sorted(df['country'].unique()))
    application_clf = st.sidebar.selectbox("Application", sorted(df['application'].unique()))
    thickness_clf = st.sidebar.number_input("Thickness", min_value=0.1, max_value=100.0, value=1.0)
    width_clf = st.sidebar.number_input("Width", min_value=0.1, max_value=100.0, value=1.0)
    delivery_period_clf = st.sidebar.number_input("Delivery Period (days)", min_value=0, max_value=365, value=30)

    # Preprocess the input data
    input_data_clf = np.array([[quantity_tons_clf, country_clf, application_clf, thickness_clf, width_clf, delivery_period_clf]])
    input_data_scaled_clf = scaler_clf.transform(input_data_clf)

    if st.sidebar.button("Predict Status"):
        # Make prediction using the classification model
        status_pred = model_clf.predict(input_data_scaled_clf)[0]

        st.write("Predicted Status: {}".format(status_pred))


if __name__ == "__main__":
    # Read data from CSV
    df = pd.read_csv("Copper_Set.csv")

    # Preprocess data for regression and classification tasks
    preprocess_data(df)
    df_regression = df.copy()
    df_classification = df.copy()

    # Remove outliers for regression task
    df_regression = remove_outliers(df_regression)

    # Transform skewed columns for regression task
    transform_skewed_columns(df_regression)

    # Train and evaluate regression models
    x_regression = df_regression[['quantity_tons', 'country', 'application', 'thickness', 'width', 'delivery_period']].values
    y_regression = df_regression[['selling_price']].values
    X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(x_regression, y_regression.reshape(-1), test_size=0.3)
    mse_scores_reg, r2_scores_reg, mae_scores_reg = train_regression_models(X_train_reg, X_test_reg, Y_train_reg, Y_test_reg)

    # Save regression model artifacts
    scaler_reg = StandardScaler().fit(x_regression)
    save_model_artifacts(scaler_reg, models[-1], 'regression_model_artifacts.pkl')

    # Remove outliers for classification task
    df_classification = remove_outliers(df_classification)

    # Transform skewed columns for classification task
    transform_skewed_columns(df_classification)

    # Train and evaluate classification models
    x_classification = df_classification[['quantity_tons', 'country', 'application', 'thickness', 'width', 'selling_price', 'delivery_period']].values
    y_classification = df_classification[['status']].values
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(x_classification, y_classification, test_size=0.2, random_state=42)
    accuracy_scores_clf, f1_scores_clf, roc_auc_scores_clf, conf_matrices_clf = train_classification_models(X_train_clf, X_test_clf, y_train_clf, y_test_clf)

    # Save classification model artifacts
    scaler_clf = StandardScaler().fit(x_classification)
    save_model_artifacts(scaler_clf, models[-2], 'classification_model_artifacts.pkl')

    # Run Streamlit apps
    run_regression_app(df_regression)
    run_classification_app(df_classification)
