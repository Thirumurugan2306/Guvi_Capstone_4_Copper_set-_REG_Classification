import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
import pickle


# Define a function for Streamlit page configuration
def page_config():
    st.set_page_config(
        page_title="Industrial Copper Modelling",
        layout="wide",
    )
    
    # Add background image using Markdown
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1496247749665-49cf5b1022e9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2073&q=80");
            background-attachment: scroll;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Custom CSS style for the title
    custom_style = """
    <style>
    h1 {
        color: #B87333;
        background-color: white; /* Change this to the highlight color you want */
        padding: 5px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """
    # Display the custom CSS style using st.markdown
    st.markdown(custom_style, unsafe_allow_html=True)
    # Display the title using h1 HTML tag
    st.markdown("<h1>Industrial Copper Modeling</h1>", unsafe_allow_html=True)
    st.write('')

# Define a function to display the navigation menu
def display_navigation():
    col1, col2, col3 = st.columns(3)
    
    with col2:
        selected = option_menu(
                menu_title="Select the Model",
                options=["Regression", "Classification"],
                orientation="horizontal",
                styles={
                    "container": {"margin": "1", "padding": "2!important", "background-color": "White"},
                    "nav-link": {"font-size": "10px", "text-align": "center", "margin": "0.5px", "--hover-color": "#B87333"},
                    "nav-link-selected": {"background-color": "#B87333"},
                }
            )
        return selected

# Define a function to preprocess the data
def preprocess_data(df):
    # Preprocess the DataFrame
    
    # Convert date columns to datetime format
    df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce')
    df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce')
    
    # Rename a column
    df.rename(columns={'quantity tons': 'quantity_tons'}, inplace=True)
    
    # Convert quantity_tons column to float64
    df['quantity_tons'] = df['quantity_tons'].replace('e', np.NaN).astype('float64').abs()
    df['quantity_tons'] = df['quantity_tons'].map('{:.2f}'.format)
    df['quantity_tons'] = pd.to_numeric(df['quantity_tons'], errors='coerce')
    
    # Set material_ref to None for specific cases
    df.loc[df['material_ref'].astype(str).str.startswith('00000'), 'material_ref'] = None
    
    # Convert product_ref column to string
    df['product_ref'] = df['product_ref'].astype(str)

    # Fill missing values in specific columns
    mode = ['item_date', 'customer', 'status', 'item type', 'application', 'material_ref', 'country', 'delivery date', 'product_ref']
    mean = ['quantity_tons', 'thickness', 'width', 'selling_price']
    columns = list(df.columns)

    for i in columns:
        if i in mode:
            df[i].fillna(df[i].mode()[0], inplace=True)
        elif i in mean:
            df[i].fillna(df[i].mean(), inplace=True)

# Define a function to remove outliers using IQR method
def remove_outliers(df):
    continuous_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
    q1 = df[continuous_columns].quantile(0.25)
    q3 = df[continuous_columns].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[((df[continuous_columns] >= lower_bound) & (df[continuous_columns] <= upper_bound)).all(axis=1)]
    return df

# Define a function to transform skewed columns using log transformation
def transform_skewed_columns(df):
    continuous_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
    skewness = df[continuous_columns].apply(lambda x: skew(x))
    skewed_columns = skewness[skewness > 0.5].index
    df[skewed_columns] = df[skewed_columns].apply(lambda x: np.log1p(x))
    df['delivery_period'] = (df['item_date'] - df['delivery date']).abs().dt.days

# Define a function to train regression models
def train_regression_models(df):
    x = df[['quantity_tons', 'country', 'application', 'thickness', 'width', 'delivery_period']].values
    y = df[['selling_price']].values
    scaler = StandardScaler().fit(x)
    
    # Saving scaler to a pickle file
    with open('scaling_regression.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Transform training data
    x = scaler.transform(x)
    
    # Split train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y.reshape(-1), test_size=0.2)
    
    model = ExtraTreesRegressor(random_state=42)  # Initialize the Extra Trees model
    model.fit(X_train, Y_train)

    # Saving the trained model to a pickle file
    with open('extra_trees_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    

# Define a function for the Regression model
def Regression_model():
    col1, col2, col3 = st.columns(3)
    
    with col2:
        custom_style = """
        <style>
            .input-field {
                background-color: white;
                color: white; /* Set font color to white */
                border: 1px solid blue;
                border-radius: 5px;
                padding: 10px;
                width: 100%;
            }
            .custom-header {
                color: white; /* Set font color of header to white */
            }
        </style>
        """
        st.markdown(custom_style, unsafe_allow_html=True)
        
        # Markdown header with white font color
        st.markdown("<h3 class='custom-header'>Enter the following information to predict the selling price</h3>", unsafe_allow_html=True)
        
        # Input fields with white font color
        st.markdown("<h4 class='custom-header'>Quantity in tons</h4>", unsafe_allow_html=True)
        quantity_tons = st.number_input("", value=100.0, format="%.2f")
        
        st.markdown("<h4 class='custom-header'>Country</h4>", unsafe_allow_html=True)
        country = st.number_input("", value=10)
        
        st.markdown("<h4 class='custom-header'>Application</h4>", unsafe_allow_html=True)
        application = st.number_input("", value=0)
        
        st.markdown("<h4 class='custom-header'>Thickness</h4>", unsafe_allow_html=True)
        thickness = st.number_input("", value=5.0, format="%.2f")
        
        st.markdown("<h4 class='custom-header'>Width</h4>", unsafe_allow_html=True)
        width = st.number_input("", value=10.0, format="%.2f")
        
        st.markdown("<h4 class='custom-header'>Delivery Period (days)</h4>", unsafe_allow_html=True)
        delivery_period = st.number_input("", value=30)
        
        
        if st.button("Predict"):
            # Load the trained scaler and model
            with open('scaling_regression.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            with open('extra_trees_regression_model.pkl', 'rb') as f:
                model = pickle.load(f)

            # Create a DataFrame with user input
            user_input = pd.DataFrame({
                'quantity_tons': [quantity_tons],
                'country': [country],
                'application': [application],
                'thickness': [thickness],
                'width': [width],
                'delivery_period': [delivery_period]
            })

            # Preprocess user input and make a prediction
            #user_input['country'] = user_input['country'].astype(str)
            #user_input['application'] = user_input['application'].astype(str)
            x = user_input[['quantity_tons', 'country', 'application', 'thickness', 'width', 'delivery_period']].values
            x = scaler.transform(x)
            predicted_price = model.predict(x)

            st.success(f"Predicted Selling Price: {predicted_price:.2f}")
            container = st.container()
            container.write(f"Predicted Selling Price: {predicted_price:.2f}")

# Define a function to train classification models
def train_classification(df):
    df_classification = df.loc[df["status"].isin(['Won', 'Lost'])]
    x = df_classification[['quantity_tons', 'country', 'application', 'thickness', 'width', 'selling_price', 'delivery_period']].values
    y = df_classification[['status']].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize the scaler and fit it on the training data
    scaler = StandardScaler().fit(X_train)

    # Save the scaler using joblib
    with open('scaling_classification.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Transform both training and testing data using the scaler
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the model and fit it on the training data
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the trained model using joblib

    col1, col2, col3 = st.columns(3)
    
    with col2:
        custom_style = """
        <style>
            .input-field {
                background-color: white;
                color: white; /* Set font color to white */
                border: 1px solid blue;
                border-radius: 5px;
                padding: 10px;
                width: 100%;
            }
            .custom-header {
                color: white; /* Set font color of header to white */
            }
        </style>
        """
        st.markdown(custom_style, unsafe_allow_html=True)
        
        # Markdown header with white font color
        st.markdown("<h3 class='custom-header'>Enter the following information to predict the status</h3>", unsafe_allow_html=True)
        
        # Input fields with white font color
        st.markdown("<h4 class='custom-header'>Quantity in tons</h4>", unsafe_allow_html=True)
        quantity_tons = st.number_input("", value=100.0, format="%.2f")
        
        st.markdown("<h4 class='custom-header'>Country</h4>", unsafe_allow_html=True)
        country = st.number_input("",value=10)
        
        st.markdown("<h4 class='custom-header'>Application</h4>", unsafe_allow_html=True)
        application = st.number_input("",Value=5)
        
        st.markdown("<h4 class='custom-header'>Thickness</h4>", unsafe_allow_html=True)
        thickness = st.number_input("", value=5.0, format="%.2f")
        
        st.markdown("<h4 class='custom-header'>Width</h4>", unsafe_allow_html=True)
        width = st.number_input("", value=10.0, format="%.2f")
        
        st.markdown("<h4 class='custom-header'>Selling Price</h4>", unsafe_allow_html=True)
        selling_price = st.number_input("", value=50.0, format="%.2f")
        
        st.markdown("<h4 class='custom-header'>Delivery Period (days)</h4>", unsafe_allow_html=True)
        delivery_period = st.number_input("", value=30)
        
        if st.button("Predict"):
            # Load the trained scaler and model
            #scaler = joblib.load('scaling_classification.pkl')
            #model = joblib.load('random_forest_classification_model.pkl')
    
            # Create a DataFrame with user input
            user_input = pd.DataFrame({
                'quantity_tons': [quantity_tons],
                'country': [country],
                'application': [application],
                'thickness': [thickness],
                'width': [width],
                'selling_price': [selling_price],
                'delivery_period': [delivery_period]
            })
    
            # Preprocess user input and make a prediction
            #user_input['country'] = user_input['country'].astype(str)
            #user_input['application'] = user_input['application'].astype(str)
            x = user_input[['quantity_tons', 'country', 'application', 'thickness', 'width', 'selling_price', 'delivery_period']].values
            x = scaler.transform(x)
            predicted_status = model.predict(x)
            if predicted_status==1:
                predicted_status="Won"
            else:
                predicted_status="Lost"
    
            st.success(f"Predicted Status: {predicted_status}")
if __name__ == '__main__':
    # Configure Streamlit page
    page_config()
    
    # Display navigation menu and select model
    selected = display_navigation()
    
    # Load the dataset
    df = pd.read_csv('Copper_Set.csv')
    
    # Preprocess the data
    preprocess_data(df)
    df = remove_outliers(df)
    transform_skewed_columns(df)
    
    # Train and run models based on user selection
    if selected == "Regression":
        train_regression_models(df)
        Regression_model()
    elif selected == "Classification":
        train_classification(df)
