# App Link : https://guvicapstone4copperset-regclassification-4cpqsjaeopsyurepdnka7.streamlit.app/
# Guvi_Capstone_4_Copper_set-_REG_Classification

## Data Preprocessing
The data was preprocessed using the preprocess_data, remove_outliers, and transform_skewed_columns functions to handle missing values, remove outliers, and transform skewed features.

# Model Training
## For the regression task, the following models were trained and evaluated:
Linear Regression
ElasticNet
LassoLars
KNeighborsRegressor
DecisionTreeRegressor
ExtraTreesRegressor

## For the classification task, the following models were trained and evaluated:
Logistic Regression
KNeighborsClassifier
AdaBoostClassifier
DecisionTreeClassifier
GaussianNB
RandomForestClassifier
ExtraTreesClassifier
XGBClassifier

# Streamlit Apps
The Streamlit apps for regression and classification tasks allow users to make predictions using the trained models.

## Results and Artifacts
The regression and classification models were evaluated using mean squared error, R-squared, and mean absolute error for regression and accuracy, F1 score, and ROC AUC for classification. The trained models were saved as artifacts for future use.

## Limitations and Future Improvements
Mention any limitations of the current implementation and possible future improvements.
