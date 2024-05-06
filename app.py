# Import necessary libraries
import joblib
import streamlit as st
import pandas as pd

# Preprocessing
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
)
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)

from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.preprocessing import PowerTransformer

# Models
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    accuracy_score,
    log_loss,
    roc_auc_score,
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
)


class HandleSmokingStatus(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["smoking_status"].fillna(value="Unknown", inplace=True)
        X_copy["smoking_not_found"] = (X_copy["smoking_status"] == "Unknown").astype(
            int
        )
        return X_copy


class SamplerTransformer:
    def __init__(self, sampler):
        self.sampler = sampler

    def fit(self, X, y):
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        return X_resampled, y_resampled

    def transform(self, X):
        return X

    def fit_transform(self, X, y):
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        return X_resampled, y_resampled


# Define function to load machine learning model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    preprocessing = joblib.load("preprocessing.pkl")
    return preprocessing, model


# Load the model
preprocessor, model = load_model()


# Define function to make predictions
def make_prediction(input_data):
    prediction = model.predict(input_data)
    return prediction


def preprocess(input_data):
    preprocessed_data = preprocessor.transform(input_data)
    return preprocessed_data


# Define the main function to run the Streamlit app
def main():
    # Set title and description
    st.title("Stroke Prediction Web App")
    st.write("This is a simple machine learning web application to predict stroke.")

    # Add a section for user input
    st.sidebar.header("User Input")

    # Allow user to input data
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 0, 100, 50)
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    heart_disease = st.sidebar.radio("Heart Disease", [0, 1])
    hypertension = st.sidebar.radio("Hypertension", [0, 1])
    work_type = st.sidebar.selectbox(
        "Work Type",
        ["children", "Private", "Govt_job", "Self-employed", "Never_worked"],
    )
    residence_type = st.sidebar.selectbox("Residence Type", ["Rural", "Urban"])
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 150.0)
    bmi = st.sidebar.slider("BMI", 10.0, 100.0, 25.0)
    smoking_status = st.sidebar.selectbox(
        "Smoking Status", ["smokes", "never smoked", "formerly smoked"]
    )

    # Store input data as a dictionary
    input_data = {
        "gender": gender,
        "age": age,
        "ever_married": ever_married,
        "heart_disease": heart_disease,
        "hypertension": hypertension,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Add a button to trigger prediction
    if st.sidebar.button("Predict"):
        # Make prediction
        processed_data = preprocess(input_df)
        prediction = make_prediction(processed_data)

        # Display prediction

        st.write("# Potential Stroke 🤒" if prediction else "# Clear 😊")


# Run the main function
if __name__ == "__main__":
    main()
