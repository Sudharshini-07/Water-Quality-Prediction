# -*- coding: utf-8 -*-
"""Water Quality Prediction App with Real-time Data"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Water Potability Prediction",
    layout="wide",
    page_icon="💧"
)

# Title and description
st.title("💧 Water Potability Prediction Dashboard")
st.markdown("""
Predict whether water is potable (safe to drink) based on its chemical properties. 
This app uses machine learning and real-time water quality data.
""")

# Data loading and preprocessing
@st.cache_data
def load_data():
    # Load the main training dataset
    df = pd.read_csv("water_potability.csv")
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_realtime_data():
    try:
        # California state code - US:06
        url = ("https://www.waterqualitydata.us/data/Result/search?"
               "statecode=US%3A06&"
               "characteristicType=Physical&"
               "characteristicType=Inorganics%2C%20Major%2C%20Metals&"
               "characteristicType=Inorganics%2C%20Major%2C%20Non-metals&"
               "siteType=Stream&"
               "startDateLo=01-01-2024&"
               "mimeType=csv&"
               "zip=no")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Convert bytes to string and then to DataFrame
            data = pd.read_csv(StringIO(response.text))
            
            # Pivot the data to get characteristics as columns
            pivoted_data = data.pivot_table(index=['MonitoringLocationIdentifier', 'ActivityStartDate'],
                                          columns='CharacteristicName',
                                          values='ResultMeasureValue',
                                          aggfunc='first').reset_index()
            
            # Convert columns to numeric and rename to match our model
            column_mapping = {
                'pH': 'ph',
                'Total hardness': 'Hardness',
                'Specific conductance': 'Conductivity',
                'Turbidity': 'Turbidity'
            }
            
            pivoted_data.rename(columns=column_mapping, inplace=True)
            
            numeric_cols = ['ph', 'Hardness', 'Conductivity', 'Turbidity']
            for col in numeric_cols:
                if col in pivoted_data.columns:
                    pivoted_data[col] = pd.to_numeric(pivoted_data[col], errors='coerce')
            
            return pivoted_data.dropna()
        else:
            st.error(f"API request failed with status {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching real-time data: {str(e)}")
        return None


data = load_data()

def preprocess(df):
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop('Potability', axis=1)
    y = df['Potability']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

X_scaled, y, scaler = preprocess(data)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Data Visualization Section
st.header("📊 Data Exploration and Insights")

# Visualization flashcards
viz_options = {
    "Potability Distribution": lambda: sns.countplot(x='Potability', data=data),
    "pH Distribution": lambda: sns.histplot(data['ph'], bins=20, kde=True, color='skyblue'),
    "Hardness Distribution": lambda: sns.boxplot(y=data['Hardness'], color='lightgreen'),
    "Solids Distribution": lambda: sns.violinplot(y=data['Solids'], color='salmon'),
    "Feature Correlation": lambda: sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
}

# Visualization selector
viz_choice = st.selectbox(
    "Select visualization:",
    list(viz_options.keys()),
    key='viz_selector'
)

# Display selected visualization
fig, ax = plt.subplots(figsize=(8, 5))
viz_options[viz_choice]()
plt.title(viz_choice)
if viz_choice == "Potability Distribution":
    ax.set_xticklabels(['Not Potable', 'Potable'])
st.pyplot(fig)

# Data summary statistics
with st.expander("📋 Show Data Statistics"):
    st.dataframe(data.describe())

# User Input Section
st.header("🔍 Water Quality Prediction")

tab1, tab2 = st.tabs(["Manual Input", "Real-time Data"])

with tab1:
    with st.form("manual_input_form"):
        st.subheader("Enter Water Parameters Manually")
        
        col1, col2 = st.columns(2)
        with col1:
            ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
            Hardness = st.slider("Hardness (mg/L)", 0.0, 500.0, 150.0, 1.0)
            Solids = st.slider("Solids (ppm)", 0.0, 50000.0, 10000.0, 100.0)
            Chloramines = st.slider("Chloramines (ppm)", 0.0, 20.0, 7.0, 0.1)
            Sulfate = st.slider("Sulfate (mg/L)", 0.0, 500.0, 330.0, 1.0)
        
        with col2:
            Conductivity = st.slider("Conductivity (μS/cm)", 0.0, 1000.0, 400.0, 1.0)
            Organic_carbon = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 10.0, 0.1)
            Trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 150.0, 66.0, 0.1)
            Turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 3.0, 0.1)
        
        submitted = st.form_submit_button("Predict Potability")

    if submitted:
        input_data = np.array([[ph, Hardness, Solids, Chloramines, Sulfate,
                              Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][prediction]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"✅ Water is **Potable** with {probability*100:.2f}% confidence.")
        else:
            st.error(f"❌ Water is **Not Potable** with {probability*100:.2f}% confidence.")

with tab2:
    st.subheader("Fetch Real-time Water Quality Data")
    st.markdown("""
    **Real-time Data Integration:**  
    This feature retrieves actual water quality measurements from USGS monitoring stations in California.
    The following parameters are fetched:
    - pH
    - Hardness
    - Conductivity
    - Turbidity
    
    Missing parameters are filled with median values from our training dataset.
    """)
    
    if st.button("🌍 Get Latest Water Quality Data", key="realtime_btn"):
        with st.spinner("Fetching real-time data from USGS..."):
            realtime_data = fetch_realtime_data()
        
        if realtime_data is not None and not realtime_data.empty:
            st.success("Successfully retrieved real-time data!")
            
            # Display the raw data
            with st.expander("View Raw API Data"):
                st.dataframe(realtime_data.head())
            
            # Process the first sample
            sample = {
                'ph': realtime_data['ph'].values[0] if 'ph' in realtime_data.columns else data['ph'].median(),
                'Hardness': realtime_data['Hardness'].values[0] if 'Hardness' in realtime_data.columns else data['Hardness'].median(),
                'Solids': data['Solids'].median(),
                'Chloramines': data['Chloramines'].median(),
                'Sulfate': data['Sulfate'].median(),
                'Conductivity': realtime_data['Conductivity'].values[0] if 'Conductivity' in realtime_data.columns else data['Conductivity'].median(),
                'Organic_carbon': data['Organic_carbon'].median(),
                'Trihalomethanes': data['Trihalomethanes'].median(),
                'Turbidity': realtime_data['Turbidity'].values[0] if 'Turbidity' in realtime_data.columns else data['Turbidity'].median()
            }
            
            # Show the processed sample
            st.subheader("Processed Sample for Prediction")
            sample_df = pd.DataFrame([sample])
            st.dataframe(sample_df.T.rename(columns={0: 'Value'}))
            
            # Highlight which values came from real-time data
            st.markdown("""
            <style>
                .real-time {
                    background-color: #e6f7ff !important;
                    font-weight: bold;
                }
            </style>
            """, unsafe_allow_html=True)
            
            realtime_cols = ['ph', 'Hardness', 'Conductivity', 'Turbidity']
            st.markdown("**Legend:** <span style='background-color:#e6f7ff; padding:2px 5px;'>Blue</span> = Real-time data | White = Estimated values", unsafe_allow_html=True)
            
            # Make prediction
            input_scaled = scaler.transform(np.array([list(sample.values())]))
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][prediction]
            
            # Display prediction
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"✅ Water is **Potable** with {probability*100:.2f}% confidence.")
            else:
                st.error(f"❌ Water is **Not Potable** with {probability*100:.2f}% confidence.")

# Model Evaluation Section
st.header("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Test Set Accuracy")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{accuracy*100:.2f}%")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'))
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': data.columns[:-1],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis', ax=ax)
    plt.title('Feature Importance')
    st.pyplot(fig)
