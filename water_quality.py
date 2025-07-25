# -*- coding: utf-8 -*-
"""Water Quality Prediction App with Real-time Data"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

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
    try:
        df = pd.read_csv("water_potability.csv")
        df.fillna(df.median(numeric_only=True), inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess(df):
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Real-time data fetching
def fetch_realtime_data():
    try:
        # USGS Instantaneous Values API
        params = {
            'format': 'json',
            'sites': '11447650',  # Sacramento River site
            'parameterCd': '00010,00400,00095',  # Temp, pH, Conductivity
            'siteStatus': 'all'
        }
        
        response = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params=params,
            timeout=15
        )
        response.raise_for_status()
        
        data = response.json()
        water_data = {}
        
        for series in data['value']['timeSeries']:
            param_code = series['variable']['variableCode'][0]['value']
            value = float(series['values'][0]['value'][0]['value'])
            
            if param_code == '00010':  # Temperature (not used but good to have)
                water_data['Temperature'] = value
            elif param_code == '00400':  # pH
                water_data['ph'] = value
            elif param_code == '00095':  # Specific conductance
                water_data['Conductivity'] = value
        
        # Default values for missing parameters
        defaults = {
            'ph': 7.0,
            'Hardness': 150.0,
            'Solids': 10000.0,
            'Chloramines': 7.0,
            'Sulfate': 330.0,
            'Conductivity': 400.0,
            'Organic_carbon': 10.0,
            'Trihalomethanes': 66.0,
            'Turbidity': 3.0
        }
        
        # Combine real data with defaults
        final_data = {**defaults, **water_data}
        return pd.DataFrame([final_data])
        
    except Exception as e:
        st.error(f"Error fetching real-time data: {str(e)}")
        return None

def generate_sample_data():
    """Fallback data generator"""
    return pd.DataFrame([{
        'ph': round(np.random.normal(7.0, 0.5), 1),
        'Hardness': np.random.randint(100, 300),
        'Solids': np.random.randint(5000, 20000),
        'Chloramines': round(np.random.normal(4.0, 1.0), 1),
        'Sulfate': np.random.randint(200, 400),
        'Conductivity': np.random.randint(200, 600),
        'Organic_carbon': round(np.random.normal(10.0, 3.0), 1),
        'Trihalomethanes': np.random.randint(20, 100),
        'Turbidity': round(np.random.normal(3.0, 1.0), 1)
    }])

def get_water_sample():
    """Get data with multiple fallback options"""
    real_data = fetch_realtime_data()
    if real_data is not None:
        return real_data
    
    st.warning("Using sample data as fallback")
    return generate_sample_data()

# Load and prepare data
data = load_data()
if data is None:
    st.error("Failed to load data. Please check your data file.")
    st.stop()

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

viz_options = {
    "Potability Distribution": lambda: sns.countplot(x='Potability', data=data),
    "pH Distribution": lambda: sns.histplot(data['ph'], bins=20, kde=True, color='skyblue'),
    "Hardness Distribution": lambda: sns.boxplot(y=data['Hardness'], color='lightgreen'),
    "Solids Distribution": lambda: sns.violinplot(y=data['Solids'], color='salmon'),
    "Feature Correlation": lambda: sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
}

viz_choice = st.selectbox("Select visualization:", list(viz_options.keys()))
fig, ax = plt.subplots(figsize=(8, 5))
viz_options[viz_choice]()
plt.title(viz_choice)
if viz_choice == "Potability Distribution":
    ax.set_xticklabels(['Not Potable', 'Potable'])
st.pyplot(fig)

# User Input Section
st.header("🔍 Water Quality Prediction")

tab1, tab2 = st.tabs(["Manual Input", "Real-time Data"])

with tab1:
    with st.form("manual_input_form"):
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
            st.success(f"✅ Potable ({probability*100:.2f}% confidence)")
        else:
            st.error(f"❌ Not Potable ({probability*100:.2f}% confidence)")

with tab2:
    st.subheader("Real-time Water Quality Data")
    if st.button("🌍 Get Latest Water Sample"):
        water_sample = get_water_sample()
        st.dataframe(water_sample)
        
        input_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        input_data = water_sample[input_cols].values
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][prediction]
        
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"✅ Potable ({probability*100:.2f}% confidence)")
        else:
            st.error(f"❌ Not Potable ({probability*100:.2f}% confidence)")

# Model Evaluation Section
st.header("📈 Model Performance")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Test Set Accuracy")
    y_pred = model.predict(X_test)
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    
    st.subheader("Confusion Matrix")
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': data.columns[:-1],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis', ax=ax)
    st.pyplot(fig)
